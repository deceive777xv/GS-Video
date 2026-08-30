from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import queue
import subprocess
import threading
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from gs_video.domain.errors import GsVideoError
from gs_video.domain.models import EffectInstance
from gs_video.environment.vram import VramLimitProvider, resolve_vram_limit_mb
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.gpu import GpuAdmissionGate
from gs_video.scene.worker_protocol import (
    CompleteEvent,
    ErrorEvent,
    MAX_EVENT_BYTES,
    ProgressEvent,
    parse_worker_event,
)
from gs_video.segmentation.tree_guard import ProcessTreeGuard, create_process_tree_guard
from gs_video.postprocess.worker_protocol import ProcessSequenceRequest, write_request


class PostProcessWorkerError(GsVideoError):
    category = "post_process"
    retryable = True

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class _PreviewSession:
    process: Any
    guard: ProcessTreeGuard
    lines: queue.Queue[bytes | None]
    stderr_chunks: list[bytes]
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread


class PostProcessWorkerClient:
    identity = "isolated-cuda-worker-v1"

    def __init__(
        self,
        *,
        worker_prefix: tuple[str, ...],
        available_vram_limit_mb: int,
        vram_limit_provider: VramLimitProvider | None = None,
        gpu_gate: GpuAdmissionGate | None = None,
        process_factory: Any = subprocess.Popen,
        tree_guard_factory: Any = create_process_tree_guard,
    ) -> None:
        if not worker_prefix or any(not item for item in worker_prefix):
            raise ValueError("post-process worker prefix must contain nonempty argv")
        if PureWindowsPath(worker_prefix[0]).name.lower() in {"wsl", "wsl.exe"}:
            raise ValueError("WSL post-process workers are not supported")
        self.worker_prefix = tuple(worker_prefix)
        self.available_vram_limit_mb = available_vram_limit_mb
        self.vram_limit_provider = vram_limit_provider
        self.gpu_gate = gpu_gate
        self.process_factory = process_factory
        self.tree_guard_factory = tree_guard_factory
        self._active: dict[int, tuple[Any, ProcessTreeGuard]] = {}
        self._lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._preview: _PreviewSession | None = None

    def _command(self, request_path: Path) -> list[str]:
        return [
            *self.worker_prefix,
            "-m",
            "gs_video.postprocess.worker",
            "--request",
            str(request_path),
        ]

    def _session_command(self) -> list[str]:
        return [*self.worker_prefix, "-m", "gs_video.postprocess.worker", "--session"]

    @staticmethod
    def _process_options(*, stdin: bool = False) -> dict[str, object]:
        options: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if stdin:
            options["stdin"] = subprocess.PIPE
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            options["start_new_session"] = True
        return options

    @staticmethod
    def _reader(stream: Any, lines: queue.Queue[bytes | None]) -> None:
        try:
            while True:
                line = stream.readline(MAX_EVENT_BYTES + 1)
                if line in (b"", ""):
                    break
                payload = line.encode("utf-8") if isinstance(line, str) else bytes(line)
                lines.put(payload)
        finally:
            lines.put(None)

    @staticmethod
    def _stderr_reader(stream: Any, chunks: list[bytes]) -> None:
        retained = 0
        while retained < 1024 * 1024:
            chunk = stream.read(min(64 * 1024, 1024 * 1024 - retained))
            if chunk in (b"", ""):
                return
            payload = chunk.encode("utf-8", errors="replace") if isinstance(chunk, str) else bytes(chunk)
            chunks.append(payload)
            retained += len(payload)

    @staticmethod
    def _stop(process: Any, guard: ProcessTreeGuard) -> None:
        if process.poll() is None:
            if not guard.terminate(force=False):
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if not guard.terminate(force=True):
                    process.kill()
                process.wait()

    def _run(
        self,
        request: ProcessSequenceRequest,
        request_path: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> None:
        write_request(request_path, request)
        process = self.process_factory(
            self._command(request_path), **self._process_options()
        )
        try:
            guard = self.tree_guard_factory(process)
        except BaseException:
            process.kill()
            process.wait()
            raise
        active = (process, guard)
        with self._lock:
            self._active[id(process)] = active
        lines: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
        stderr_chunks: list[bytes] = []
        if process.stdout is None or process.stderr is None:
            self._stop(process, guard)
            raise PostProcessWorkerError(
                "postprocess_worker_failed", "post-process worker pipes are unavailable"
            )
        stdout_thread = threading.Thread(
            target=self._reader, args=(process.stdout, lines), daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._stderr_reader,
            args=(process.stderr, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        terminal: CompleteEvent | ErrorEvent | None = None
        current = 0
        try:
            while True:
                token.raise_if_cancelled()
                try:
                    line = lines.get(timeout=0.05)
                except queue.Empty:
                    if process.poll() is not None:
                        continue
                    continue
                if line is None:
                    break
                try:
                    event = parse_worker_event(line)
                except ValueError as error:
                    raise PostProcessWorkerError(
                        "postprocess_worker_protocol", "worker returned invalid JSONL"
                    ) from error
                if isinstance(event, ProgressEvent):
                    if event.total != len(request.frame_paths) or event.current != current + 1:
                        raise PostProcessWorkerError(
                            "postprocess_worker_protocol", "worker progress is invalid"
                        )
                    current = event.current
                    emit(event.current, event.total, event.message)
                elif isinstance(event, (CompleteEvent, ErrorEvent)):
                    if terminal is not None:
                        raise PostProcessWorkerError(
                            "postprocess_worker_protocol", "worker emitted two terminal events"
                        )
                    terminal = event
                else:
                    raise PostProcessWorkerError(
                        "postprocess_worker_protocol", "worker emitted an unsupported event"
                    )
            returncode = process.wait(timeout=5)
            if isinstance(terminal, ErrorEvent):
                raise PostProcessWorkerError(terminal.code, terminal.message)
            expected = [f"{index:06d}.png" for index in range(1, len(request.frame_paths) + 1)]
            if (
                returncode != 0
                or not isinstance(terminal, CompleteEvent)
                or terminal.outputs != expected
                or current != len(request.frame_paths)
            ):
                diagnostics = b"".join(stderr_chunks).decode("utf-8", errors="replace")
                raise PostProcessWorkerError(
                    "postprocess_worker_failed",
                    (diagnostics or "post-process worker exited without a valid result")[:512],
                )
        finally:
            try:
                self._stop(process, guard)
            finally:
                guard.close()
                with self._lock:
                    self._active.pop(id(process), None)
                stdout_thread.join(timeout=0.5)
                stderr_thread.join(timeout=0.5)
                process.stdout.close()
                process.stderr.close()
                request_path.unlink(missing_ok=True)

    def process_sequence(
        self,
        frame_paths: tuple[Path, ...],
        output_dir: Path,
        effects: list[EffectInstance],
        lut_paths: dict[str, Path],
        *,
        spatial_scale: float,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> None:
        request_path = output_dir / f".postprocess-request-{uuid4().hex}.json"
        request = ProcessSequenceRequest(
            frame_paths=list(frame_paths),
            output_dir=output_dir.absolute(),
            effects=effects,
            lut_paths=lut_paths,
            spatial_scale=spatial_scale,
            vram_limit_mb=resolve_vram_limit_mb(
                self.available_vram_limit_mb, self.vram_limit_provider
            ),
        )
        if self.gpu_gate is None:
            self._run(request, request_path, emit, token)
        else:
            with self.gpu_gate.hold(token):
                self._run(request, request_path, emit, token)

    def _close_preview_locked(self) -> None:
        session = self._preview
        self._preview = None
        if session is None:
            return
        try:
            self._stop(session.process, session.guard)
        finally:
            try:
                session.guard.close()
            finally:
                with self._lock:
                    self._active.pop(id(session.process), None)
                session.stdout_thread.join(timeout=0.5)
                session.stderr_thread.join(timeout=0.5)
                for stream in (
                    session.process.stdin,
                    session.process.stdout,
                    session.process.stderr,
                ):
                    if stream is not None:
                        stream.close()

    def _start_preview_locked(self) -> _PreviewSession:
        process = self.process_factory(
            self._session_command(), **self._process_options(stdin=True)
        )
        try:
            guard = self.tree_guard_factory(process)
        except BaseException:
            process.kill()
            process.wait()
            raise
        lines: queue.Queue[bytes | None] = queue.Queue(maxsize=128)
        stderr_chunks: list[bytes] = []
        if process.stdin is None or process.stdout is None or process.stderr is None:
            self._stop(process, guard)
            guard.close()
            raise PostProcessWorkerError(
                "postprocess_worker_failed", "post-process preview pipes are unavailable"
            )
        stdout_thread = threading.Thread(
            target=self._reader, args=(process.stdout, lines), daemon=True
        )
        stderr_thread = threading.Thread(
            target=self._stderr_reader,
            args=(process.stderr, stderr_chunks),
            daemon=True,
        )
        session = _PreviewSession(
            process,
            guard,
            lines,
            stderr_chunks,
            stdout_thread,
            stderr_thread,
        )
        with self._lock:
            self._active[id(process)] = (process, guard)
        stdout_thread.start()
        stderr_thread.start()
        self._preview = session
        try:
            try:
                line = lines.get(timeout=30)
            except queue.Empty as error:
                raise PostProcessWorkerError(
                    "postprocess_worker_failed", "post-process preview warmup timed out"
                ) from error
            if line is None:
                raise PostProcessWorkerError(
                    "postprocess_worker_failed", "post-process preview exited during warmup"
                )
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise PostProcessWorkerError(
                    "postprocess_worker_protocol", "preview worker returned invalid warmup JSON"
                ) from error
            if isinstance(payload, dict) and payload.get("type") == "error":
                try:
                    event = parse_worker_event(line)
                except ValueError as error:
                    raise PostProcessWorkerError(
                        "postprocess_worker_protocol",
                        "preview worker returned an invalid warmup error",
                    ) from error
                if isinstance(event, ErrorEvent):
                    raise PostProcessWorkerError(event.code, event.message)
            if (
                not isinstance(payload, dict)
                or payload.get("type") != "ready"
                or not isinstance(payload.get("implementation_version"), str)
            ):
                raise PostProcessWorkerError(
                    "postprocess_worker_protocol", "preview worker did not become ready"
                )
            return session
        except BaseException:
            self._close_preview_locked()
            raise

    def _preview_session_locked(self) -> _PreviewSession:
        if self._preview is None or self._preview.process.poll() is not None:
            self._close_preview_locked()
            return self._start_preview_locked()
        return self._preview

    def _run_preview_locked(
        self,
        request: ProcessSequenceRequest,
        request_path: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> None:
        write_request(request_path, request)
        try:
            session = self._preview_session_locked()
            assert session.process.stdin is not None
            command = json.dumps(
                {"request": str(request_path)},
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
            session.process.stdin.write(command)
            session.process.stdin.flush()
            terminal: CompleteEvent | ErrorEvent | None = None
            current = 0
            while terminal is None:
                token.raise_if_cancelled()
                try:
                    line = session.lines.get(timeout=0.05)
                except queue.Empty:
                    if session.process.poll() is not None:
                        raise PostProcessWorkerError(
                            "postprocess_worker_failed", "preview worker exited unexpectedly"
                        )
                    continue
                if line is None:
                    raise PostProcessWorkerError(
                        "postprocess_worker_failed", "preview worker closed its output"
                    )
                try:
                    event = parse_worker_event(line)
                except ValueError as error:
                    raise PostProcessWorkerError(
                        "postprocess_worker_protocol", "preview worker returned invalid JSONL"
                    ) from error
                if isinstance(event, ProgressEvent):
                    if event.total != len(request.frame_paths) or event.current != current + 1:
                        raise PostProcessWorkerError(
                            "postprocess_worker_protocol", "preview worker progress is invalid"
                        )
                    current = event.current
                    emit(event.current, event.total, event.message)
                elif isinstance(event, (CompleteEvent, ErrorEvent)):
                    terminal = event
                else:
                    raise PostProcessWorkerError(
                        "postprocess_worker_protocol", "preview worker event is unsupported"
                    )
            if isinstance(terminal, ErrorEvent):
                raise PostProcessWorkerError(terminal.code, terminal.message)
            expected = [
                f"{index:06d}.png"
                for index in range(1, len(request.frame_paths) + 1)
            ]
            if terminal.outputs != expected or current != len(request.frame_paths):
                raise PostProcessWorkerError(
                    "postprocess_worker_protocol", "preview worker result is invalid"
                )
        except BaseException:
            self._close_preview_locked()
            raise
        finally:
            request_path.unlink(missing_ok=True)

    def process_preview_sequence(
        self,
        frame_paths: tuple[Path, ...],
        output_dir: Path,
        effects: list[EffectInstance],
        lut_paths: dict[str, Path],
        *,
        spatial_scale: float,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> None:
        request_path = output_dir / f".postprocess-preview-{uuid4().hex}.json"
        request = ProcessSequenceRequest(
            frame_paths=list(frame_paths),
            output_dir=output_dir.absolute(),
            effects=effects,
            lut_paths=lut_paths,
            spatial_scale=spatial_scale,
            vram_limit_mb=resolve_vram_limit_mb(
                self.available_vram_limit_mb, self.vram_limit_provider
            ),
        )
        with self._preview_lock:
            self._run_preview_locked(request, request_path, emit, token)

    def close_preview(self) -> None:
        with self._preview_lock:
            self._close_preview_locked()

    def _terminate_all_sync(self) -> None:
        self.close_preview()
        with self._lock:
            active = tuple(self._active.values())
        for process, guard in active:
            try:
                self._stop(process, guard)
            except BaseException:
                continue

    async def terminate_all(self) -> None:
        await asyncio.to_thread(self._terminate_all_sync)
