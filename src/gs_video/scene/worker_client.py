from __future__ import annotations

import asyncio
import hashlib
import os
import queue
import stat
import subprocess
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, cast

import numpy as np
from PIL import Image

from gs_video.camera.serialization import read_mapped_trajectory
from gs_video.domain.contracts import PickBuffer, RenderSequence
from gs_video.domain.errors import GsVideoError, UnsupportedMaterialError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.gpu import GpuAdmissionGate
from gs_video.scene.worker_protocol import (
    CompleteEvent,
    ErrorEvent,
    MAX_EVENT_BYTES,
    ProbeEvent,
    ProbeRequest,
    ProgressEvent,
    RenderPickRequest,
    RenderSequenceRequest,
    WorkerEvent,
    WorkerRequest,
    assert_safe_directory,
    ensure_safe_directory,
    parse_worker_event,
    write_worker_request,
)
from gs_video.segmentation.paths import has_reparse_component, worker_path
from gs_video.segmentation.tree_guard import ProcessTreeGuard, create_process_tree_guard


MAX_DIAGNOSTIC_BYTES = 1024 * 1024
MAX_PICK_BYTES = 512 * 1024 * 1024
MAX_STDOUT_EVENTS = 64
PNG_OVERHEAD_BYTES = 64 * 1024


@dataclass(frozen=True)
class RendererWorkerIdentity:
    torch: str
    gsplat: str
    device: str
    total_vram_mb: int = 0
    free_vram_mb: int = 0


@dataclass(frozen=True)
class _ActiveWorker:
    process: Any
    guard: ProcessTreeGuard


@dataclass(frozen=True)
class _FrameSnapshot:
    identity: tuple[int, int, int, int, int, int]
    sha256: str


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_nlink),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


class RendererWorkerClient:
    def __init__(
        self,
        *,
        worker_prefix: tuple[str, ...],
        process_factory: Any = subprocess.Popen,
        tree_guard_factory: Any = create_process_tree_guard,
        log_path: Path | None = None,
        gpu_gate: GpuAdmissionGate | None = None,
    ) -> None:
        if not worker_prefix or any(not item for item in worker_prefix):
            raise ValueError("worker prefix must contain nonempty argv entries")
        executable_name = PureWindowsPath(worker_prefix[0]).name.lower()
        if executable_name in {"wsl", "wsl.exe"}:
            raise ValueError(
                "WSL renderer worker is unsupported until JSON payload paths are translated"
            )
        self.worker_prefix = tuple(worker_prefix)
        self._process_factory = process_factory
        self._tree_guard_factory = tree_guard_factory
        prefix_path = Path(worker_prefix[0])
        default_root = prefix_path.absolute().parent if prefix_path.parent != Path() else Path.cwd()
        self.log_path = Path(log_path or default_root / "logs" / "renderer-worker.log")
        self._active: set[_ActiveWorker] = set()
        self._active_lock = threading.Lock()
        self._gpu_gate = gpu_gate

    @staticmethod
    def _create_control_file(parent: Path, label: str, contents: bytes) -> Path:
        root = Path(parent).absolute()
        try:
            root_identity = ensure_safe_directory(root)
        except OSError as exc:
            raise GsVideoError(
                f"renderer {label} parent contains a link or reparse point"
            ) from exc
        path = root / f".gs-video-renderer-{label}-{uuid.uuid4().hex}"
        with path.open("xb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        assert_safe_directory(root, root_identity)
        return path

    @staticmethod
    def _release_gate(gate: Path) -> None:
        parent_identity = ensure_safe_directory(gate.parent)
        release = gate.parent / f".{gate.name}.release-{uuid.uuid4().hex}"
        try:
            with release.open("xb") as stream:
                stream.write(b"RELEASE\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(release, gate)
            assert_safe_directory(gate.parent, parent_identity)
        finally:
            release.unlink(missing_ok=True)

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def _request_parent(request: WorkerRequest, log_path: Path) -> Path:
        if isinstance(request, RenderSequenceRequest):
            return request.output_dir.parent
        if isinstance(request, RenderPickRequest):
            return request.output_npz.parent
        return log_path.parent.absolute()

    def _command(self, request_path: Path, startup_gate: Path) -> list[str]:
        return [
            *self.worker_prefix,
            "-m",
            "gs_video.scene.worker",
            "--request",
            worker_path(request_path, self.worker_prefix),
            "--startup-gate",
            worker_path(startup_gate, self.worker_prefix),
        ]

    def _popen_options(self) -> dict[str, object]:
        options: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
        }
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            options["start_new_session"] = True
        return options

    @staticmethod
    def _reap_direct_best_effort(process: Any) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            else:
                process.wait()
        except BaseException:
            pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def _start_worker(
        self, request_path: Path, startup_gate: Path
    ) -> tuple[Any, ProcessTreeGuard]:
        options = self._popen_options()
        process = self._process_factory(self._command(request_path, startup_gate), **options)
        try:
            guard = self._tree_guard_factory(process)
        except BaseException as exc:
            self._reap_direct_best_effort(process)
            raise GsVideoError("cannot establish renderer worker process-tree guard") from exc
        active = _ActiveWorker(process, guard)
        with self._active_lock:
            self._active.add(active)
        try:
            self._release_gate(startup_gate)
        except BaseException as exc:
            with self._active_lock:
                self._active.discard(active)
            try:
                guard.close()
            except BaseException:
                pass
            self._reap_direct_best_effort(process)
            raise GsVideoError("cannot release renderer worker startup gate") from exc
        return process, guard

    @staticmethod
    def _stdout_reader(
        stream: Any,
        target: queue.Queue[bytes | None],
        overflow: threading.Event,
    ) -> None:
        def enqueue(payload: bytes | None) -> bool:
            try:
                target.put(payload, timeout=0.05)
            except queue.Full:
                overflow.set()
                return False
            return True

        try:
            while True:
                chunk = stream.readline(MAX_EVENT_BYTES + 1)
                if chunk in (b"", ""):
                    return
                payload = chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
                if len(payload) > MAX_EVENT_BYTES or not payload.endswith(b"\n"):
                    if not overflow.is_set():
                        enqueue(b"x" * (MAX_EVENT_BYTES + 1))
                    while payload and not payload.endswith(b"\n"):
                        more = stream.readline(MAX_EVENT_BYTES + 1)
                        if more in (b"", ""):
                            break
                        payload = more.encode("utf-8") if isinstance(more, str) else bytes(more)
                    continue
                if not overflow.is_set():
                    enqueue(payload)
        except (OSError, ValueError):
            return
        finally:
            if not overflow.is_set():
                enqueue(None)

    @staticmethod
    def _stderr_reader(stream: Any, chunks: list[bytes], truncated: list[bool]) -> None:
        retained = 0
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if chunk in (b"", ""):
                    return
                payload = chunk.encode("utf-8", errors="replace") if isinstance(chunk, str) else bytes(chunk)
                available = MAX_DIAGNOSTIC_BYTES - retained
                if available > 0:
                    chunks.append(payload[:available])
                    retained += min(len(payload), available)
                if len(payload) > available:
                    truncated[0] = True
        except (OSError, ValueError):
            return

    @staticmethod
    def _stop(process: Any, guard: ProcessTreeGuard) -> None:
        if process.poll() is None:
            if not guard.terminate(force=False):
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tree_killed = guard.terminate(force=True)
                if not tree_killed or process.poll() is None:
                    process.kill()
                process.wait()
        else:
            process.wait()

    def _unregister(self, process: Any, guard: ProcessTreeGuard) -> None:
        with self._active_lock:
            self._active.discard(_ActiveWorker(process, guard))

    def _write_diagnostics(self, chunks: list[bytes], truncated: bool) -> None:
        payload = b"".join(chunks)
        marker = b"\n[renderer worker diagnostics truncated]\n"
        if truncated:
            payload = payload[: max(0, MAX_DIAGNOSTIC_BYTES - len(marker))] + marker
        try:
            parent_identity = ensure_safe_directory(self.log_path.parent)
        except OSError:
            return
        temporary = self.log_path.parent / (
            f".{self.log_path.name}.staging-{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload[:MAX_DIAGNOSTIC_BYTES])
                stream.flush()
                os.fsync(stream.fileno())
            metadata = temporary.lstat()
            if (
                has_reparse_component(temporary)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise OSError("renderer diagnostic staging file is unsafe")
            os.replace(temporary, self.log_path)
            assert_safe_directory(self.log_path.parent, parent_identity)
        finally:
            temporary.unlink(missing_ok=True)

    def _cleanup_process(
        self,
        process: Any,
        guard: ProcessTreeGuard,
        threads: tuple[threading.Thread | None, threading.Thread | None],
        stderr_chunks: list[bytes],
        stderr_truncated: list[bool],
    ) -> None:
        diagnostics_truncated = stderr_truncated[0]
        try:
            if process.poll() is None:
                self._stop(process, guard)
        except BaseException as exc:
            diagnostics_truncated = True
            stderr_chunks.append(f"\n[renderer cleanup error: {exc}]\n".encode())
        try:
            guard.close()
        except BaseException as exc:
            diagnostics_truncated = True
            stderr_chunks.append(f"\n[renderer tree guard close error: {exc}]\n".encode())
        self._unregister(process, guard)
        for thread in (item for item in threads if item is not None):
            thread.join(timeout=0.5)
            if thread.is_alive():
                diagnostics_truncated = True
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        try:
            self._write_diagnostics(stderr_chunks, diagnostics_truncated)
        except OSError:
            pass

    @staticmethod
    def _require_owned_file(path: Path, label: str, *, maximum: int | None = None) -> None:
        requested = Path(path).absolute()
        if has_reparse_component(requested):
            raise GsVideoError(f"{label} contains a link or reparse point")
        metadata = requested.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or (maximum is not None and metadata.st_size > maximum)
        ):
            raise GsVideoError(f"{label} must be an owned ordinary file")

    @staticmethod
    def _validate_before_start(request: WorkerRequest, log_path: Path) -> None:
        if isinstance(request, ProbeRequest):
            parent = log_path.parent.absolute()
            try:
                ensure_safe_directory(parent)
            except OSError as exc:
                raise GsVideoError(
                    "renderer log parent contains a link or reparse point"
                ) from exc
            return
        RendererWorkerClient._require_owned_file(request.scene_path, "Gaussian scene")
        inputs: tuple[Path, ...]
        if isinstance(request, RenderSequenceRequest):
            RendererWorkerClient._require_owned_file(request.camera_manifest, "camera manifest")
            output = request.output_dir
            inputs = (request.scene_path, request.camera_manifest)
        else:
            output = request.output_npz
            inputs = (request.scene_path,)
        parent = output.parent.absolute()
        try:
            ensure_safe_directory(parent)
        except OSError as exc:
            raise GsVideoError("renderer output parent contains a link or reparse point") from exc
        if output.exists() or output.is_symlink():
            raise GsVideoError("renderer output must be a new path beneath an ordinary parent")
        output_resolved = output.absolute()
        for input_path in inputs:
            input_resolved = input_path.resolve()
            if output_resolved == input_resolved or output_resolved in input_resolved.parents:
                raise GsVideoError("renderer output overlaps an input")

    @staticmethod
    def _wait_for_exit(process: Any, token: CancellationToken) -> int:
        while True:
            try:
                return cast(int, process.wait(timeout=0.05))
            except subprocess.TimeoutExpired:
                token.raise_if_cancelled()

    def _run(
        self,
        request: WorkerRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> tuple[WorkerEvent, int]:
        if self._gpu_gate is not None:
            with self._gpu_gate.hold(token):
                return self._run_admitted(request, emit, token)
        return self._run_admitted(request, emit, token)

    def _run_admitted(
        self,
        request: WorkerRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> tuple[WorkerEvent, int]:
        self._validate_before_start(request, self.log_path)
        parent = self._request_parent(request, self.log_path)
        request_path = parent / f".gs-video-renderer-request-{uuid.uuid4().hex}.json"
        gate = self._create_control_file(parent, "gate", b"WAIT\n")
        process: Any | None = None
        guard: ProcessTreeGuard | None = None
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        stderr_chunks: list[bytes] = []
        stderr_truncated = [False]
        try:
            write_worker_request(request_path, request)
            process, guard = self._start_worker(request_path, gate)
            if process.stdout is None or process.stderr is None:
                raise GsVideoError("renderer worker pipes are unavailable")
            lines: queue.Queue[bytes | None] = queue.Queue(maxsize=MAX_STDOUT_EVENTS)
            stdout_overflow = threading.Event()
            stdout_thread = threading.Thread(
                target=self._stdout_reader,
                args=(process.stdout, lines, stdout_overflow),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(process.stderr, stderr_chunks, stderr_truncated),
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            terminal: WorkerEvent | None = None
            last_current = 0
            expected_total: int | None = None
            if isinstance(request, RenderSequenceRequest):
                count = len(read_mapped_trajectory(request.camera_manifest).camera_to_world)
                expected_total = len(range(0, count, request.preview_stride))
            exit_drain_deadline: float | None = None
            while True:
                token.raise_if_cancelled()
                if stdout_overflow.is_set():
                    raise GsVideoError("renderer worker stdout queue overflow")
                if process.poll() is not None and exit_drain_deadline is None:
                    exit_drain_deadline = time.monotonic() + 1.0
                timeout = 0.05
                if exit_drain_deadline is not None:
                    remaining = exit_drain_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    timeout = min(timeout, remaining)
                try:
                    line = lines.get(timeout=timeout)
                except queue.Empty:
                    continue
                if line is None:
                    break
                try:
                    event = parse_worker_event(line)
                except ValueError as exc:
                    raise GsVideoError("renderer worker returned invalid JSONL") from exc
                if terminal is not None:
                    raise GsVideoError("renderer worker emitted an event after terminal")
                if isinstance(event, ProgressEvent):
                    if (
                        expected_total is None
                        or event.total != expected_total
                        or event.current != last_current + 1
                    ):
                        raise GsVideoError("renderer worker progress is non-monotonic or invalid")
                    last_current = event.current
                    emit(event.current, event.total, event.message)
                else:
                    terminal = event
            returncode = self._wait_for_exit(process, token)
            if terminal is None:
                raise GsVideoError(
                    f"renderer worker exited without terminal event: {returncode}"
                )
            if isinstance(terminal, ErrorEvent):
                if terminal.code == "unsupported_material" and returncode == 0:
                    raise UnsupportedMaterialError(terminal.message)
                if terminal.code == "system_error" and returncode != 0:
                    raise GsVideoError(terminal.message)
                raise GsVideoError("renderer worker error and exit status disagree")
            if returncode != 0:
                raise GsVideoError("renderer worker terminal and exit status disagree")
            if expected_total is not None and last_current != expected_total:
                raise GsVideoError("renderer worker completed before all progress events")
            return terminal, returncode
        finally:
            if process is not None and guard is not None:
                self._cleanup_process(
                    process,
                    guard,
                    (stdout_thread, stderr_thread),
                    stderr_chunks,
                    stderr_truncated,
                )
            self._safe_unlink(request_path)
            self._safe_unlink(gate)

    @staticmethod
    def _validate_png(path: Path, width: int, height: int) -> _FrameSnapshot:
        maximum = width * height * 4 + height + PNG_OVERHEAD_BYTES
        RendererWorkerClient._require_owned_file(
            path, "render frame size", maximum=maximum
        )
        before = _file_identity(path.lstat())
        try:
            with path.open("rb") as stream:
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("render frame identity changed before validation")
                with Image.open(stream) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
                        raise GsVideoError("render frame PNG format, mode, or dimensions are invalid")
                    image.load()
                stream.seek(0)
                digest = hashlib.sha256()
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("render frame identity changed during validation")
        except (OSError, ValueError) as exc:
            if isinstance(exc, GsVideoError):
                raise
            raise GsVideoError("render frame is unreadable") from exc
        if _file_identity(path.lstat()) != before:
            raise GsVideoError("render frame path identity changed during validation")
        return _FrameSnapshot(identity=before, sha256=digest.hexdigest())

    @staticmethod
    def _assert_frame_snapshot(path: Path, expected: _FrameSnapshot) -> None:
        try:
            before = _file_identity(path.lstat())
            if before != expected.identity or has_reparse_component(path):
                raise GsVideoError("render frame changed after validation")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                if _file_identity(os.fstat(stream.fileno())) != expected.identity:
                    raise GsVideoError("render frame changed after validation")
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                if _file_identity(os.fstat(stream.fileno())) != expected.identity:
                    raise GsVideoError("render frame changed after validation")
            if (
                _file_identity(path.lstat()) != expected.identity
                or digest.hexdigest() != expected.sha256
            ):
                raise GsVideoError("render frame changed after validation")
        except OSError as exc:
            raise GsVideoError("render frame changed after validation") from exc

    def render_sequence(
        self,
        request: RenderSequenceRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        terminal, _returncode = self._run(request, emit, token)
        if not isinstance(terminal, CompleteEvent):
            raise GsVideoError("renderer sequence returned the wrong terminal event")
        trajectory = read_mapped_trajectory(request.camera_manifest)
        source_indices = tuple(range(0, len(trajectory.camera_to_world), request.preview_stride))
        expected_names = tuple(f"{index + 1:06d}.png" for index in source_indices)
        if tuple(terminal.outputs) != expected_names:
            raise GsVideoError("renderer terminal inventory does not match requested frames")
        if not request.output_dir.is_dir() or has_reparse_component(request.output_dir):
            raise GsVideoError("renderer output directory is invalid")
        directory_before = request.output_dir.lstat()
        if not stat.S_ISDIR(directory_before.st_mode):
            raise GsVideoError("renderer output directory is not ordinary")
        directory_identity = _file_identity(directory_before)
        entries = tuple(sorted(request.output_dir.iterdir(), key=lambda item: item.name))
        if tuple(path.name for path in entries) != expected_names:
            raise GsVideoError("renderer output inventory differs from terminal event")
        snapshots = tuple(
            self._validate_png(path, request.width, request.height) for path in entries
        )
        for path, snapshot in zip(entries, snapshots, strict=True):
            self._assert_frame_snapshot(path, snapshot)
        directory_after = request.output_dir.lstat()
        if (
            has_reparse_component(request.output_dir)
            or not stat.S_ISDIR(directory_after.st_mode)
            or _file_identity(directory_after) != directory_identity
        ):
            raise GsVideoError("renderer output directory identity changed during validation")
        return RenderSequence(
            frame_dir=request.output_dir,
            frame_paths=entries,
            source_frame_indices=source_indices,
            width=request.width,
            height=request.height,
            implementation_version=terminal.implementation_version,
        )

    def render_pick(
        self, request: RenderPickRequest, token: CancellationToken
    ) -> PickBuffer:
        terminal, _returncode = self._run(request, lambda *_: None, token)
        if not isinstance(terminal, CompleteEvent) or terminal.outputs != [request.output_npz.name]:
            raise GsVideoError("renderer pick terminal inventory is invalid")
        self._require_owned_file(request.output_npz, "pick buffer", maximum=MAX_PICK_BYTES)
        before = _file_identity(request.output_npz.lstat())
        try:
            with request.output_npz.open("rb") as stream:
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("pick buffer identity changed before validation")
                try:
                    with zipfile.ZipFile(stream) as archive_file:
                        members = archive_file.infolist()
                except zipfile.BadZipFile as exc:
                    raise GsVideoError("pick buffer is not a valid NPZ archive") from exc
                names = tuple(member.filename for member in members)
                if len(members) != 3 or set(names) != {
                    "rgb.npy", "expected_depth.npy", "opacity.npy"
                }:
                    raise GsVideoError("pick buffer member inventory is invalid")
                member_limits = {
                    "rgb.npy": request.width * request.height * 3 + 4096,
                    "expected_depth.npy": request.width * request.height * 4 + 4096,
                    "opacity.npy": request.width * request.height * 4 + 4096,
                }
                if any(
                    member.flag_bits & 0x1
                    or member.file_size <= 0
                    or member.file_size > member_limits[member.filename]
                    for member in members
                ):
                    raise GsVideoError("pick buffer member size or flags are invalid")
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as archive:
                    if set(archive.files) != {"rgb", "expected_depth", "opacity"}:
                        raise GsVideoError("pick buffer fields are invalid")
                    rgb = archive["rgb"]
                    depth = archive["expected_depth"]
                    opacity = archive["opacity"]
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("pick buffer identity changed during validation")
        except (OSError, ValueError) as exc:
            if isinstance(exc, GsVideoError):
                raise
            raise GsVideoError("pick buffer is unreadable") from exc
        if _file_identity(request.output_npz.lstat()) != before:
            raise GsVideoError("pick buffer path identity changed during validation")
        if rgb.dtype != np.uint8 or rgb.shape != (request.height, request.width, 3):
            raise GsVideoError("pick RGB dtype or dimensions are invalid")
        if depth.dtype != np.float32 or depth.shape != (request.height, request.width):
            raise GsVideoError("pick depth dtype or dimensions are invalid")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise GsVideoError("pick depth must contain finite nonnegative values")
        if opacity.dtype != np.float32 or opacity.shape != (request.height, request.width):
            raise GsVideoError("pick opacity dtype or dimensions are invalid")
        if not np.isfinite(opacity).all() or np.any((opacity < 0) | (opacity > 1)):
            raise GsVideoError("pick opacity must remain in the unit interval")
        return PickBuffer(
            rgb=np.ascontiguousarray(rgb),
            expected_depth=np.ascontiguousarray(depth),
            opacity=np.ascontiguousarray(opacity),
        )

    def probe(
        self,
        request: ProbeRequest | None = None,
        token: CancellationToken | None = None,
    ) -> RendererWorkerIdentity:
        terminal, _returncode = self._run(
            request or ProbeRequest(type="probe"),
            lambda *_: None,
            token or CancellationToken(),
        )
        if not isinstance(terminal, ProbeEvent):
            raise GsVideoError("renderer probe returned the wrong terminal event")
        return RendererWorkerIdentity(
            torch=terminal.torch,
            gsplat=terminal.gsplat,
            device=terminal.device,
            total_vram_mb=terminal.total_vram_mb,
            free_vram_mb=terminal.free_vram_mb,
        )

    def _terminate_all_sync(self) -> None:
        with self._active_lock:
            workers = tuple(self._active)
        for worker in workers:
            try:
                self._stop(worker.process, worker.guard)
            except BaseException:
                continue

    async def terminate_all(self) -> None:
        await asyncio.to_thread(self._terminate_all_sync)
