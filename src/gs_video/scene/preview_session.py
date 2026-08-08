from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import queue
import stat
import subprocess
import threading
from typing import Any
import uuid
import zipfile

import numpy as np
from PIL import Image

from gs_video.domain.contracts import PickBuffer
from gs_video.domain.errors import GsVideoError
from gs_video.domain.models import SceneSummary
from gs_video.environment.vram import (
    VramLimitProvider,
    resolve_vram_limit_mb,
    validated_vram_limit_mb,
)
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.preview_protocol import (
    CompleteEvent,
    ErrorEvent,
    OpenPreviewSessionRequest,
    PreviewTimingPayload,
    ReadyEvent,
    RenderLiveCommand,
    RenderPickCommand,
    encode_message,
    parse_event,
    write_open_request,
)
from gs_video.scene.worker_client import (
    MAX_PICK_BYTES,
    RendererWorkerClient,
    _file_identity,
)
from gs_video.scene.worker_protocol import OrbitCameraPayload
from gs_video.segmentation.paths import has_reparse_component, worker_path


STARTUP_EVENT_TIMEOUT_SECONDS = 30.0
RENDER_EVENT_TIMEOUT_SECONDS = 10.0


@dataclass
class _SessionProcess:
    key: tuple[object, ...]
    process: Any
    guard: Any
    request_path: Path
    startup_gate: Path
    stderr_thread: threading.Thread
    stderr_chunks: list[bytes]
    stderr_truncated: list[bool]
    gpu_context: AbstractContextManager[None]


class PreviewRequestSuperseded(GsVideoError):
    """A pending live request was replaced before it reached the renderer."""


class PreviewResourceError(GsVideoError):
    pass


class PreviewSceneChangedError(GsVideoError):
    pass


class PreviewSceneUnavailableError(GsVideoError):
    pass


class PreviewSession(RendererWorkerClient):
    """Keep one validated Gaussian scene resident in an isolated CUDA worker."""

    def __init__(
        self,
        *,
        worker_prefix: tuple[str, ...],
        sh_degree: int,
        available_vram_limit_mb: int,
        vram_limit_provider: VramLimitProvider | None = None,
        process_factory: Any = subprocess.Popen,
        tree_guard_factory: Any = None,
        log_path: Path | None = None,
        gpu_gate: Any = None,
        idle_timeout_seconds: float = 30.0,
    ) -> None:
        if type(sh_degree) is not int or not 0 <= sh_degree <= 3:
            raise ValueError("sh_degree must be between 0 and 3")
        validated_vram_limit_mb(available_vram_limit_mb)
        if (
            isinstance(idle_timeout_seconds, bool)
            or not isinstance(idle_timeout_seconds, (int, float))
            or not 0 < idle_timeout_seconds <= 300
        ):
            raise ValueError("idle_timeout_seconds must be between 0 and 300")
        options: dict[str, object] = {
            "worker_prefix": worker_prefix,
            "process_factory": process_factory,
            "log_path": log_path,
            "gpu_gate": gpu_gate,
        }
        if tree_guard_factory is not None:
            options["tree_guard_factory"] = tree_guard_factory
        super().__init__(**options)  # type: ignore[arg-type]
        self._sh_degree = sh_degree
        self._available_vram_limit_mb = available_vram_limit_mb
        self._vram_limit_provider = vram_limit_provider
        self._session_lock = threading.RLock()
        self._session: _SessionProcess | None = None
        self._internal_request_id = 0
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._idle_timer: threading.Timer | None = None
        self._schedule = threading.Condition()
        self._render_running = False
        self._authority_waiters = 0
        self._pending_live: object | None = None
        self._closing = False
        self._schedule_generation = 0
        self._last_live_timings: PreviewTimingPayload | None = None

    @property
    def last_live_timings(self) -> PreviewTimingPayload | None:
        timings = self._last_live_timings
        return None if timings is None else timings.model_copy(deep=True)

    def _begin_live(self) -> None:
        ticket = object()
        with self._schedule:
            generation = self._schedule_generation
            self._pending_live = ticket
            self._schedule.notify_all()
            while True:
                if (
                    generation != self._schedule_generation
                    or self._pending_live is not ticket
                ):
                    raise PreviewRequestSuperseded(
                        "pending live preview was replaced by a newer request"
                    )
                if (
                    not self._closing
                    and not self._render_running
                    and self._authority_waiters == 0
                ):
                    self._pending_live = None
                    self._render_running = True
                    return
                self._schedule.wait()

    def _begin_authoritative(self) -> None:
        with self._schedule:
            generation = self._schedule_generation
            self._authority_waiters += 1
            try:
                while self._closing or self._render_running:
                    if generation != self._schedule_generation:
                        raise PreviewRequestSuperseded(
                            "authoritative preview was cancelled by session close"
                        )
                    self._schedule.wait()
                if generation != self._schedule_generation:
                    raise PreviewRequestSuperseded(
                        "authoritative preview was cancelled by session close"
                    )
                self._render_running = True
            finally:
                self._authority_waiters -= 1

    def _finish_render(self) -> None:
        with self._schedule:
            self._render_running = False
            self._schedule.notify_all()

    def _cancel_idle_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _arm_idle_locked(self) -> None:
        self._cancel_idle_locked()
        timer = threading.Timer(self._idle_timeout_seconds, self.close)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _command(self, request_path: Path, startup_gate: Path) -> list[str]:
        return [
            *self.worker_prefix,
            "-m",
            "gs_video.scene.preview_worker",
            "--request",
            worker_path(request_path, self.worker_prefix),
            "--startup-gate",
            worker_path(startup_gate, self.worker_prefix),
        ]

    def _popen_options(self) -> dict[str, object]:
        options = super()._popen_options()
        options["stdin"] = subprocess.PIPE
        return options

    @staticmethod
    def _scene_authority(
        project_root: Path, scene_path: str, summary: SceneSummary
    ) -> tuple[Path, Path, tuple[object, ...]]:
        try:
            root = Path(project_root).resolve(strict=True)
            scene = (root / scene_path).resolve(strict=True)
            source_root = (root / "source").resolve(strict=True)
            previews = (root / "previews").resolve(strict=True)
            metadata = scene.stat()
        except OSError as error:
            raise PreviewSceneUnavailableError(
                "Gaussian scene is unavailable for preview"
            ) from error
        if (
            not scene.is_relative_to(source_root)
            or not scene.is_file()
            or has_reparse_component(scene)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise PreviewSceneUnavailableError(
                "Gaussian scene is unavailable for preview"
            )
        if metadata.st_size != summary.size or scene.name != summary.filename:
            raise PreviewSceneChangedError(
                "Gaussian scene no longer matches its imported authority"
            )
        fingerprint = _file_identity(metadata)
        return scene, previews, (
            scene,
            summary.sha256,
            summary.size,
            fingerprint,
        )

    @staticmethod
    def _camera_payload(camera: OrbitCamera) -> OrbitCameraPayload:
        return OrbitCameraPayload(
            target=camera.target,
            distance=camera.distance,
            yaw=camera.yaw,
            pitch=camera.pitch,
            fov_y_degrees=camera.fov_y_degrees,
        )

    def _read_event(
        self, process: Any, *, timeout_seconds: float
    ) -> ReadyEvent | CompleteEvent | ErrorEvent:
        if process.stdout is None:
            raise GsVideoError("preview worker stdout is unavailable")
        result: queue.Queue[bytes | str | BaseException] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                result.put_nowait(process.stdout.readline(64 * 1024 + 1))
            except BaseException as error:
                result.put_nowait(error)

        reader = threading.Thread(target=read_line, daemon=True)
        reader.start()
        try:
            payload = result.get(timeout=timeout_seconds)
        except queue.Empty as error:
            raise GsVideoError("preview worker event timed out") from error
        reader.join(timeout=0.1)
        if isinstance(payload, BaseException):
            raise GsVideoError("preview worker event pipe failed") from payload
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not payload or len(payload) > 64 * 1024 or not payload.endswith(b"\n"):
            raise GsVideoError("preview worker returned an invalid bounded event")
        try:
            return parse_event(payload)
        except (ValueError, TypeError) as error:
            raise GsVideoError("preview worker returned invalid JSONL") from error

    @staticmethod
    def _raise_worker_error(event: ErrorEvent) -> None:
        if event.code == "scene_changed":
            raise PreviewSceneChangedError(event.message)
        if event.code == "resource_exhausted":
            raise PreviewResourceError(event.message)
        raise GsVideoError(event.message)

    def _start_session(
        self,
        scene: Path,
        previews: Path,
        summary: SceneSummary,
        key: tuple[object, ...],
        camera: OrbitCamera,
    ) -> _SessionProcess:
        request_path = previews / f".preview-session-open-{uuid.uuid4().hex}.json"
        startup_gate = self._create_control_file(previews, "preview-gate", b"WAIT\n")
        write_open_request(
            request_path,
            OpenPreviewSessionRequest(
                type="open_preview_session",
                scene_path=scene,
                output_root=previews,
                scene_size=summary.size,
                scene_sha256=summary.sha256,
                sh_degree=self._sh_degree,
                maximum_width=960,
                maximum_height=540,
                available_vram_limit_mb=resolve_vram_limit_mb(
                    self._available_vram_limit_mb,
                    self._vram_limit_provider,
                ),
                initial_camera=self._camera_payload(camera),
            ),
        )
        gpu_context: AbstractContextManager[None] = (
            self._gpu_gate.hold(CancellationToken())
            if self._gpu_gate is not None
            else nullcontext()
        )
        process: Any | None = None
        guard: Any | None = None
        stderr_thread: threading.Thread | None = None
        stderr_chunks: list[bytes] = []
        stderr_truncated = [False]
        try:
            gpu_context.__enter__()
            process, guard = self._start_worker(request_path, startup_gate)
            if process.stdin is None or process.stderr is None:
                raise GsVideoError("preview worker pipes are unavailable")
            stderr_thread = threading.Thread(
                target=self._stderr_reader,
                args=(process.stderr, stderr_chunks, stderr_truncated),
                daemon=True,
            )
            stderr_thread.start()
            event = self._read_event(
                process, timeout_seconds=STARTUP_EVENT_TIMEOUT_SECONDS
            )
            if not isinstance(event, ReadyEvent):
                if isinstance(event, ErrorEvent):
                    self._raise_worker_error(event)
                raise GsVideoError("preview worker did not complete its ready handshake")
            return _SessionProcess(
                key=key,
                process=process,
                guard=guard,
                request_path=request_path,
                startup_gate=startup_gate,
                stderr_thread=stderr_thread,
                stderr_chunks=stderr_chunks,
                stderr_truncated=stderr_truncated,
                gpu_context=gpu_context,
            )
        except BaseException:
            if process is not None and guard is not None:
                self._cleanup_process(
                    process,
                    guard,
                    (None, stderr_thread),
                    stderr_chunks,
                    stderr_truncated,
                )
            gpu_context.__exit__(None, None, None)
            self._safe_unlink(request_path)
            self._safe_unlink(startup_gate)
            raise

    def _ensure_session(
        self,
        scene: Path,
        previews: Path,
        summary: SceneSummary,
        key: tuple[object, ...],
        camera: OrbitCamera,
    ) -> _SessionProcess:
        current = self._session
        if current is not None and (
            current.key != key or current.process.poll() is not None
        ):
            self._close_locked()
            current = None
        if current is None:
            current = self._start_session(scene, previews, summary, key, camera)
            self._session = current
        return current

    def _send(
        self,
        session: _SessionProcess,
        command: RenderLiveCommand | RenderPickCommand,
    ) -> CompleteEvent:
        if session.process.stdin is None:
            raise GsVideoError("preview worker stdin is unavailable")
        try:
            session.process.stdin.write(encode_message(command))
            session.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as error:
            raise GsVideoError("preview worker command pipe failed") from error
        event = self._read_event(
            session.process, timeout_seconds=RENDER_EVENT_TIMEOUT_SECONDS
        )
        if isinstance(event, ErrorEvent):
            self._raise_worker_error(event)
        if (
            not isinstance(event, CompleteEvent)
            or event.request_id != command.request_id
            or event.output != command.output_path.name
        ):
            raise GsVideoError("preview worker completion does not match its command")
        return event

    @staticmethod
    def _read_live(path: Path, width: int, height: int) -> bytes:
        RendererWorkerClient._require_owned_file(
            path,
            "live preview JPEG",
            maximum=width * height * 3 + 64 * 1024,
        )
        before = _file_identity(path.lstat())
        try:
            with path.open("rb") as stream:
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("live preview JPEG identity changed before validation")
                payload = stream.read()
                stream.seek(0)
                with Image.open(stream) as image:
                    if (
                        image.format != "JPEG"
                        or image.mode != "RGB"
                        or image.size != (width, height)
                    ):
                        raise GsVideoError(
                            "live preview JPEG format or dimensions are invalid"
                        )
                    image.load()
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("live preview JPEG identity changed during validation")
        except (OSError, ValueError) as error:
            if isinstance(error, GsVideoError):
                raise
            raise GsVideoError("live preview JPEG is unreadable") from error
        if _file_identity(path.lstat()) != before or not payload.startswith(b"\xff\xd8"):
            raise GsVideoError("live preview JPEG identity changed during validation")
        return payload

    @staticmethod
    def _read_pick(path: Path, width: int, height: int) -> PickBuffer:
        RendererWorkerClient._require_owned_file(path, "preview pick buffer", maximum=MAX_PICK_BYTES)
        before = _file_identity(path.lstat())
        try:
            with path.open("rb") as stream:
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("preview pick identity changed before validation")
                try:
                    with zipfile.ZipFile(stream) as archive_file:
                        members = archive_file.infolist()
                except zipfile.BadZipFile as error:
                    raise GsVideoError("preview pick buffer is not a valid NPZ") from error
                limits = {
                    "rgb.npy": width * height * 3 + 4096,
                    "expected_depth.npy": width * height * 4 + 4096,
                }
                if (
                    len(members) != 2
                    or {member.filename for member in members} != set(limits)
                    or any(
                        member.flag_bits & 0x1
                        or member.file_size <= 0
                        or member.file_size > limits[member.filename]
                        for member in members
                    )
                ):
                    raise GsVideoError("preview pick NPZ member inventory is invalid")
                stream.seek(0)
                with np.load(stream, allow_pickle=False) as archive:
                    if set(archive.files) != {"rgb", "expected_depth"}:
                        raise GsVideoError("preview pick fields are invalid")
                    rgb = archive["rgb"]
                    depth = archive["expected_depth"]
                if _file_identity(os.fstat(stream.fileno())) != before:
                    raise GsVideoError("preview pick identity changed during validation")
        except (OSError, ValueError) as error:
            if isinstance(error, GsVideoError):
                raise
            raise GsVideoError("preview pick buffer is unreadable") from error
        if _file_identity(path.lstat()) != before:
            raise GsVideoError("preview pick path identity changed during validation")
        if rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise GsVideoError("preview pick RGB dimensions are invalid")
        if depth.dtype != np.float32 or depth.shape != (height, width):
            raise GsVideoError("preview pick depth dimensions are invalid")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise GsVideoError("preview pick depth is invalid")
        return PickBuffer(
            rgb=np.ascontiguousarray(rgb),
            expected_depth=np.ascontiguousarray(depth),
        )

    def _render_live_serial(
        self,
        project_root: Path,
        scene_path: str,
        scene_summary: SceneSummary,
        request_id: int,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> bytes:
        if type(request_id) is not int or request_id < 1:
            raise ValueError("request_id must be a positive integer")
        with self._session_lock:
            self._cancel_idle_locked()
            scene, previews, key = self._scene_authority(
                project_root, scene_path, scene_summary
            )
            output = previews / f".live-preview-{uuid.uuid4().hex}.jpg"
            try:
                command = RenderLiveCommand(
                    type="render_live",
                    request_id=request_id,
                    output_path=output,
                    camera=self._camera_payload(camera),
                    width=width,
                    height=height,
                )
                for attempt in range(2):
                    try:
                        session = self._ensure_session(
                            scene, previews, scene_summary, key, camera
                        )
                        complete = self._send(session, command)
                        self._last_live_timings = complete.timings
                        break
                    except (PreviewResourceError, PreviewSceneChangedError):
                        raise
                    except GsVideoError:
                        self._close_locked()
                        self._safe_unlink(output)
                        if attempt == 1:
                            raise
                        session = self._ensure_session(
                            scene, previews, scene_summary, key, camera
                        )
                payload = self._read_live(output, width, height)
                self._arm_idle_locked()
                return payload
            except BaseException:
                self._close_locked()
                raise
            finally:
                self._safe_unlink(output)

    def render_live(
        self,
        project_root: Path,
        scene_path: str,
        scene_summary: SceneSummary,
        request_id: int,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> bytes:
        self._begin_live()
        try:
            return self._render_live_serial(
                project_root,
                scene_path,
                scene_summary,
                request_id,
                camera,
                width,
                height,
            )
        finally:
            self._finish_render()

    def _render_preview_pick_serial(
        self,
        project_root: Path,
        scene_path: str,
        scene_summary: SceneSummary,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer:
        with self._session_lock:
            self._cancel_idle_locked()
            scene, previews, key = self._scene_authority(
                project_root, scene_path, scene_summary
            )
            self._internal_request_id = max(self._internal_request_id + 1, 1)
            output = previews / f".preview-pick-{uuid.uuid4().hex}.npz"
            try:
                command = RenderPickCommand(
                    type="render_pick",
                    request_id=self._internal_request_id,
                    output_path=output,
                    camera=self._camera_payload(camera),
                    width=width,
                    height=height,
                )
                for attempt in range(2):
                    try:
                        session = self._ensure_session(
                            scene, previews, scene_summary, key, camera
                        )
                        self._send(session, command)
                        break
                    except (PreviewResourceError, PreviewSceneChangedError):
                        raise
                    except GsVideoError:
                        self._close_locked()
                        self._safe_unlink(output)
                        if attempt == 1:
                            raise
                        session = self._ensure_session(
                            scene, previews, scene_summary, key, camera
                        )
                buffer = self._read_pick(output, width, height)
                self._arm_idle_locked()
                return buffer
            except BaseException:
                self._close_locked()
                raise
            finally:
                self._safe_unlink(output)

    def render_preview_pick(
        self,
        project_root: Path,
        scene_path: str,
        scene_summary: SceneSummary,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer:
        self._begin_authoritative()
        try:
            return self._render_preview_pick_serial(
                project_root,
                scene_path,
                scene_summary,
                camera,
                width,
                height,
            )
        finally:
            self._finish_render()

    def _close_locked(self) -> None:
        self._cancel_idle_locked()
        session = self._session
        self._session = None
        if session is None:
            return
        try:
            if session.process.stdin is not None:
                try:
                    session.process.stdin.close()
                except (OSError, ValueError):
                    pass
            self._cleanup_process(
                session.process,
                session.guard,
                (None, session.stderr_thread),
                session.stderr_chunks,
                session.stderr_truncated,
            )
        finally:
            session.gpu_context.__exit__(None, None, None)
            self._safe_unlink(session.request_path)
            self._safe_unlink(session.startup_gate)

    def close(self) -> None:
        with self._schedule:
            self._closing = True
            self._schedule_generation += 1
            self._pending_live = None
            self._schedule.notify_all()
            while self._render_running:
                self._schedule.wait()
        try:
            with self._session_lock:
                self._close_locked()
        finally:
            with self._schedule:
                self._closing = False
                self._schedule.notify_all()

    async def terminate_all(self) -> None:
        await asyncio.to_thread(self.close)
        await super().terminate_all()


__all__ = [
    "PreviewRequestSuperseded",
    "PreviewResourceError",
    "PreviewSceneChangedError",
    "PreviewSceneUnavailableError",
    "PreviewSession",
]
