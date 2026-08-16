from __future__ import annotations

import argparse
import contextlib
import os
import stat
import sys
import time
import uuid
from dataclasses import dataclass
from math import radians, tan
from pathlib import Path
from typing import Any

from gs_video.scene.worker_protocol import (
    CompleteEvent,
    ErrorEvent,
    ProbeEvent,
    ProbeRequest,
    ProgressEvent,
    RenderPickRequest,
    RenderSequenceRequest,
    WorkerEvent,
    WorkerRequest,
    assert_safe_directory,
    encode_worker_event,
    ensure_safe_directory,
    read_worker_request,
)
from gs_video.segmentation.paths import has_reparse_component


_PROTOCOL_STREAM: Any | None = None


def _emit(event: WorkerEvent) -> None:
    stream = _PROTOCOL_STREAM or sys.stdout.buffer
    stream.write(encode_worker_event(event))
    stream.flush()


def _bounded_message(error: BaseException) -> str:
    message = " ".join(str(error).split())
    if not message:
        message = type(error).__name__
    return message[:512]


def _wait_for_startup_gate(path: Path) -> None:
    gate = Path(path).absolute()
    if has_reparse_component(gate):
        raise ValueError("startup gate contains a link or reparse point")
    before = gate.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 32:
        raise ValueError("startup gate must be an owned regular file")
    deadline = time.monotonic() + 30.0
    while True:
        try:
            value = gate.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("startup gate became unreadable") from exc
        if value == "RELEASE\n":
            return
        if value != "WAIT\n":
            raise ValueError("startup gate value is invalid")
        if time.monotonic() >= deadline:
            raise TimeoutError("startup gate was not released within 30 seconds")
        time.sleep(0.01)


def _require_input_file(path: Path, label: str) -> None:
    requested = Path(path).absolute()
    if has_reparse_component(requested):
        raise ValueError(f"{label} contains a link or reparse point")
    metadata = requested.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size <= 0:
        raise ValueError(f"{label} must be a non-empty owned regular file")


def _require_owned_output(path: Path, *, directory: bool) -> None:
    del directory
    requested = Path(path).absolute()
    parent = requested.parent
    parent_identity = ensure_safe_directory(parent)
    if requested.exists() or requested.is_symlink():
        raise ValueError("renderer output must not exist before rendering")
    assert_safe_directory(parent, parent_identity)


@dataclass(frozen=True)
class _MatrixCamera:
    camera_to_world_matrix: Any
    fov_y_degrees: float

    def view_matrix(self) -> Any:
        import numpy as np

        return np.asarray(np.linalg.inv(self.camera_to_world_matrix), dtype=np.float64)

    def intrinsics(self, width: int, height: int) -> Any:
        import numpy as np

        focal = 0.5 * height / tan(radians(self.fov_y_degrees) * 0.5)
        return np.asarray(
            [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def _render_sequence(request: RenderSequenceRequest) -> CompleteEvent:
    # Heavy renderer/model modules are intentionally imported only after the request
    # and startup gate have both been validated by main().
    from gs_video.camera.serialization import read_mapped_trajectory
    from gs_video.domain.contracts import RenderSettings
    from gs_video.pipeline.cancellation import CancellationToken
    from gs_video.scene.gsplat_renderer import GsplatRenderer
    from gs_video.scene.ply import load_gaussian_ply

    _require_input_file(request.scene_path, "Gaussian scene")
    _require_input_file(request.camera_manifest, "camera manifest")
    _require_owned_output(request.output_dir, directory=True)
    trajectory = read_mapped_trajectory(request.camera_manifest)
    cameras = tuple(
        _MatrixCamera(pose, trajectory.fov_y_degrees)
        for pose in trajectory.camera_to_world
    )
    renderer = GsplatRenderer()
    sequence = renderer.render(
        load_gaussian_ply(request.scene_path),
        cameras,
        request.output_dir,
        RenderSettings(
            width=request.width,
            height=request.height,
            sh_degree=request.sh_degree,
            background=request.background,
            preview_stride=request.preview_stride,
        ),
        lambda current, total, message: _emit(
            ProgressEvent(type="progress", current=current, total=total, message=message)
        ),
        CancellationToken(),
    )
    return CompleteEvent(
        type="complete",
        implementation_version=sequence.implementation_version,
        outputs=[path.name for path in sequence.frame_paths],
    )


def _render_pick(request: RenderPickRequest) -> CompleteEvent:
    import numpy as np

    from gs_video.scene.gsplat_renderer import GsplatRenderer
    from gs_video.scene.ply import load_gaussian_ply
    from gs_video.scene.worker_protocol import MatrixCameraPayload
    from gs_video.scene.synthesis_camera import MatrixCamera
    from gs_video.scene.camera import OrbitCamera

    _require_input_file(request.scene_path, "Gaussian scene")
    _require_owned_output(request.output_npz, directory=False)
    camera = (
        MatrixCamera(
            camera_to_world_matrix=np.asarray(
                request.camera.camera_to_world, dtype=np.float64
            ),
            fov_y_degrees=request.camera.fov_y_degrees,
        )
        if isinstance(request.camera, MatrixCameraPayload)
        else OrbitCamera(**request.camera.model_dump())
    )
    renderer = GsplatRenderer()
    pick = renderer.render_pick(
        load_gaussian_ply(request.scene_path), camera, request.width, request.height
    )
    temporary = request.output_npz.parent / (
        f".{request.output_npz.name}.staging-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, rgb=pick.rgb, expected_depth=pick.expected_depth)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = temporary.lstat()
        if (
            has_reparse_component(temporary)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
        ):
            raise ValueError("pick staging output is unsafe")
        os.replace(temporary, request.output_npz)
    finally:
        temporary.unlink(missing_ok=True)
    version = getattr(renderer._dependencies()[0], "version", "unknown")
    return CompleteEvent(
        type="complete",
        implementation_version=f"gsplat-{version}",
        outputs=[request.output_npz.name],
    )


def _probe(_request: ProbeRequest) -> ProbeEvent:
    import gsplat  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]

    torch_version = getattr(torch, "__version__", None)
    gsplat_version = getattr(gsplat, "__version__", None)
    if not isinstance(torch_version, str) or not torch_version:
        raise RuntimeError("torch version is unavailable")
    if not isinstance(gsplat_version, str) or not gsplat_version:
        raise RuntimeError("gsplat version is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    return ProbeEvent(
        type="probe",
        torch=torch_version,
        gsplat=gsplat_version,
        device="cuda",
        total_vram_mb=total_vram // (1024 * 1024),
        free_vram_mb=free_vram // (1024 * 1024),
    )


def _run_validated_request(request: WorkerRequest) -> WorkerEvent:
    if isinstance(request, RenderSequenceRequest):
        return _render_sequence(request)
    if isinstance(request, RenderPickRequest):
        return _render_pick(request)
    return _probe(request)


def main(argv: list[str] | None = None) -> int:
    global _PROTOCOL_STREAM
    parser = argparse.ArgumentParser(prog="gs-video-renderer-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--startup-gate", type=Path, required=True)
    arguments = parser.parse_args(argv)
    explicit_argv = argv is not None
    previous_protocol_stream = _PROTOCOL_STREAM
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout_fd = os.dup(1)
    protocol_stream = os.fdopen(os.dup(saved_stdout_fd), "wb", closefd=True)
    os.dup2(2, 1)
    _PROTOCOL_STREAM = protocol_stream
    try:
        request = read_worker_request(arguments.request)
        _wait_for_startup_gate(arguments.startup_gate)
        with contextlib.redirect_stdout(sys.stderr):
            terminal = _run_validated_request(request)
        _emit(terminal)
        return 0
    except BaseException as exc:
        from gs_video.domain.errors import UnsupportedMaterialError

        code = "unsupported_material" if isinstance(exc, UnsupportedMaterialError) else "system_error"
        try:
            _emit(ErrorEvent(type="error", code=code, message=_bounded_message(exc)))
        except BaseException:
            pass
        return 1 if code == "system_error" else 0
    finally:
        try:
            protocol_stream.flush()
        finally:
            _PROTOCOL_STREAM = previous_protocol_stream
            if explicit_argv:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_stdout_fd, 1)
            protocol_stream.close()
            os.close(saved_stdout_fd)


if __name__ == "__main__":
    raise SystemExit(main())
