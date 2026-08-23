from __future__ import annotations

import argparse
import contextlib
import hashlib
from io import BytesIO
import os
from pathlib import Path
import stat
import sys
from time import perf_counter
from typing import BinaryIO, Literal
import uuid

import numpy as np
from PIL import Image

from gs_video.domain.errors import UnsupportedMaterialError
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.camera import MatrixCamera
from gs_video.scene.gsplat_renderer import GsplatRenderer, PreparedPreviewScene
from gs_video.scene.ply import load_gaussian_ply
from gs_video.scene.preview_protocol import (
    CompleteEvent,
    ErrorEvent,
    MAX_PREVIEW_MESSAGE_BYTES,
    OpenPreviewSessionRequest,
    PreviewTimingPayload,
    ReadyEvent,
    RenderLiveCommand,
    RenderPickCommand,
    encode_message,
    parse_command,
    read_open_request,
)
from gs_video.scene.worker import _bounded_message, _require_input_file, _wait_for_startup_gate
from gs_video.scene.worker_protocol import (
    MatrixCameraPayload,
    OrbitCameraPayload,
    assert_safe_directory,
    ensure_safe_directory,
)
from gs_video.segmentation.paths import has_reparse_component


class _SceneChangedError(ValueError):
    pass


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_nlink),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _sha256(path: Path, expected_identity: tuple[int, int, int, int, int, int]) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != expected_identity:
            raise _SceneChangedError("Gaussian scene identity changed before hashing")
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        if _identity(os.fstat(stream.fileno())) != expected_identity:
            raise _SceneChangedError("Gaussian scene identity changed while hashing")
    if _identity(path.lstat()) != expected_identity:
        raise _SceneChangedError("Gaussian scene path changed while hashing")
    return digest.hexdigest()


def _validate_open(request: OpenPreviewSessionRequest) -> tuple[int, int, int, int, int, int]:
    try:
        _require_input_file(request.scene_path, "Gaussian scene")
        metadata = request.scene_path.stat()
        identity = _identity(metadata)
        if (
            metadata.st_size != request.scene_size
            or _sha256(request.scene_path, identity) != request.scene_sha256
        ):
            raise _SceneChangedError(
                "Gaussian scene does not match its imported authority"
            )
    except _SceneChangedError:
        raise
    except (OSError, ValueError) as error:
        raise _SceneChangedError(
            "Gaussian scene changed before the preview session opened"
        ) from error
    ensure_safe_directory(request.output_root)
    if has_reparse_component(request.output_root):
        raise ValueError("preview output root contains a link or reparse point")
    return identity


def _validate_output(path: Path, root: Path) -> None:
    root_identity = ensure_safe_directory(root)
    if path.parent != root or path.exists() or path.is_symlink() or has_reparse_component(path):
        raise ValueError("preview output path is not a new file in the owned output root")
    assert_safe_directory(root, root_identity)


def _publish(temporary: Path, output: Path) -> None:
    metadata = temporary.lstat()
    if (
        has_reparse_component(temporary)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size <= 0
    ):
        raise ValueError("preview staging output is unsafe")
    os.replace(temporary, output)


def _camera(payload: OrbitCameraPayload | MatrixCameraPayload) -> OrbitCamera | MatrixCamera:
    if isinstance(payload, MatrixCameraPayload):
        return MatrixCamera(
            camera_to_world_matrix=np.asarray(payload.camera_to_world, dtype=np.float64),
            fov_y_degrees=payload.fov_y_degrees,
            intrinsics_matrix=(
                None
                if payload.intrinsics is None
                else np.asarray(payload.intrinsics, dtype=np.float64)
            ),
            source_size=payload.source_size,
        )
    return OrbitCamera(**payload.model_dump())


def _render_live(
    renderer: GsplatRenderer,
    prepared: PreparedPreviewScene,
    command: RenderLiveCommand,
    root: Path,
) -> PreviewTimingPayload:
    _validate_output(command.output_path, root)
    temporary = root / f".{command.output_path.name}.staging-{uuid.uuid4().hex}"
    try:
        rgb, render_timings = renderer.render_prepared_rgb_profiled(
            prepared, _camera(command.camera), command.width, command.height
        )
        jpeg_started = perf_counter()
        encoded = BytesIO()
        Image.fromarray(rgb).save(
            encoded, format="JPEG", quality=85, optimize=False
        )
        payload = encoded.getvalue()
        jpeg_ms = (perf_counter() - jpeg_started) * 1000
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, command.output_path)
        return PreviewTimingPayload(
            gpu_raster_ms=render_timings.gpu_raster_ms,
            readback_ms=render_timings.readback_ms,
            jpeg_ms=jpeg_ms,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _render_pick(
    renderer: GsplatRenderer,
    prepared: PreparedPreviewScene,
    command: RenderPickCommand,
    root: Path,
) -> None:
    _validate_output(command.output_path, root)
    temporary = root / f".{command.output_path.name}.staging-{uuid.uuid4().hex}"
    try:
        pick = renderer.render_prepared_pick(
            prepared, _camera(command.camera), command.width, command.height
        )
        with temporary.open("xb") as stream:
            np.savez(
                stream,
                rgb=pick.rgb,
                expected_depth=pick.expected_depth,
                opacity=pick.opacity,
            )
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, command.output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _serve(request: OpenPreviewSessionRequest, protocol: BinaryIO) -> None:
    identity = _validate_open(request)
    renderer = GsplatRenderer(
        available_vram_bytes=request.available_vram_limit_mb * 1024**2
    )
    try:
        scene = load_gaussian_ply(request.scene_path)
    except UnsupportedMaterialError:
        try:
            current = request.scene_path.stat()
            if (
                _identity(current) != identity
                or _sha256(request.scene_path, identity)
                != request.scene_sha256
            ):
                raise _SceneChangedError(
                    "Gaussian scene changed while opening the preview session"
                )
        except _SceneChangedError:
            raise
        except (OSError, ValueError) as authority_error:
            raise _SceneChangedError(
                "Gaussian scene became unavailable while opening the preview session"
            ) from authority_error
        raise
    prepared = renderer.prepare_preview(
        scene,
        width=request.maximum_width,
        height=request.maximum_height,
        sh_degree=request.sh_degree,
    )
    del scene
    renderer.render_prepared_rgb(
        prepared,
        _camera(request.initial_camera),
        request.maximum_width,
        request.maximum_height,
    )
    protocol.write(
        encode_message(
            ReadyEvent(
                type="ready",
                implementation_version=f"gsplat-{renderer._version(prepared.rasterizer)}",
            )
        )
    )
    protocol.flush()
    while payload := sys.stdin.buffer.readline(MAX_PREVIEW_MESSAGE_BYTES + 1):
        command = parse_command(payload)
        try:
            current = request.scene_path.stat()
        except OSError as error:
            raise _SceneChangedError(
                "Gaussian scene became unavailable during preview session"
            ) from error
        current_identity = _identity(current)
        if current_identity != identity:
            raise _SceneChangedError(
                "Gaussian scene identity changed during preview session"
            )
        timings = None
        if isinstance(command, RenderLiveCommand):
            timings = _render_live(
                renderer, prepared, command, request.output_root
            )
        elif isinstance(command, RenderPickCommand):
            _render_pick(renderer, prepared, command, request.output_root)
        else:
            raise ValueError("unsupported preview command")
        protocol.write(
            encode_message(
                CompleteEvent(
                    type="complete",
                    request_id=command.request_id,
                    output=command.output_path.name,
                    timings=timings,
                )
            )
        )
        protocol.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gs-video-preview-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--startup-gate", type=Path, required=True)
    arguments = parser.parse_args(argv)
    saved_stdout_fd = os.dup(1)
    protocol = os.fdopen(os.dup(saved_stdout_fd), "wb", closefd=True)
    os.dup2(2, 1)
    request_id: int | None = None
    try:
        request = read_open_request(arguments.request)
        _wait_for_startup_gate(arguments.startup_gate)
        with contextlib.redirect_stdout(sys.stderr):
            _serve(request, protocol)
        return 0
    except BaseException as error:
        message = str(error).lower()
        code: Literal["resource_exhausted", "scene_changed", "system_error"] = (
            "scene_changed"
            if isinstance(error, _SceneChangedError)
            else "resource_exhausted"
            if "out of memory" in message or "vram" in message
            else "system_error"
        )
        try:
            protocol.write(
                encode_message(
                    ErrorEvent(
                        type="error",
                        request_id=request_id,
                        code=code,
                        message=_bounded_message(error),
                    )
                )
            )
            protocol.flush()
        except BaseException:
            pass
        return 1
    finally:
        protocol.close()
        os.close(saved_stdout_fd)


if __name__ == "__main__":
    raise SystemExit(main())
