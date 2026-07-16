from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Protocol
from uuid import uuid4

import numpy as np
from PIL import Image

from gs_video.api.schemas import ApiError
from gs_video.domain.contracts import PickBuffer
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.gsplat_renderer import GsplatRenderer
from gs_video.scene.ply import load_gaussian_ply
from gs_video.segmentation.paths import has_reparse_component


MAX_PREVIEW_ARTIFACT_BYTES = 16 * 1024 * 1024


def validate_pick_buffer(
    buffer: PickBuffer, *, width: int, height: int
) -> None:
    valid = (
        isinstance(buffer.rgb, np.ndarray)
        and buffer.rgb.shape == (height, width, 3)
        and buffer.rgb.dtype == np.uint8
        and buffer.rgb.flags.c_contiguous
        and isinstance(buffer.expected_depth, np.ndarray)
        and buffer.expected_depth.shape == (height, width)
        and buffer.expected_depth.dtype == np.float32
        and buffer.expected_depth.flags.c_contiguous
        and np.isfinite(buffer.expected_depth).all()
        and np.all(buffer.expected_depth >= 0.0)
    )
    if not valid:
        raise ApiError(
            500,
            code="invalid_preview",
            category="render",
            message="The preview renderer returned an invalid pick buffer.",
        )


class PreviewServiceLike(Protocol):
    def render_pick(
        self,
        project_root: Path,
        scene_path: str,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer: ...


class GsplatPreviewService:
    def __init__(self, renderer: GsplatRenderer | None = None) -> None:
        self._renderer = renderer or GsplatRenderer()

    def render_pick(
        self,
        project_root: Path,
        scene_path: str,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer:
        root = project_root.resolve()
        try:
            scene = (root / scene_path).resolve(strict=True)
        except OSError as error:
            raise ApiError(
                409,
                code="scene_unavailable",
                category="project",
                message="The Gaussian scene is unavailable.",
            ) from error
        source_root = (root / "source").resolve()
        if (
            not scene.is_relative_to(source_root)
            or not scene.is_file()
            or has_reparse_component(scene)
        ):
            raise ApiError(
                409,
                code="scene_unavailable",
                category="project",
                message="The Gaussian scene is unavailable.",
            )
        gaussian_scene = load_gaussian_ply(scene)
        return self._renderer.render_pick(
            gaussian_scene, camera, width=width, height=height
        )


class PreviewArtifactStore:
    def __init__(self, project_root: Path, *, buffer_limit: int = 4) -> None:
        self._root = project_root.resolve()
        self._preview_root = (self._root / "previews").resolve()
        self._buffer_limit = buffer_limit
        self._buffers: OrderedDict[str, PickBuffer] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def _png_bytes(buffer: PickBuffer) -> bytes:
        if (
            not isinstance(buffer.expected_depth, np.ndarray)
            or buffer.expected_depth.ndim != 2
        ):
            raise ApiError(
                500,
                code="invalid_preview",
                category="render",
                message="The preview renderer returned an invalid pick buffer.",
            )
        height, width = buffer.expected_depth.shape
        validate_pick_buffer(buffer, width=width, height=height)
        stream = BytesIO()
        Image.fromarray(buffer.rgb).save(stream, format="PNG")
        payload = stream.getvalue()
        if len(payload) > MAX_PREVIEW_ARTIFACT_BYTES:
            raise ApiError(
                413,
                code="preview_too_large",
                category="render",
                message="The generated preview exceeds the local response limit.",
            )
        return payload

    def publish(self, buffer: PickBuffer) -> tuple[str, int, str]:
        artifact_id = uuid4().hex
        payload = self._png_bytes(buffer)
        destination = self._preview_root / f"{artifact_id}.png"
        temporary = destination.with_suffix(".png.tmp")
        if (
            has_reparse_component(self._root)
            or has_reparse_component(self._preview_root)
            or not self._preview_root.is_relative_to(self._root)
        ):
            raise ApiError(
                409,
                code="preview_storage_changed",
                category="filesystem",
                message="The preview storage is no longer safely owned.",
            )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError as error:
            raise ApiError(
                500,
                code="preview_storage_failed",
                category="filesystem",
                message="The preview image could not be stored.",
                retryable=True,
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
        with self._lock:
            self._buffers[artifact_id] = buffer
            self._buffers.move_to_end(artifact_id)
            while len(self._buffers) > self._buffer_limit:
                self._buffers.popitem(last=False)
        return artifact_id, len(payload), hashlib.sha256(payload).hexdigest()

    def read(
        self,
        *,
        artifact_id: str,
        expected_size: int,
        expected_sha256: str,
    ) -> bytes:
        if len(artifact_id) != 32 or any(character not in "0123456789abcdef" for character in artifact_id):
            raise ApiError(
                404,
                code="preview_unavailable",
                category="project",
                message="The requested preview is unavailable.",
            )
        expected_path = f"previews/{artifact_id}.png"
        try:
            path = (self._root / expected_path).resolve(strict=True)
            stat = path.stat()
        except OSError as error:
            raise ApiError(
                404,
                code="preview_unavailable",
                category="project",
                message="The requested preview is unavailable.",
            ) from error
        if (
            not path.is_relative_to(self._preview_root)
            or has_reparse_component(path)
            or not path.is_file()
            or stat.st_nlink != 1
            or stat.st_size != expected_size
            or stat.st_size > MAX_PREVIEW_ARTIFACT_BYTES
        ):
            raise ApiError(
                409,
                code="preview_artifact_changed",
                category="filesystem",
                message="The preview artifact identity changed.",
            )
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ApiError(
                409,
                code="preview_artifact_changed",
                category="filesystem",
                message="The preview artifact identity changed.",
            )
        return payload

    def pick_buffer(self, artifact_id: str) -> PickBuffer:
        with self._lock:
            try:
                buffer = self._buffers[artifact_id]
            except KeyError as error:
                raise ApiError(
                    409,
                    code="pick_buffer_unavailable",
                    category="render",
                    message="Regenerate the preview before choosing a foot point.",
                    retryable=True,
                ) from error
            self._buffers.move_to_end(artifact_id)
            return buffer
