from __future__ import annotations

import hashlib
import os
import stat
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Literal, Protocol
from uuid import uuid4

import numpy as np
from PIL import Image

from gs_video.api.schemas import ApiError, SubjectMediaRole
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import ArtifactRole, Project, StageName, StageStatus
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.gsplat_renderer import GsplatRenderer
from gs_video.scene.ply import load_gaussian_ply
from gs_video.segmentation.paths import has_reparse_component


MAX_PREVIEW_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_SUBJECT_IMAGE_PIXELS = 1920 * 1080


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


@dataclass(frozen=True)
class ResolvedSubjectMedia:
    role: SubjectMediaRole
    artifact_id: str
    frame_index: int
    width: int
    height: int
    size: int
    mime_type: Literal["image/jpeg", "image/png"]
    payload: bytes


@dataclass(frozen=True)
class _SubjectMediaDefinition:
    stage_name: StageName
    artifact_role: ArtifactRole
    directory_name: str
    suffix: str
    image_format: str
    image_mode: str
    mime_type: Literal["image/jpeg", "image/png"]


_SUBJECT_MEDIA_DEFINITIONS = {
    SubjectMediaRole.PROXY: _SubjectMediaDefinition(
        stage_name=StageName.INGEST,
        artifact_role=ArtifactRole.PROXY_FRAMES,
        directory_name="proxies",
        suffix=".jpg",
        image_format="JPEG",
        image_mode="RGB",
        mime_type="image/jpeg",
    ),
    SubjectMediaRole.ALPHA: _SubjectMediaDefinition(
        stage_name=StageName.SEGMENT,
        artifact_role=ArtifactRole.SUBJECT_MASKS,
        directory_name="masks",
        suffix=".png",
        image_format="PNG",
        image_mode="L",
        mime_type="image/png",
    ),
}


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_nlink),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _subject_media_changed(message: str) -> ApiError:
    return ApiError(
        409,
        code="subject_media_changed",
        category="filesystem",
        message=message,
    )


def _subject_artifact_inventory(
    project: Project,
    project_root: Path,
    definition: _SubjectMediaDefinition,
) -> tuple[Path, tuple[Path, ...]]:
    stage = project.stages.get(definition.stage_name)
    if stage is None or stage.status is not StageStatus.SUCCEEDED:
        raise ApiError(
            409,
            code="subject_media_not_ready",
            category="project",
            message="The requested subject media is not ready.",
        )
    registered = stage.artifacts.get(definition.artifact_role)
    if stage.cache_key is None or registered != definition.directory_name:
        raise ApiError(
            409,
            code="subject_media_contract_missing",
            category="project",
            message="The completed stage did not register its exact subject media root.",
        )

    try:
        requested_root = project_root.absolute()
        if has_reparse_component(requested_root):
            raise OSError("project root contains a reparse point")
        root = requested_root.resolve(strict=True)
        allowed_root = root / definition.directory_name
        artifact_root = allowed_root.resolve(strict=True)
        entries = tuple(sorted(artifact_root.iterdir(), key=lambda path: path.name))
    except OSError as error:
        raise ApiError(
            409,
            code="subject_media_unavailable",
            category="project",
            message="The requested subject media inventory is unavailable.",
        ) from error
    if (
        artifact_root != allowed_root
        or not artifact_root.is_dir()
        or has_reparse_component(artifact_root)
        or not entries
    ):
        raise _subject_media_changed(
            "The subject media artifact root identity changed."
        )
    expected_names = tuple(
        f"{index:06d}{definition.suffix}"
        for index in range(1, len(entries) + 1)
    )
    if tuple(path.name for path in entries) != expected_names:
        raise _subject_media_changed(
            "The subject media frame inventory is no longer canonical."
        )
    for path in entries:
        try:
            path_stat = path.stat()
        except OSError as error:
            raise _subject_media_changed(
                "The subject media frame inventory changed."
            ) from error
        if (
            path.parent != artifact_root
            or has_reparse_component(path)
            or not stat.S_ISREG(path_stat.st_mode)
            or (int(path_stat.st_dev), int(path_stat.st_ino)) == (0, 0)
            or path_stat.st_nlink != 1
            or path_stat.st_size <= 0
            or path_stat.st_size > MAX_PREVIEW_ARTIFACT_BYTES
        ):
            raise _subject_media_changed(
                "The subject media frame identity changed."
            )
    return artifact_root, entries


def _read_subject_image(
    path: Path,
    definition: _SubjectMediaDefinition,
    *,
    expected_dimensions: tuple[int, int] | None = None,
) -> tuple[bytes, str, int, int]:
    try:
        before = path.stat()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                _file_fingerprint(opened) != _file_fingerprint(before)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_size <= 0
                or opened.st_size > MAX_PREVIEW_ARTIFACT_BYTES
            ):
                raise _subject_media_changed(
                    "The subject media frame identity changed before it was read."
                )
            payload = stream.read(MAX_PREVIEW_ARTIFACT_BYTES + 1)
            after_handle = os.fstat(stream.fileno())
        after_path = path.stat()
    except ApiError:
        raise
    except OSError as error:
        raise _subject_media_changed(
            "The subject media frame could not be read from its owned identity."
        ) from error
    if (
        len(payload) != before.st_size
        or len(payload) > MAX_PREVIEW_ARTIFACT_BYTES
        or _file_fingerprint(after_handle) != _file_fingerprint(before)
        or _file_fingerprint(after_path) != _file_fingerprint(before)
    ):
        raise _subject_media_changed(
            "The subject media frame changed while it was being read."
        )

    try:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            if (
                image.format != definition.image_format
                or image.mode != definition.image_mode
                or width <= 0
                or height <= 0
                or width * height > MAX_SUBJECT_IMAGE_PIXELS
                or (
                    expected_dimensions is not None
                    and image.size != expected_dimensions
                )
            ):
                raise ValueError("unexpected subject image properties")
            image.load()
    except (OSError, ValueError) as error:
        raise ApiError(
            409,
            code="subject_media_invalid",
            category="project",
            message="The subject media artifact has invalid format or dimensions.",
        ) from error
    return payload, hashlib.sha256(payload).hexdigest(), width, height


def resolve_subject_media(
    project: Project,
    project_root: Path,
    role: SubjectMediaRole,
) -> ResolvedSubjectMedia:
    definition = _SUBJECT_MEDIA_DEFINITIONS[role]
    prompt = project.workflow.subject_prompt
    if role is SubjectMediaRole.ALPHA and prompt is None:
        raise ApiError(
            409,
            code="subject_media_not_ready",
            category="project",
            message="Select the subject before requesting its Alpha mask.",
        )
    stage = project.stages[definition.stage_name]
    artifact_root, inventory = _subject_artifact_inventory(
        project, project_root, definition
    )
    frame_index = 0 if prompt is None else prompt.frame_index
    if frame_index >= len(inventory):
        raise ApiError(
            409,
            code="subject_media_unavailable",
            category="project",
            message="The requested subject media frame is outside the inventory.",
        )
    path = artifact_root / f"{frame_index + 1:06d}{definition.suffix}"
    expected_dimensions = None
    if role is SubjectMediaRole.ALPHA:
        proxy = resolve_subject_media(
            project, project_root, SubjectMediaRole.PROXY
        )
        expected_dimensions = (proxy.width, proxy.height)
    payload, digest, width, height = _read_subject_image(
        path, definition, expected_dimensions=expected_dimensions
    )
    try:
        current_names = tuple(
            sorted(child.name for child in artifact_root.iterdir())
        )
    except OSError as error:
        raise _subject_media_changed(
            "The subject media frame inventory changed while it was read."
        ) from error
    if current_names != tuple(item.name for item in inventory):
        raise _subject_media_changed(
            "The subject media frame inventory changed while it was read."
        )
    artifact_id = hashlib.sha256(
        f"{role.value}\0{stage.cache_key}\0{frame_index}\0{digest}".encode()
    ).hexdigest()[:32]
    return ResolvedSubjectMedia(
        role=role,
        artifact_id=artifact_id,
        frame_index=frame_index,
        width=width,
        height=height,
        size=len(payload),
        mime_type=definition.mime_type,
        payload=payload,
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
