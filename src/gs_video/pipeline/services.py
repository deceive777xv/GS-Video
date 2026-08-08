from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from gs_video.camera.mapping import map_trajectory
from gs_video.camera.opencv_solver import CameraSolution, OpenCvCameraSolver
from gs_video.camera.serialization import (
    MappedTrajectory,
    read_camera_solution,
    read_mapped_trajectory,
    write_camera_solution,
    write_mapped_trajectory,
)
from gs_video.composite.alpha import composite_frame
from gs_video.domain.contracts import (
    MaskSequence,
    Prompt,
    RenderSequence,
    SegmentationBackend,
    StageResult,
)
from gs_video.domain.errors import RepairableError
from gs_video.environment.vram import (
    VramLimitProvider,
    resolve_vram_limit_mb,
    validated_vram_limit_mb,
)
from gs_video.domain.models import (
    ArtifactRole,
    Project,
    SceneSummary,
    StageName,
    StageState,
    StageStatus,
    VideoSummary,
)
from gs_video.media.export import ExportResult, export_mp4, probe_mp4
from gs_video.media.ingest import extract_proxy_frames, extract_source_frames
from gs_video.pipeline.artifacts import ArtifactPublisher, validate_cache_key
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.project.cache import cache_key
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.worker_client import RendererWorkerIdentity
from gs_video.scene.worker_protocol import RenderSequenceRequest
from gs_video.segmentation.paths import has_reparse_component


INGEST_IMPLEMENTATION_VERSION = "media-ingest-v1"
SEGMENT_IMPLEMENTATION_VERSION = "segment-adapter-v1"
CAMERA_IMPLEMENTATION_VERSION = "opencv-camera-adapter-v1"
TRAJECTORY_IMPLEMENTATION_VERSION = "trajectory-map-v1"
COMPOSITE_IMPLEMENTATION_VERSION = "full-resolution-composite-v1"
EXPORT_IMPLEMENTATION_VERSION = "verified-export-v1"
RENDER_IMPLEMENTATION_VERSION = "renderer-worker-adapter-v1"
_FRAME_NAME = re.compile(r"^(\d{6})\.(png|jpg)$")


ProjectMutation = Callable[[Project], None]
ProjectUpdater = Callable[[ProjectMutation], Project]
AssetResolver = Callable[[str, str], Path]


class ExportCallable(Protocol):
    def __call__(
        self,
        frames_dir: Path,
        source_video: Path,
        fps: Fraction,
        frame_count: int,
        output: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult: ...


class Mp4Prober(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult: ...


class MediaIngestBackend(Protocol):
    identity: str

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]: ...

    def extract_proxy_frames(
        self, source: Path, output_dir: Path, max_height: int
    ) -> list[Path]: ...


class ForegroundSegmenterLike(Protocol):
    backend: SegmentationBackend
    worker_prefix: tuple[str, ...]
    model_config: Path
    checkpoint: Path

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence: ...


class CameraSolverLike(Protocol):
    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution: ...


class RendererWorkerLike(Protocol):
    def probe(
        self, *, token: CancellationToken | None = None
    ) -> RendererWorkerIdentity: ...

    def render_sequence(
        self,
        request: RenderSequenceRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence: ...


@dataclass(frozen=True)
class WorkflowPaths:
    root: Path
    update_project: ProjectUpdater | None = None
    resolve_asset: AssetResolver | None = None
    publisher: ArtifactPublisher = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root = Path(self.root).absolute()
        root.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "publisher", ArtifactPublisher(root))

    def relative(self, path: Path) -> Path:
        try:
            return path.relative_to(self.root)
        except ValueError as exc:
            raise RepairableError("流水线产物不在项目目录内") from exc


@dataclass(frozen=True)
class FfmpegMediaIngestBackend:
    identity: str = "ffmpeg-cli"

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]:
        return extract_source_frames(source, output_dir)

    def extract_proxy_frames(
        self, source: Path, output_dir: Path, max_height: int
    ) -> list[Path]:
        return extract_proxy_frames(source, output_dir, max_height=max_height)


@dataclass(frozen=True)
class _FrameInventory:
    paths: tuple[Path, ...]
    sizes: tuple[tuple[int, int], ...]
    fingerprint: str
    directory_identity: tuple[int, int, int, int]

    @property
    def count(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class _FileSnapshot:
    size: int
    sha256: str


def _ordinary_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RepairableError(f"{label}不存在或不可读") from exc
    if (
        has_reparse_component(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise RepairableError(f"{label}必须是项目内单链接普通文件")
    return metadata


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
        int(metadata.st_nlink),
    )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_ctime_ns),
        int(metadata.st_mode),
    )


def _sha256(path: Path, label: str, token: CancellationToken) -> str:
    token.raise_if_cancelled()
    before = _ordinary_file(path, label)
    expected = _file_identity(before)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if _file_identity(opened) != expected:
                raise RepairableError(f"{label}读取前身份发生变化")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                token.raise_if_cancelled()
                digest.update(block)
            after_handle = os.fstat(stream.fileno())
    except OSError as exc:
        raise RepairableError(f"{label}不可读") from exc
    after_path = _ordinary_file(path, label)
    if expected != _file_identity(after_handle) or expected != _file_identity(after_path):
        raise RepairableError(f"{label}读取期间身份发生变化")
    return digest.hexdigest()


def _file_snapshot(
    path: Path, label: str, token: CancellationToken
) -> _FileSnapshot:
    before = _ordinary_file(path, label)
    digest = _sha256(path, label, token)
    after = _ordinary_file(path, label)
    if _file_identity(before) != _file_identity(after):
        raise RepairableError(f"{label}快照期间身份发生变化")
    return _FileSnapshot(int(after.st_size), digest)


def _assert_file_snapshot(
    path: Path,
    expected: _FileSnapshot,
    label: str,
    token: CancellationToken,
) -> None:
    if _file_snapshot(path, label, token) != expected:
        raise RepairableError(f"{label} authority 在处理期间发生变化")


def _source_material(
    paths: WorkflowPaths, project: Project, token: CancellationToken
) -> tuple[Path, VideoSummary, _FileSnapshot]:
    summary = project.workflow.source_summary
    if summary is None or (
        project.source_video_asset_id is None and project.source_video is None
    ):
        raise RepairableError("尚未导入源视频")
    _require_exportable_dimensions(summary)
    if project.source_video_asset_id is not None:
        if paths.resolve_asset is None:
            raise RepairableError("共享素材解析器不可用")
        source = paths.resolve_asset(project.source_video_asset_id, "video")
    else:
        assert project.source_video is not None
        relative = Path(project.source_video)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:1] != ("source",)
        ):
            raise RepairableError("源视频路径不属于项目 source 目录")
        source = paths.root / relative
    snapshot = _file_snapshot(source, "源视频", token)
    if snapshot != _FileSnapshot(summary.size, summary.sha256):
        raise RepairableError("源视频与已登记摘要不一致")
    return source, summary, snapshot


def _scene_material(
    paths: WorkflowPaths,
    project: Project,
    token: CancellationToken,
) -> tuple[Path, SceneSummary, _FileSnapshot]:
    summary = project.workflow.scene_summary
    if summary is None or (
        project.scene_ply_asset_id is None and project.scene_ply is None
    ):
        raise RepairableError("尚未导入 Gaussian 场景")
    if project.scene_ply_asset_id is not None:
        if paths.resolve_asset is None:
            raise RepairableError("共享素材解析器不可用")
        resolved = paths.resolve_asset(project.scene_ply_asset_id, "ply")
        snapshot = _file_snapshot(resolved, "Gaussian 场景", token)
        if snapshot.size != summary.size or snapshot.sha256 != summary.sha256:
            raise RepairableError("Gaussian 场景与已登记摘要不一致")
        return resolved, summary, snapshot
    assert project.scene_ply is not None
    relative = Path(project.scene_ply)
    if relative.is_absolute() or ".." in relative.parts:
        raise RepairableError("Gaussian 场景路径无效")
    scene = (paths.root / relative).absolute()
    source_root = (paths.root / "source").absolute()
    try:
        resolved = scene.resolve(strict=True)
        resolved_source = source_root.resolve(strict=True)
    except OSError as exc:
        raise RepairableError("Gaussian 场景不可用") from exc
    if (
        resolved != scene
        or not resolved.is_relative_to(resolved_source)
        or resolved.name != summary.filename
        or has_reparse_component(resolved)
    ):
        raise RepairableError("Gaussian 场景 authority 无效")
    snapshot = _file_snapshot(resolved, "Gaussian 场景", token)
    if snapshot.size != summary.size or snapshot.sha256 != summary.sha256:
        raise RepairableError("Gaussian 场景与已登记摘要不一致")
    return resolved, summary, snapshot


def _require_exportable_dimensions(summary: VideoSummary) -> None:
    if (
        summary.width < 2
        or summary.height < 2
        or summary.width % 2
        or summary.height % 2
    ):
        raise RepairableError(
            "源视频宽高必须是不小于 2 的偶数，才能无缩放编码为 yuv420p"
        )


def _stage_state(project: Project, name: StageName) -> StageState:
    state = project.stages.get(name)
    if (
        state is None
        or state.status is not StageStatus.SUCCEEDED
        or state.cache_key is None
    ):
        raise RepairableError(f"上游阶段 {name.value} 尚未成功")
    try:
        validate_cache_key(state.cache_key)
    except ValueError as exc:
        raise RepairableError(f"上游阶段 {name.value} 缓存键无效") from exc
    return state


def _artifact_path(
    paths: WorkflowPaths,
    state: StageState,
    role: ArtifactRole,
    category: str,
    *,
    filename: str | None = None,
) -> Path:
    if state.cache_key is None:
        raise RepairableError("上游阶段缺少缓存键")
    registered = state.artifacts.get(role)
    expected = Path(category) / state.cache_key
    if filename is not None:
        expected /= filename
    if registered is None or Path(registered) != expected:
        raise RepairableError(f"上游 artifact {role.value} 与成功缓存不一致")
    candidate = paths.root / expected
    try:
        resolved = candidate.resolve(strict=True)
        root = paths.root.resolve(strict=True)
    except OSError as exc:
        raise RepairableError(f"上游 artifact {role.value} 不存在") from exc
    if not resolved.is_relative_to(root) or has_reparse_component(candidate):
        raise RepairableError(f"上游 artifact {role.value} 越出项目目录")
    return candidate


def _frame_inventory(
    directory: Path,
    *,
    suffix: str,
    image_format: str,
    mode: str,
    label: str,
    expected_count: int | None = None,
    expected_size: tuple[int, int] | None = None,
    expected_sizes: tuple[tuple[int, int], ...] | None = None,
    token: CancellationToken,
) -> _FrameInventory:
    token.raise_if_cancelled()
    try:
        metadata = directory.lstat()
        directory_expected = _directory_identity(metadata)
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise RepairableError(f"{label}帧目录不存在或不可读") from exc
    if has_reparse_component(directory) or not stat.S_ISDIR(metadata.st_mode):
        raise RepairableError(f"{label}帧目录不是普通目录")
    if not entries:
        raise RepairableError(f"{label}帧目录为空")
    ordered = tuple(sorted(entries))
    names = tuple(path.name for path in ordered)
    expected_names = tuple(
        f"{index:06d}.{suffix}" for index in range(1, len(ordered) + 1)
    )
    if names != expected_names:
        raise RepairableError(f"{label}帧 inventory 必须从 000001 开始连续编号")
    if expected_count is not None and len(ordered) != expected_count:
        raise RepairableError(f"{label}帧数与已登记帧数不一致")
    if expected_sizes is not None and len(expected_sizes) != len(ordered):
        raise RepairableError(f"{label}帧数与尺寸 authority 不一致")

    sizes: list[tuple[int, int]] = []
    fingerprint = hashlib.sha256()
    try:
        for index, path in enumerate(ordered):
            token.raise_if_cancelled()
            file_metadata = _ordinary_file(path, f"{label}帧")
            expected_identity = _file_identity(file_metadata)
            match = _FRAME_NAME.fullmatch(path.name)
            if match is None or match.group(2) != suffix:
                raise RepairableError(f"{label}帧文件名无效")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if _file_identity(opened) != expected_identity:
                    raise RepairableError(f"{label}帧打开前身份发生变化")
                image = Image.open(stream)
                try:
                    image.load()
                    if image.format != image_format or image.mode != mode:
                        raise RepairableError(
                            f"{label}帧必须是 {mode} {image_format} 图像"
                        )
                    size = image.size
                    token.raise_if_cancelled()
                    stream.seek(0)
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        token.raise_if_cancelled()
                        digest.update(block)
                    after_handle = os.fstat(stream.fileno())
                finally:
                    image.close()
            after_path = _ordinary_file(path, f"{label}帧")
            if (
                _file_identity(after_handle) != expected_identity
                or _file_identity(after_path) != expected_identity
            ):
                raise RepairableError(f"{label}帧读取期间身份被替换或发生变化")
            if expected_size is not None and size != expected_size:
                raise RepairableError(f"{label}帧尺寸与源视频不一致")
            if expected_sizes is not None and size != expected_sizes[index]:
                raise RepairableError(f"{label}帧尺寸与上游帧不一致")
            sizes.append(size)
            fingerprint.update(path.name.encode("ascii"))
            fingerprint.update(str(file_metadata.st_size).encode("ascii"))
            fingerprint.update(digest.hexdigest().encode("ascii"))
            token.raise_if_cancelled()
    except (OSError, UnidentifiedImageError) as exc:
        raise RepairableError(f"{label}帧不可读") from exc
    try:
        directory_after = directory.lstat()
    except OSError as exc:
        raise RepairableError(f"{label}帧目录在读取期间消失") from exc
    if (
        has_reparse_component(directory)
        or _directory_identity(directory_after) != directory_expected
    ):
        raise RepairableError(f"{label}帧目录身份在读取期间发生变化")
    return _FrameInventory(
        ordered,
        tuple(sizes),
        fingerprint.hexdigest(),
        directory_expected,
    )


def _assert_frame_snapshot(
    directory: Path,
    expected: _FrameInventory,
    *,
    suffix: str,
    image_format: str,
    mode: str,
    label: str,
    token: CancellationToken,
) -> None:
    actual = _frame_inventory(
        directory,
        suffix=suffix,
        image_format=image_format,
        mode=mode,
        label=label,
        expected_count=expected.count,
        expected_sizes=expected.sizes,
        token=token,
    )
    if actual != expected:
        raise RepairableError(f"{label}帧 authority 在处理期间发生变化")


def _model_identity(
    path: Path, label: str, token: CancellationToken
) -> dict[str, object]:
    snapshot = _file_snapshot(path, label, token)
    return {
        "filename": path.name,
        "size": snapshot.size,
        "sha256": snapshot.sha256,
    }


def _segmentation_identity(
    segmenter: ForegroundSegmenterLike, token: CancellationToken
) -> dict[str, object]:
    backend_value = getattr(segmenter.backend, "value", segmenter.backend)
    return {
        "backend": str(backend_value),
        "worker_prefix": list(segmenter.worker_prefix),
        "config": _model_identity(segmenter.model_config, "分割配置", token),
        "checkpoint": _model_identity(segmenter.checkpoint, "分割模型", token),
    }


def _validate_proxy_dimensions(
    inventory: _FrameInventory,
    summary: VideoSummary,
    maximum_height: int,
) -> None:
    if len(set(inventory.sizes)) != 1 or any(
        width > summary.width or height > min(summary.height, maximum_height)
        for width, height in inventory.sizes
    ):
        raise RepairableError("代理帧尺寸不一致或超过源视频/代理高度上限")


_MP4_MANIFEST_NAME = "manifest.json"
_MP4_MANIFEST_KEYS = frozenset(
    {
        "version",
        "cache_key",
        "filename",
        "size",
        "sha256",
        "fps",
        "frame_count",
        "duration",
        "has_audio",
    }
)


def _validate_export_result(
    result: ExportResult,
    output: Path,
    *,
    fps: Fraction,
    frame_count: int,
    has_audio: bool,
    label: str,
    allow_duration_tolerance: bool,
) -> None:
    expected_duration = Fraction(frame_count, 1) / fps
    if Path(result.output).absolute() != output.absolute():
        raise RepairableError(f"{label} metadata 输出路径不一致")
    if result.fps != fps:
        raise RepairableError(f"{label} metadata 帧率不一致")
    if result.frame_count != frame_count:
        raise RepairableError(f"{label} metadata 帧数不一致")
    if result.has_audio is not has_audio:
        raise RepairableError(f"{label} metadata 音轨 authority 不一致")
    duration_error = abs(result.duration - expected_duration)
    tolerance = Fraction(1, 1) / fps if allow_duration_tolerance else Fraction(0)
    if duration_error > tolerance:
        raise RepairableError(f"{label} metadata 时长不一致")


def _write_mp4_manifest(
    directory: Path,
    *,
    cache_key_value: str,
    filename: str,
    result: ExportResult,
    token: CancellationToken,
) -> None:
    output = directory / filename
    metadata = _ordinary_file(output, "MP4")
    if metadata.st_size <= 0:
        raise RepairableError("MP4 不能为空")
    payload = {
        "version": 1,
        "cache_key": cache_key_value,
        "filename": filename,
        "size": int(metadata.st_size),
        "sha256": _sha256(output, "MP4", token),
        "fps": str(result.fps),
        "frame_count": result.frame_count,
        "duration": str(result.duration),
        "has_audio": result.has_audio,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    (directory / _MP4_MANIFEST_NAME).write_text(encoded, encoding="utf-8")
    token.raise_if_cancelled()


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RepairableError(f"MP4 manifest 包含重复字段: {key}")
        result[key] = value
    return result


def _forbid_json_constant(value: str) -> None:
    raise RepairableError(f"MP4 manifest 包含非有限数值: {value}")


def _read_mp4_manifest(path: Path, token: CancellationToken) -> dict[str, object]:
    metadata = _ordinary_file(path, "MP4 manifest")
    if metadata.st_size <= 0 or metadata.st_size > 64 * 1024:
        raise RepairableError("MP4 manifest 大小无效")
    expected = _file_identity(metadata)
    token.raise_if_cancelled()
    try:
        with path.open("rb") as stream:
            if _file_identity(os.fstat(stream.fileno())) != expected:
                raise RepairableError("MP4 manifest 打开前身份发生变化")
            payload = stream.read(64 * 1024 + 1)
            after_handle = os.fstat(stream.fileno())
    except OSError as exc:
        raise RepairableError("MP4 manifest 不可读") from exc
    after_path = _ordinary_file(path, "MP4 manifest")
    if (
        len(payload) != metadata.st_size
        or _file_identity(after_handle) != expected
        or _file_identity(after_path) != expected
    ):
        raise RepairableError("MP4 manifest 读取期间身份发生变化")
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_forbid_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairableError("MP4 manifest JSON 无效") from exc
    if not isinstance(document, dict) or set(document) != _MP4_MANIFEST_KEYS:
        raise RepairableError("MP4 manifest 字段不完整或包含未知字段")
    return document


def _validate_published_mp4(
    directory: Path,
    *,
    cache_key_value: str,
    filename: str,
    fps: Fraction,
    frame_count: int,
    has_audio: bool,
    prober: Mp4Prober,
    token: CancellationToken,
) -> Path:
    token.raise_if_cancelled()
    allowed = {filename, _MP4_MANIFEST_NAME, ".gs-video-logs"}
    try:
        names = {entry.name for entry in directory.iterdir()}
    except OSError as exc:
        raise RepairableError("MP4 artifact 目录不可读") from exc
    if not {filename, _MP4_MANIFEST_NAME}.issubset(names) or not names <= allowed:
        raise RepairableError("MP4 artifact 目录成员与 manifest authority 不一致")
    output = directory / filename
    metadata = _ordinary_file(output, "MP4")
    if metadata.st_size <= 0:
        raise RepairableError("MP4 不能为空")
    document = _read_mp4_manifest(directory / _MP4_MANIFEST_NAME, token)
    expected_duration = Fraction(frame_count, 1) / fps
    raw_cache_key = document["cache_key"]
    raw_filename = document["filename"]
    raw_sha256 = document["sha256"]
    raw_fps = document["fps"]
    raw_duration = document["duration"]
    if not all(
        type(value) is str
        for value in (
            raw_cache_key,
            raw_filename,
            raw_sha256,
            raw_fps,
            raw_duration,
        )
    ):
        raise RepairableError("MP4 manifest 字符串字段类型无效")
    assert isinstance(raw_cache_key, str)
    assert isinstance(raw_filename, str)
    assert isinstance(raw_sha256, str)
    assert isinstance(raw_fps, str)
    assert isinstance(raw_duration, str)
    try:
        manifest_fps = Fraction(raw_fps)
        manifest_duration = Fraction(raw_duration)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise RepairableError("MP4 manifest 的帧率或时长无效") from exc
    values_match = (
        type(document["version"]) is int
        and document["version"] == 1
        and raw_cache_key == cache_key_value
        and raw_filename == filename
        and type(document["size"]) is int
        and document["size"] == metadata.st_size
        and len(raw_sha256) == 64
        and all(character in "0123456789abcdef" for character in raw_sha256)
        and manifest_fps == fps
        and type(document["frame_count"]) is int
        and document["frame_count"] == frame_count
        and manifest_duration == expected_duration
        and type(document["has_audio"]) is bool
        and document["has_audio"] is has_audio
    )
    if not values_match:
        raise RepairableError("MP4 manifest 与当前 cache authority 不一致")
    digest = _sha256(output, "MP4", token)
    if digest != raw_sha256:
        raise RepairableError("MP4 hash 与 manifest 摘要不一致")
    probed = prober(output, cancellation_check=token.raise_if_cancelled)
    _validate_export_result(
        probed,
        output,
        fps=fps,
        frame_count=frame_count,
        has_audio=has_audio,
        label="ffprobe",
        allow_duration_tolerance=True,
    )
    if (
        _ordinary_file(output, "MP4").st_size != metadata.st_size
        or _sha256(output, "MP4", token) != digest
    ):
        raise RepairableError("MP4 在 ffprobe 期间发生变化")
    return output


class MediaIngestService:
    def __init__(
        self,
        paths: WorkflowPaths,
        backend: MediaIngestBackend | None = None,
        *,
        proxy_height: int = 540,
    ) -> None:
        if type(proxy_height) is not int or not 1 <= proxy_height <= 540:
            raise ValueError("proxy_height must be an integer between 1 and 540")
        self.paths = paths
        self.backend = backend or FfmpegMediaIngestBackend()
        if not self.backend.identity:
            raise ValueError("media backend identity must not be empty")
        self.proxy_height = proxy_height

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        source, summary, source_snapshot = _source_material(
            self.paths, project, token
        )
        source_identity = summary.model_dump(mode="json")
        # The decoded inventory can fill this derived probe field. Keeping it out of
        # the key makes the first unknown-count run reusable after persistence.
        source_identity.pop("frame_count", None)
        result_key = cache_key(
            StageName.INGEST.value,
            {"source": source_identity},
            {
                "proxy_height": self.proxy_height,
                "backend_identity": self.backend.identity,
            },
            INGEST_IMPLEMENTATION_VERSION,
        )
        discovered_count: int | None = None

        def build_source(staging: Path) -> None:
            nonlocal discovered_count
            token.raise_if_cancelled()
            self.backend.extract_source_frames(source, staging)
            inventory = _frame_inventory(
                staging,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="源",
                expected_count=summary.frame_count,
                expected_size=(summary.width, summary.height),
                token=token,
            )
            discovered_count = inventory.count
            _assert_file_snapshot(source, source_snapshot, "源视频", token)
            token.raise_if_cancelled()

        source_directory = self.paths.publisher.publish_tree(
            "frames", result_key, build_source
        )
        source_inventory = _frame_inventory(
            source_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="源",
            expected_count=summary.frame_count,
            expected_size=(summary.width, summary.height),
            token=token,
        )
        discovered_count = source_inventory.count
        emit(1, 2, "提取全分辨率源帧 1/2")

        def build_proxies(staging: Path) -> None:
            token.raise_if_cancelled()
            self.backend.extract_proxy_frames(source, staging, self.proxy_height)
            inventory = _frame_inventory(
                staging,
                suffix="jpg",
                image_format="JPEG",
                mode="RGB",
                label="代理",
                expected_count=source_inventory.count,
                token=token,
            )
            _validate_proxy_dimensions(inventory, summary, self.proxy_height)
            _assert_file_snapshot(source, source_snapshot, "源视频", token)
            token.raise_if_cancelled()

        proxy_directory = self.paths.publisher.publish_tree(
            "proxies", result_key, build_proxies
        )
        published_proxies = _frame_inventory(
            proxy_directory,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            expected_count=source_inventory.count,
            token=token,
        )
        _validate_proxy_dimensions(published_proxies, summary, self.proxy_height)
        token.raise_if_cancelled()
        if summary.frame_count is None:
            self._persist_discovered_count(project, summary, discovered_count)
        emit(2, 2, "提取代理帧 2/2")
        source_relative = self.paths.relative(source_directory)
        proxy_relative = self.paths.relative(proxy_directory)
        return StageResult(
            output_paths=(source_relative, proxy_relative),
            cache_key=result_key,
            artifacts={
                ArtifactRole.SOURCE_FRAMES: source_relative,
                ArtifactRole.PROXY_FRAMES: proxy_relative,
            },
        )

    def _persist_discovered_count(
        self, project: Project, summary: VideoSummary, discovered_count: int
    ) -> None:
        def mutation(latest: Project) -> None:
            current = latest.workflow.source_summary
            if (
                current is None
                or current.sha256 != summary.sha256
                or current.size != summary.size
            ):
                raise RepairableError("源视频 authority 在帧提取期间发生变化")
            if current.frame_count not in {None, discovered_count}:
                raise RepairableError("探测帧数与已登记帧数不一致")
            current.frame_count = discovered_count

        if self.paths.update_project is not None:
            self.paths.update_project(mutation)
        mutation(project)


class SegmentWorkflowService:
    def __init__(
        self, paths: WorkflowPaths, segmenter: ForegroundSegmenterLike
    ) -> None:
        self.paths = paths
        self.segmenter = segmenter

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        ingest = _stage_state(project, StageName.INGEST)
        proxies = _artifact_path(
            self.paths, ingest, ArtifactRole.PROXY_FRAMES, "proxies"
        )
        inventory = _frame_inventory(
            proxies,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            token=token,
        )
        prompt_state = project.workflow.subject_prompt
        if prompt_state is None:
            raise RepairableError("尚未选择前景主体")
        if prompt_state.frame_index >= inventory.count:
            raise RepairableError("主体提示帧超出代理帧范围")
        width, height = inventory.sizes[prompt_state.frame_index]
        if prompt_state.x >= width or prompt_state.y >= height:
            raise RepairableError("主体提示点超出代理帧范围")
        prompt = Prompt(
            frame_index=prompt_state.frame_index,
            x=prompt_state.x,
            y=prompt_state.y,
        )
        identity = _segmentation_identity(self.segmenter, token)
        result_key = cache_key(
            StageName.SEGMENT.value,
            {"ingest_cache_key": ingest.cache_key},
            {"prompt": prompt_state.model_dump(mode="json"), "backend": identity},
            SEGMENT_IMPLEMENTATION_VERSION,
        )

        def build(staging: Path) -> None:
            token.raise_if_cancelled()
            worker_output = staging / "worker-output"
            sequence = self.segmenter.segment(
                list(inventory.paths), prompt, worker_output, emit, token
            )
            token.raise_if_cancelled()
            if Path(sequence.mask_dir).absolute() != worker_output.absolute():
                raise RepairableError("分割 worker 返回了错误的输出目录")
            masks = _frame_inventory(
                worker_output,
                suffix="png",
                image_format="PNG",
                mode="L",
                label="遮罩",
                expected_count=inventory.count,
                expected_sizes=inventory.sizes,
                token=token,
            )
            if sequence.frame_count != masks.count:
                raise RepairableError("分割 worker 返回的帧数不一致")
            for mask in masks.paths:
                mask.replace(staging / mask.name)
            try:
                worker_output.rmdir()
            except OSError as exc:
                raise RepairableError("分割 worker 输出包含未登记成员") from exc
            _frame_inventory(
                staging,
                suffix="png",
                image_format="PNG",
                mode="L",
                label="遮罩",
                expected_count=inventory.count,
                expected_sizes=inventory.sizes,
                token=token,
            )
            _assert_frame_snapshot(
                proxies,
                inventory,
                suffix="jpg",
                image_format="JPEG",
                mode="RGB",
                label="代理",
                token=token,
            )
            if _segmentation_identity(self.segmenter, token) != identity:
                raise RepairableError("分割 backend identity 在运行期间发生变化")
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("masks", result_key, build)
        _frame_inventory(
            output,
            suffix="png",
            image_format="PNG",
            mode="L",
            label="遮罩",
            expected_count=inventory.count,
            expected_sizes=inventory.sizes,
            token=token,
        )
        relative = self.paths.relative(output)
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.SUBJECT_MASKS: relative},
        )


class CameraSolveWorkflowService:
    def __init__(
        self,
        paths: WorkflowPaths,
        solver: CameraSolverLike | None = None,
        *,
        backend_identity: str | None = None,
    ) -> None:
        self.paths = paths
        self.solver: CameraSolverLike
        if solver is None:
            self.solver = OpenCvCameraSolver()
            self.backend_identity = backend_identity or "opencv-camera-solver-v1"
        else:
            if not backend_identity:
                raise ValueError(
                    "custom camera solver requires an explicit nonempty identity"
                )
            self.solver = solver
            self.backend_identity = backend_identity

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        ingest = _stage_state(project, StageName.INGEST)
        proxy_directory = _artifact_path(
            self.paths, ingest, ArtifactRole.PROXY_FRAMES, "proxies"
        )
        proxies = _frame_inventory(
            proxy_directory,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            token=token,
        )
        result_key = cache_key(
            StageName.SOLVE_CAMERA.value,
            {
                "ingest_cache_key": ingest.cache_key,
                "proxy_inventory": proxies.fingerprint,
            },
            {"backend_identity": self.backend_identity},
            CAMERA_IMPLEMENTATION_VERSION,
        )

        def build(staging: Path) -> None:
            token.raise_if_cancelled()
            solution = self.solver.solve(list(proxies.paths), emit, token)
            if len(solution.camera_to_world) != proxies.count:
                raise RepairableError("相机求解轨迹帧数与代理帧数不一致")
            write_camera_solution(staging / "solution.json", solution)
            _assert_frame_snapshot(
                proxy_directory,
                proxies,
                suffix="jpg",
                image_format="JPEG",
                mode="RGB",
                label="代理",
                token=token,
            )
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("camera", result_key, build)
        relative = self.paths.relative(output / "solution.json")
        restored = read_camera_solution(self.paths.root / relative)
        if len(restored.camera_to_world) != proxies.count:
            raise RepairableError("缓存相机轨迹帧数与代理帧数不一致")
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.CAMERA_SOLUTION: relative},
        )


class TrajectoryMapWorkflowService:
    def __init__(self, paths: WorkflowPaths) -> None:
        self.paths = paths

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        solve = _stage_state(project, StageName.SOLVE_CAMERA)
        solution_path = _artifact_path(
            self.paths,
            solve,
            ArtifactRole.CAMERA_SOLUTION,
            "camera",
            filename="solution.json",
        )
        solution_snapshot = _file_snapshot(solution_path, "相机求解产物", token)
        solution = read_camera_solution(solution_path)
        workflow = project.workflow
        camera = workflow.target_camera
        preview = workflow.preview
        foot = workflow.foot_point
        if camera is None or preview is None or foot is None:
            raise RepairableError("目标相机、确认预览和落脚点尚未完整设置")
        authority_matches = (
            camera.revision == workflow.confirmed_camera_revision
            and camera.revision == preview.camera_revision
            and camera.revision == foot.camera_revision
            and preview.artifact_id == workflow.confirmed_preview_artifact_id
            and preview.artifact_id == foot.preview_artifact_id
            and preview.pick_buffer_revision == foot.pick_buffer_revision
        )
        if not authority_matches:
            raise RepairableError("目标相机、确认预览和落脚点 authority 不一致")
        target = OrbitCamera(
            target=camera.target,
            distance=camera.distance,
            yaw=camera.yaw,
            pitch=camera.pitch,
            fov_y_degrees=camera.fov_y_degrees,
        )
        mapped = map_trajectory(solution, target.camera_to_world(), workflow.motion_scale)
        token.raise_if_cancelled()
        result_key = cache_key(
            StageName.MAP_TRAJECTORY.value,
            {
                "solve_cache_key": solve.cache_key,
                "camera_artifact_sha256": solution_snapshot.sha256,
            },
            {
                "camera": camera.model_dump(mode="json"),
                "confirmed_preview_artifact_id": workflow.confirmed_preview_artifact_id,
                "foot_point": foot.model_dump(mode="json"),
                "motion_scale": workflow.motion_scale,
            },
            TRAJECTORY_IMPLEMENTATION_VERSION,
        )

        def build(staging: Path) -> None:
            token.raise_if_cancelled()
            write_mapped_trajectory(
                staging / "trajectory.json",
                MappedTrajectory(camera.fov_y_degrees, mapped),
            )
            _assert_file_snapshot(
                solution_path,
                solution_snapshot,
                "相机求解产物",
                token,
            )
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("trajectories", result_key, build)
        relative = self.paths.relative(output / "trajectory.json")
        restored = read_mapped_trajectory(self.paths.root / relative)
        if (
            restored.fov_y_degrees != camera.fov_y_degrees
            or len(restored.camera_to_world) != len(mapped)
            or any(
                not np.allclose(actual, expected, atol=1e-12)
                for actual, expected in zip(
                    restored.camera_to_world, mapped, strict=True
                )
            )
        ):
            raise RepairableError("缓存映射轨迹与当前相机 authority 不一致")
        emit(1, 1, "映射目标相机轨迹 1/1")
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.MAPPED_TRAJECTORY: relative},
        )


class RendererWorkflowService:
    def __init__(
        self,
        paths: WorkflowPaths,
        worker: RendererWorkerLike,
        *,
        sh_degree: int = 3,
        available_vram_limit_mb: int = 8192,
        vram_limit_provider: VramLimitProvider | None = None,
    ) -> None:
        if type(sh_degree) is not int or not 0 <= sh_degree <= 3:
            raise ValueError("sh_degree must be an integer between 0 and 3")
        validated_vram_limit_mb(available_vram_limit_mb)
        self.paths = paths
        self.worker = worker
        self.sh_degree = sh_degree
        self.available_vram_limit_mb = available_vram_limit_mb
        self._vram_limit_provider = vram_limit_provider

    def run(
        self,
        project: Project,
        namespace: object,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        namespace_value = getattr(namespace, "value", namespace)
        if namespace_value not in {"preview", "final"}:
            raise RepairableError("渲染缓存 namespace 无效")
        scene, scene_summary, scene_snapshot = _scene_material(
            self.paths, project, token
        )
        mapped = _stage_state(project, StageName.MAP_TRAJECTORY)
        trajectory_path = _artifact_path(
            self.paths,
            mapped,
            ArtifactRole.MAPPED_TRAJECTORY,
            "trajectories",
            filename="trajectory.json",
        )
        trajectory_snapshot = _file_snapshot(
            trajectory_path, "映射轨迹", token
        )
        trajectory = read_mapped_trajectory(trajectory_path)
        summary = project.workflow.source_summary
        if summary is None:
            raise RepairableError("源视频摘要不可用")
        frame_count = len(trajectory.camera_to_world)
        if frame_count <= 0 or (
            summary.frame_count is not None and frame_count != summary.frame_count
        ):
            raise RepairableError("映射轨迹帧数与源视频不一致")
        preview_stride = (
            1
            if namespace_value == "final"
            else max(1, (frame_count + 149) // 150)
        )
        if namespace_value == "final":
            width, height = summary.width, summary.height
        else:
            width, height = _preview_size(
                (summary.width, summary.height), project.workflow.preview_height
            )
        extra_framebuffer_bytes = (
            max(0, width * height - 1920 * 1080) * 24
        )
        estimated_vram_mb = scene_summary.estimated_vram_mb + (
            extra_framebuffer_bytes + 1024**2 - 1
        ) // 1024**2
        available_vram_limit_mb = resolve_vram_limit_mb(
            self.available_vram_limit_mb,
            self._vram_limit_provider,
        )
        if estimated_vram_mb * 5 > available_vram_limit_mb * 4:
            raise RepairableError("Gaussian 场景超过配置的保守显存预算")
        identity = self.worker.probe(token=token)
        expected_implementation = f"gsplat-{identity.gsplat}"
        result_key = cache_key(
            StageName.RENDER.value,
            {
                "mapped_cache_key": mapped.cache_key,
                "trajectory_sha256": trajectory_snapshot.sha256,
                "scene_sha256": scene_snapshot.sha256,
            },
            {
                "namespace": namespace_value,
                "width": width,
                "height": height,
                "sh_degree": self.sh_degree,
                "preview_stride": preview_stride,
                "worker": {
                    "torch": identity.torch,
                    "gsplat": identity.gsplat,
                    "device": identity.device,
                },
            },
            RENDER_IMPLEMENTATION_VERSION,
        )
        expected_count = len(range(0, frame_count, preview_stride))

        def build(staging: Path) -> None:
            token.raise_if_cancelled()
            worker_output = staging / "worker-output"
            rendered = self.worker.render_sequence(
                RenderSequenceRequest(
                    type="render_sequence",
                    scene_path=scene,
                    camera_manifest=trajectory_path,
                    output_dir=worker_output,
                    width=width,
                    height=height,
                    sh_degree=self.sh_degree,
                    background=(0.0, 0.0, 0.0),
                    preview_stride=preview_stride,
                ),
                emit,
                token,
            )
            if (
                rendered.frame_dir.absolute() != worker_output.absolute()
                or rendered.frame_count != expected_count
                or rendered.width != width
                or rendered.height != height
                or rendered.implementation_version != expected_implementation
            ):
                raise RepairableError("渲染 worker 返回的实现版本或帧序列无效")
            for frame in rendered.frame_paths:
                frame.replace(staging / frame.name)
            try:
                worker_output.rmdir()
            except OSError as exc:
                raise RepairableError("渲染 worker 输出包含未登记成员") from exc
            _frame_inventory(
                staging,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="渲染",
                expected_count=expected_count,
                expected_size=(width, height),
                token=token,
            )
            _assert_file_snapshot(scene, scene_snapshot, "Gaussian 场景", token)
            _assert_file_snapshot(
                trajectory_path, trajectory_snapshot, "映射轨迹", token
            )
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("renders", result_key, build)
        _frame_inventory(
            output,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="渲染",
            expected_count=expected_count,
            expected_size=(width, height),
            token=token,
        )
        _assert_file_snapshot(scene, scene_snapshot, "Gaussian 场景", token)
        _assert_file_snapshot(
            trajectory_path, trajectory_snapshot, "映射轨迹", token
        )
        relative = self.paths.relative(output)
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.RENDER_FRAMES: relative},
        )


def _preview_size(source: tuple[int, int], maximum_height: int) -> tuple[int, int]:
    width, height = source
    if width < 2 or height < 2 or maximum_height < 2:
        raise RepairableError("无法在不放大的前提下生成偶数尺寸预览")
    scale = min(1.0, maximum_height / height)
    target_width = min(width, int(width * scale))
    target_height = min(height, int(height * scale))
    target_width -= target_width % 2
    target_height -= target_height % 2
    if target_width < 2 or target_height < 2:
        raise RepairableError("无法在不放大的前提下生成偶数尺寸预览")
    return target_width, target_height


class CompositeWorkflowService:
    def __init__(
        self,
        paths: WorkflowPaths,
        *,
        preview_frame_limit: int = 150,
        edge_px: int = 1,
        exporter: ExportCallable | None = None,
        exporter_identity: str | None = None,
        prober: Mp4Prober | None = None,
    ) -> None:
        if type(preview_frame_limit) is not int or preview_frame_limit <= 0:
            raise ValueError("preview_frame_limit must be a positive integer")
        if type(edge_px) is not int or not 0 <= edge_px <= 3:
            raise ValueError("edge_px must be an integer between 0 and 3")
        self.paths = paths
        self.preview_frame_limit = preview_frame_limit
        self.edge_px = edge_px
        self.exporter: ExportCallable
        self.prober: Mp4Prober
        if exporter is None:
            self.exporter = export_mp4
            self.exporter_identity = exporter_identity or "ffmpeg-export-v1"
            if prober is not None:
                raise ValueError("default exporter must use the trusted ffprobe adapter")
            self.prober = probe_mp4
        else:
            if not exporter_identity:
                raise ValueError(
                    "custom preview exporter requires an explicit nonempty identity"
                )
            self.exporter = exporter
            self.exporter_identity = exporter_identity
            if prober is None:
                raise ValueError("custom preview exporter requires an explicit prober")
            self.prober = prober

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        source_video, summary, source_snapshot = _source_material(
            self.paths, project, token
        )
        ingest = _stage_state(project, StageName.INGEST)
        segment = _stage_state(project, StageName.SEGMENT)
        render = _stage_state(project, StageName.RENDER)
        source_directory = _artifact_path(
            self.paths, ingest, ArtifactRole.SOURCE_FRAMES, "frames"
        )
        proxy_directory = _artifact_path(
            self.paths, ingest, ArtifactRole.PROXY_FRAMES, "proxies"
        )
        mask_directory = _artifact_path(
            self.paths, segment, ArtifactRole.SUBJECT_MASKS, "masks"
        )
        render_directory = _artifact_path(
            self.paths, render, ArtifactRole.RENDER_FRAMES, "renders"
        )
        sources = _frame_inventory(
            source_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="源",
            expected_count=summary.frame_count,
            expected_size=(summary.width, summary.height),
            token=token,
        )
        proxies = _frame_inventory(
            proxy_directory,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            expected_count=sources.count,
            token=token,
        )
        masks = _frame_inventory(
            mask_directory,
            suffix="png",
            image_format="PNG",
            mode="L",
            label="遮罩",
            expected_count=sources.count,
            expected_sizes=proxies.sizes,
            token=token,
        )
        renders = _frame_inventory(
            render_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="渲染",
            expected_count=sources.count,
            expected_size=(summary.width, summary.height),
            token=token,
        )
        result_key = cache_key(
            StageName.COMPOSITE.value,
            {
                "segment_cache_key": segment.cache_key,
                "render_cache_key": render.cache_key,
                "source_inventory": sources.fingerprint,
                "mask_inventory": masks.fingerprint,
                "render_inventory": renders.fingerprint,
            },
            {
                "edge_px": self.edge_px,
                "preview_height": project.workflow.preview_height,
                "preview_frame_limit": self.preview_frame_limit,
                "exporter_identity": self.exporter_identity,
            },
            COMPOSITE_IMPLEMENTATION_VERSION,
        )

        def build_composites(staging: Path) -> None:
            for index, (source_path, render_path, mask_path) in enumerate(
                zip(sources.paths, renders.paths, masks.paths, strict=True), start=1
            ):
                token.raise_if_cancelled()
                with Image.open(source_path) as source_image:
                    foreground = np.asarray(
                        source_image.convert("RGB"), dtype=np.uint8
                    )
                with Image.open(render_path) as render_image:
                    background = np.asarray(
                        render_image.convert("RGB"), dtype=np.uint8
                    )
                with Image.open(mask_path) as mask_image:
                    alpha_image = np.asarray(mask_image.convert("L"), dtype=np.uint8)
                alpha = cast(
                    NDArray[np.uint8],
                    cv2.resize(
                        alpha_image,
                        (summary.width, summary.height),
                        interpolation=cv2.INTER_NEAREST,
                    ),
                )
                composite = composite_frame(
                    foreground, background, alpha, edge_px=self.edge_px
                )
                Image.fromarray(composite).save(staging / f"{index:06d}.png")
                emit(index, sources.count, f"合成全分辨率帧 {index}/{sources.count}")
            _assert_frame_snapshot(
                source_directory,
                sources,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="源",
                token=token,
            )
            _assert_frame_snapshot(
                proxy_directory,
                proxies,
                suffix="jpg",
                image_format="JPEG",
                mode="RGB",
                label="代理",
                token=token,
            )
            _assert_frame_snapshot(
                mask_directory,
                masks,
                suffix="png",
                image_format="PNG",
                mode="L",
                label="遮罩",
                token=token,
            )
            _assert_frame_snapshot(
                render_directory,
                renders,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="渲染",
                token=token,
            )
            token.raise_if_cancelled()

        composite_directory = self.paths.publisher.publish_tree(
            "composites", result_key, build_composites
        )
        composites = _frame_inventory(
            composite_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="合成",
            expected_count=sources.count,
            expected_size=(summary.width, summary.height),
            token=token,
        )
        preview_count = min(composites.count, self.preview_frame_limit)
        preview_size = _preview_size(
            (summary.width, summary.height), project.workflow.preview_height
        )

        def build_preview(staging: Path) -> None:
            token.raise_if_cancelled()
            preview_frames = staging / "frames"
            preview_frames.mkdir()
            for index, source_path in enumerate(
                composites.paths[:preview_count], start=1
            ):
                token.raise_if_cancelled()
                with Image.open(source_path) as image:
                    resized = image.resize(preview_size, Image.Resampling.LANCZOS)
                    resized.save(preview_frames / f"{index:06d}.png")
            output = staging / "composite-preview.mp4"
            exported = self.exporter(
                preview_frames,
                source_video,
                Fraction(summary.fps),
                preview_count,
                output,
                cancellation_check=token.raise_if_cancelled,
            )
            _validate_export_result(
                exported,
                output,
                fps=Fraction(summary.fps),
                frame_count=preview_count,
                has_audio=summary.has_audio,
                label="预览导出器",
                allow_duration_tolerance=False,
            )
            _assert_frame_snapshot(
                composite_directory,
                composites,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="合成",
                token=token,
            )
            _assert_file_snapshot(
                source_video,
                source_snapshot,
                "源视频",
                token,
            )
            _write_mp4_manifest(
                staging,
                cache_key_value=result_key,
                filename="composite-preview.mp4",
                result=exported,
                token=token,
            )
            for path in preview_frames.iterdir():
                path.unlink()
            preview_frames.rmdir()
            token.raise_if_cancelled()

        preview_directory = self.paths.publisher.publish_tree(
            "previews", result_key, build_preview
        )
        _validate_published_mp4(
            preview_directory,
            cache_key_value=result_key,
            filename="composite-preview.mp4",
            fps=Fraction(summary.fps),
            frame_count=preview_count,
            has_audio=summary.has_audio,
            prober=self.prober,
            token=token,
        )
        composite_relative = self.paths.relative(composite_directory)
        preview_relative = self.paths.relative(
            preview_directory / "composite-preview.mp4"
        )
        return StageResult(
            output_paths=(composite_relative, preview_relative),
            cache_key=result_key,
            artifacts={
                ArtifactRole.COMPOSITE_FRAMES: composite_relative,
                ArtifactRole.COMPOSITE_PREVIEW: preview_relative,
            },
        )


class ExportWorkflowService:
    def __init__(
        self,
        paths: WorkflowPaths,
        *,
        exporter: ExportCallable | None = None,
        exporter_identity: str | None = None,
        prober: Mp4Prober | None = None,
    ) -> None:
        self.paths = paths
        self.exporter: ExportCallable
        self.prober: Mp4Prober
        if exporter is None:
            self.exporter = export_mp4
            self.exporter_identity = exporter_identity or "ffmpeg-export-v1"
            if prober is not None:
                raise ValueError("default exporter must use the trusted ffprobe adapter")
            self.prober = probe_mp4
        else:
            if not exporter_identity:
                raise ValueError(
                    "custom final exporter requires an explicit nonempty identity"
                )
            self.exporter = exporter
            self.exporter_identity = exporter_identity
            if prober is None:
                raise ValueError("custom final exporter requires an explicit prober")
            self.prober = prober

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        source_video, summary, source_snapshot = _source_material(
            self.paths, project, token
        )
        composite = _stage_state(project, StageName.COMPOSITE)
        composite_directory = _artifact_path(
            self.paths,
            composite,
            ArtifactRole.COMPOSITE_FRAMES,
            "composites",
        )
        frames = _frame_inventory(
            composite_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="合成",
            expected_count=summary.frame_count,
            expected_size=(summary.width, summary.height),
            token=token,
        )
        result_key = cache_key(
            StageName.EXPORT.value,
            {
                "composite_cache_key": composite.cache_key,
                "composite_inventory": frames.fingerprint,
            },
            {
                "fps": summary.fps,
                "frame_count": frames.count,
                "source_size": summary.size,
                "source_sha256": summary.sha256,
                "has_audio": summary.has_audio,
                "exporter_identity": self.exporter_identity,
            },
            EXPORT_IMPLEMENTATION_VERSION,
        )

        def build(staging: Path) -> None:
            token.raise_if_cancelled()
            output = staging / "final.mp4"
            exported = self.exporter(
                composite_directory,
                source_video,
                Fraction(summary.fps),
                frames.count,
                output,
                cancellation_check=token.raise_if_cancelled,
            )
            _validate_export_result(
                exported,
                output,
                fps=Fraction(summary.fps),
                frame_count=frames.count,
                has_audio=summary.has_audio,
                label="最终导出器",
                allow_duration_tolerance=False,
            )
            _assert_frame_snapshot(
                composite_directory,
                frames,
                suffix="png",
                image_format="PNG",
                mode="RGB",
                label="合成",
                token=token,
            )
            _assert_file_snapshot(
                source_video,
                source_snapshot,
                "源视频",
                token,
            )
            _write_mp4_manifest(
                staging,
                cache_key_value=result_key,
                filename="final.mp4",
                result=exported,
                token=token,
            )
            token.raise_if_cancelled()

        output_directory = self.paths.publisher.publish_tree(
            "exports", result_key, build
        )
        _validate_published_mp4(
            output_directory,
            cache_key_value=result_key,
            filename="final.mp4",
            fps=Fraction(summary.fps),
            frame_count=frames.count,
            has_audio=summary.has_audio,
            prober=self.prober,
            token=token,
        )
        relative = self.paths.relative(output_directory / "final.mp4")
        emit(1, 1, "验证并发布最终视频 1/1")
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.EXPORT_VIDEO: relative},
        )
