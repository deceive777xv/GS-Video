from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Sequence
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
from gs_video.domain.contracts import MaskSequence, Prompt, StageResult
from gs_video.domain.errors import RepairableError
from gs_video.domain.models import (
    ArtifactRole,
    Project,
    StageName,
    StageState,
    StageStatus,
    VideoSummary,
)
from gs_video.media.export import ExportResult, export_mp4
from gs_video.media.ingest import extract_proxy_frames, extract_source_frames
from gs_video.pipeline.artifacts import ArtifactPublisher, validate_cache_key
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.project.cache import cache_key
from gs_video.scene.camera import OrbitCamera
from gs_video.segmentation.paths import has_reparse_component


INGEST_IMPLEMENTATION_VERSION = "media-ingest-v1"
SEGMENT_IMPLEMENTATION_VERSION = "segment-adapter-v1"
CAMERA_IMPLEMENTATION_VERSION = "opencv-camera-adapter-v1"
TRAJECTORY_IMPLEMENTATION_VERSION = "trajectory-map-v1"
COMPOSITE_IMPLEMENTATION_VERSION = "full-resolution-composite-v1"
EXPORT_IMPLEMENTATION_VERSION = "verified-export-v1"
_FRAME_NAME = re.compile(r"^(\d{6})\.(png|jpg)$")


ProjectMutation = Callable[[Project], None]
ProjectUpdater = Callable[[ProjectMutation], Project]
ExportCallable = Callable[[Path, Path, Fraction, int, Path], ExportResult]


class MediaIngestBackend(Protocol):
    identity: str

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]: ...

    def extract_proxy_frames(
        self, source: Path, output_dir: Path, max_height: int
    ) -> list[Path]: ...


class ForegroundSegmenterLike(Protocol):
    backend: object
    worker_prefix: Sequence[str]
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


@dataclass(frozen=True)
class WorkflowPaths:
    root: Path
    update_project: ProjectUpdater | None = None
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

    @property
    def count(self) -> int:
        return len(self.paths)


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


def _sha256(path: Path, label: str) -> str:
    before = _ordinary_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RepairableError(f"{label}读取前身份发生变化")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            after_handle = os.fstat(stream.fileno())
    except OSError as exc:
        raise RepairableError(f"{label}不可读") from exc
    after_path = _ordinary_file(path, label)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity_before != (
        after_handle.st_dev,
        after_handle.st_ino,
        after_handle.st_size,
        after_handle.st_mtime_ns,
    ) or identity_before != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
    ):
        raise RepairableError(f"{label}读取期间身份发生变化")
    return digest.hexdigest()


def _source_material(paths: WorkflowPaths, project: Project) -> tuple[Path, VideoSummary]:
    summary = project.workflow.source_summary
    if summary is None or project.source_video is None:
        raise RepairableError("尚未导入源视频")
    relative = Path(project.source_video)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("source",):
        raise RepairableError("源视频路径不属于项目 source 目录")
    source = paths.root / relative
    metadata = _ordinary_file(source, "源视频")
    if metadata.st_size != summary.size or _sha256(source, "源视频") != summary.sha256:
        raise RepairableError("源视频与已登记摘要不一致")
    return source, summary


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
) -> _FrameInventory:
    try:
        metadata = directory.lstat()
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
            file_metadata = _ordinary_file(path, f"{label}帧")
            match = _FRAME_NAME.fullmatch(path.name)
            if match is None or match.group(2) != suffix:
                raise RepairableError(f"{label}帧文件名无效")
            with Image.open(path) as image:
                image.load()
                if image.format != image_format or image.mode != mode:
                    raise RepairableError(
                        f"{label}帧必须是 {mode} {image_format} 图像"
                    )
                size = image.size
            if expected_size is not None and size != expected_size:
                raise RepairableError(f"{label}帧尺寸与源视频不一致")
            if expected_sizes is not None and size != expected_sizes[index]:
                raise RepairableError(f"{label}帧尺寸与上游帧不一致")
            sizes.append(size)
            fingerprint.update(path.name.encode("ascii"))
            fingerprint.update(str(file_metadata.st_size).encode("ascii"))
            fingerprint.update(_sha256(path, f"{label}帧").encode("ascii"))
    except (OSError, UnidentifiedImageError) as exc:
        raise RepairableError(f"{label}帧不可读") from exc
    return _FrameInventory(ordered, tuple(sizes), fingerprint.hexdigest())


def _model_identity(path: Path, label: str) -> dict[str, object]:
    metadata = _ordinary_file(path, label)
    return {
        "filename": path.name,
        "size": metadata.st_size,
        "sha256": _sha256(path, label),
    }


def _callable_identity(value: object) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


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
        source, summary = _source_material(self.paths, project)
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
            )
            discovered_count = inventory.count
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
            )
            if any(
                width > summary.width
                or height > min(summary.height, self.proxy_height)
                for width, height in inventory.sizes
            ):
                raise RepairableError("代理帧尺寸超过源视频或代理高度上限")
            token.raise_if_cancelled()

        proxy_directory = self.paths.publisher.publish_tree(
            "proxies", result_key, build_proxies
        )
        _frame_inventory(
            proxy_directory,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            expected_count=source_inventory.count,
        )
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
        backend_value = getattr(self.segmenter.backend, "value", self.segmenter.backend)
        identity = {
            "backend": str(backend_value),
            "worker_prefix": list(self.segmenter.worker_prefix),
            "config": _model_identity(self.segmenter.model_config, "分割配置"),
            "checkpoint": _model_identity(self.segmenter.checkpoint, "分割模型"),
        }
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
            )
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("masks", result_key, build)
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
        backend_identity: str = "opencv-camera-solver-v1",
    ) -> None:
        if not backend_identity:
            raise ValueError("camera backend identity must not be empty")
        self.paths = paths
        self.solver = solver or OpenCvCameraSolver()
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
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("camera", result_key, build)
        relative = self.paths.relative(output / "solution.json")
        read_camera_solution(self.paths.root / relative)
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
        _ordinary_file(solution_path, "相机求解产物")
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
        result_key = cache_key(
            StageName.MAP_TRAJECTORY.value,
            {
                "solve_cache_key": solve.cache_key,
                "camera_artifact_sha256": _sha256(solution_path, "相机求解产物"),
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
            target = OrbitCamera(
                target=camera.target,
                distance=camera.distance,
                yaw=camera.yaw,
                pitch=camera.pitch,
                fov_y_degrees=camera.fov_y_degrees,
            )
            mapped = map_trajectory(solution, target.camera_to_world(), workflow.motion_scale)
            write_mapped_trajectory(
                staging / "trajectory.json",
                MappedTrajectory(camera.fov_y_degrees, mapped),
            )
            token.raise_if_cancelled()

        output = self.paths.publisher.publish_tree("trajectories", result_key, build)
        relative = self.paths.relative(output / "trajectory.json")
        read_mapped_trajectory(self.paths.root / relative)
        emit(1, 1, "映射目标相机轨迹 1/1")
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.MAPPED_TRAJECTORY: relative},
        )


def _preview_size(source: tuple[int, int], maximum_height: int) -> tuple[int, int]:
    width, height = source
    target_height = min(height, maximum_height)
    if target_height % 2:
        target_height -= 1
    target_height = max(2, target_height)
    target_width = int(round(width * target_height / height))
    if target_width % 2:
        target_width -= 1
    target_width = max(2, min(width if width % 2 == 0 else width - 1, target_width))
    return target_width, target_height


class CompositeWorkflowService:
    def __init__(
        self,
        paths: WorkflowPaths,
        *,
        preview_frame_limit: int = 150,
        edge_px: int = 1,
        exporter: ExportCallable = export_mp4,
        exporter_identity: str = "ffmpeg-export-v1",
    ) -> None:
        if type(preview_frame_limit) is not int or preview_frame_limit <= 0:
            raise ValueError("preview_frame_limit must be a positive integer")
        if type(edge_px) is not int or not 0 <= edge_px <= 3:
            raise ValueError("edge_px must be an integer between 0 and 3")
        if not exporter_identity:
            raise ValueError("exporter identity must not be empty")
        self.paths = paths
        self.preview_frame_limit = preview_frame_limit
        self.edge_px = edge_px
        self.exporter = exporter
        self.exporter_identity = exporter_identity

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        source_video, summary = _source_material(self.paths, project)
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
        )
        proxies = _frame_inventory(
            proxy_directory,
            suffix="jpg",
            image_format="JPEG",
            mode="RGB",
            label="代理",
            expected_count=sources.count,
        )
        masks = _frame_inventory(
            mask_directory,
            suffix="png",
            image_format="PNG",
            mode="L",
            label="遮罩",
            expected_count=sources.count,
            expected_sizes=proxies.sizes,
        )
        renders = _frame_inventory(
            render_directory,
            suffix="png",
            image_format="PNG",
            mode="RGB",
            label="渲染",
            expected_count=sources.count,
            expected_size=(summary.width, summary.height),
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
            )
            if Path(exported.output).absolute() != output.absolute():
                raise RepairableError("预览导出器返回了错误的输出路径")
            metadata = _ordinary_file(output, "合成预览")
            if metadata.st_size <= 0:
                raise RepairableError("合成预览为空")
            for path in preview_frames.iterdir():
                path.unlink()
            preview_frames.rmdir()
            token.raise_if_cancelled()

        preview_directory = self.paths.publisher.publish_tree(
            "previews", result_key, build_preview
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
        exporter: ExportCallable = export_mp4,
        exporter_identity: str = "ffmpeg-export-v1",
    ) -> None:
        if not exporter_identity:
            raise ValueError("exporter identity must not be empty")
        self.paths = paths
        self.exporter = exporter
        self.exporter_identity = exporter_identity

    def run(
        self,
        project: Project,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageResult:
        token.raise_if_cancelled()
        source_video, summary = _source_material(self.paths, project)
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
            )
            if Path(exported.output).absolute() != output.absolute():
                raise RepairableError("最终导出器返回了错误的输出路径")
            metadata = _ordinary_file(output, "最终视频")
            if metadata.st_size <= 0:
                raise RepairableError("最终视频为空")
            token.raise_if_cancelled()

        output_directory = self.paths.publisher.publish_tree(
            "exports", result_key, build
        )
        relative = self.paths.relative(output_directory / "final.mp4")
        emit(1, 1, "验证并发布最终视频 1/1")
        return StageResult(
            output_paths=(relative,),
            cache_key=result_key,
            artifacts={ArtifactRole.EXPORT_VIDEO: relative},
        )
