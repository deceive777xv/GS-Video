from __future__ import annotations

import asyncio
import stat
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
)

from gs_video.api.assets import AssetInspector
from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.api.workflow import WorkerPreviewService
from gs_video.domain.contracts import SegmentationBackend
from gs_video.domain.errors import GsVideoError
from gs_video.environment.doctor import EnvironmentDoctor
from gs_video.pipeline.services import (
    CameraSolveWorkflowService,
    CompositeWorkflowService,
    ExportWorkflowService,
    MediaIngestService,
    RendererWorkflowService,
    SegmentWorkflowService,
    TrajectoryMapWorkflowService,
    WorkflowPaths,
)
from gs_video.pipeline.workflow import WorkflowServices, build_mvp_workflow
from gs_video.project.repository import ProjectRepository
from gs_video.scene.worker_client import RendererWorkerClient
from gs_video.segmentation.client import VideoSegmenterClient
from gs_video.segmentation.paths import has_reparse_component, is_wsl_prefix


_TAURI_ORIGINS = (
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
)


class WorkflowRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_root: Path
    model_root: Path
    segmentation_backend: SegmentationBackend
    segmentation_worker_prefix: tuple[str, ...]
    segmentation_model_config: Path
    segmentation_checkpoint: Path
    renderer_worker_prefix: tuple[str, ...]
    renderer_sh_degree: int = Field(default=3, ge=0, le=3)
    available_vram_limit_mb: int = Field(default=8192, ge=1024, le=8192)

    _workspace_root: Path | None = PrivateAttr(default=None)
    _runtime_path: Path | None = PrivateAttr(default=None)

    @field_validator(
        "project_root",
        "model_root",
        "segmentation_model_config",
        "segmentation_checkpoint",
    )
    @classmethod
    def absolute_paths(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("runtime paths must be absolute")
        return value

    @field_validator("segmentation_worker_prefix", "renderer_worker_prefix")
    @classmethod
    def worker_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if (
            not value
            or any(
                not item
                or len(item) > 32767
                or "\x00" in item
                or any(ord(character) < 32 for character in item)
                for item in value
            )
        ):
            raise ValueError("worker argv must contain nonempty safe entries")
        return value

    @field_validator("segmentation_worker_prefix")
    @classmethod
    def segmentation_worker_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if is_wsl_prefix(value):
            separator = value.index("--")
            if (
                value[0].lower() not in {"wsl", "wsl.exe"}
                or value.count("--") != 1
                or separator + 1 >= len(value)
                or not PurePosixPath(value[separator + 1]).is_absolute()
            ):
                raise ValueError("WSL worker prefix must name an absolute Linux interpreter")
            return value
        if not Path(value[0]).is_absolute():
            raise ValueError("worker executable path must be absolute")
        return value

    @field_validator("renderer_worker_prefix")
    @classmethod
    def renderer_worker_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not Path(value[0]).is_absolute():
            raise ValueError("renderer worker executable path must be absolute")
        return value


def _ordinary_file(path: Path, label: str) -> None:
    metadata = path.lstat()
    if (
        has_reparse_component(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{label} must be an ordinary file")


def _resolved_inside(
    path: Path,
    root: Path,
    label: str,
    *,
    root_label: str = "application workspace",
) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must remain inside {root_label}")
    return resolved


def _validate_loaded_config(
    config: WorkflowRuntimeConfig, *, workspace: Path, runtime_path: Path
) -> None:
    model_root = _resolved_inside(config.model_root, workspace, "model_root")
    if not model_root.is_dir() or has_reparse_component(model_root):
        raise ValueError("model_root must be an ordinary directory")
    project_root = config.project_root
    if project_root == workspace or not project_root.is_relative_to(workspace):
        raise ValueError("project_root must be a child of the application workspace")
    normalized_project = project_root.resolve(strict=False)
    if normalized_project != project_root or has_reparse_component(project_root.parent):
        raise ValueError("project_root contains an unexpected alias or reparse point")
    if runtime_path == project_root or runtime_path.is_relative_to(project_root):
        raise ValueError("project_root must not contain the runtime configuration")
    for path, label in (
        (config.segmentation_model_config, "segmentation_model_config"),
        (config.segmentation_checkpoint, "segmentation_checkpoint"),
    ):
        resolved = _resolved_inside(path, model_root, label, root_label="model_root")
        _ordinary_file(resolved, label)
    if not is_wsl_prefix(config.segmentation_worker_prefix):
        executable = _resolved_inside(
            Path(config.segmentation_worker_prefix[0]), workspace, "segmentation worker"
        )
        _ordinary_file(executable, "segmentation worker")
    renderer_executable = _resolved_inside(
        Path(config.renderer_worker_prefix[0]), workspace, "renderer worker"
    )
    _ordinary_file(renderer_executable, "renderer worker")


def load_runtime_config(path: Path) -> WorkflowRuntimeConfig:
    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError("runtime configuration path must be absolute")
    if requested.suffix.lower() != ".json":
        raise ValueError("runtime configuration must be a JSON file")
    if has_reparse_component(requested):
        raise ValueError("runtime configuration contains a link or reparse point")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError("runtime configuration is unavailable") from error
    if resolved != requested:
        raise ValueError("runtime configuration contains an unexpected alias")
    _ordinary_file(resolved, "runtime configuration")
    workspace = resolved.parent.resolve(strict=True)
    if has_reparse_component(workspace):
        raise ValueError("application workspace contains a link or reparse point")
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError("runtime configuration is unreadable") from error
    if not payload or len(payload) > 64 * 1024:
        raise ValueError("runtime configuration must be between 1 byte and 64 KiB")
    config = WorkflowRuntimeConfig.model_validate_json(payload, strict=True)
    _validate_loaded_config(config, workspace=workspace, runtime_path=resolved)
    config._workspace_root = workspace
    config._runtime_path = resolved
    return config


class WorkerRegistry:
    def __init__(self, *workers: Any) -> None:
        self._workers = tuple(workers)

    async def terminate_all(self) -> None:
        await asyncio.gather(
            *(worker.terminate_all() for worker in self._workers),
            return_exceptions=True,
        )


def assemble_api_services(
    config: WorkflowRuntimeConfig, session_token: SecretStr
) -> tuple[ApiSettings, ApiServices]:
    if config._workspace_root is not None:
        assert config._runtime_path is not None
        _validate_loaded_config(
            config,
            workspace=config._workspace_root,
            runtime_path=config._runtime_path,
        )
    config.project_root.mkdir(parents=True, exist_ok=True)
    if has_reparse_component(config.project_root):
        raise ValueError("project_root contains a link or reparse point")
    repository = ProjectRepository(config.project_root)
    if repository.path.exists():
        project = repository.load()
    else:
        project = repository.create("GS Video project")
        repository.save(project)
    segmentation = VideoSegmenterClient(
        backend=config.segmentation_backend,
        worker_prefix=config.segmentation_worker_prefix,
        model_config=config.segmentation_model_config,
        checkpoint=config.segmentation_checkpoint,
        log_path=config.project_root / "logs" / "segmentation-worker.log",
    )
    renderer = RendererWorkerClient(
        worker_prefix=config.renderer_worker_prefix,
        log_path=config.project_root / "logs" / "renderer-worker.log",
    )
    paths = WorkflowPaths(config.project_root, update_project=repository.update)
    workflow_services = WorkflowServices(
        media_ingest=MediaIngestService(paths),
        segmenter=SegmentWorkflowService(paths, segmentation),
        camera_solver=CameraSolveWorkflowService(paths),
        trajectory_mapper=TrajectoryMapWorkflowService(paths),
        renderer=RendererWorkflowService(
            paths,
            renderer,
            sh_degree=config.renderer_sh_degree,
            available_vram_limit_mb=config.available_vram_limit_mb,
        ),
        compositor=CompositeWorkflowService(paths),
        exporter=ExportWorkflowService(paths),
    )
    runner = build_mvp_workflow(
        workflow_services,
        project,
        save=repository.save,
        persist_stage=repository.update_stage,
        compare_and_set_stage=repository.compare_and_set_stage,
        claim_stage=repository.claim_stage,
    )

    identity_lock = Lock()
    identity_attempted = False
    renderer_identity: object | None = None

    def probe_identity() -> object | None:
        nonlocal identity_attempted, renderer_identity
        with identity_lock:
            if not identity_attempted:
                try:
                    renderer_identity = renderer.probe()
                except (GsVideoError, OSError, ValueError):
                    renderer_identity = None
                identity_attempted = True
            return renderer_identity

    def external_cuda_probe() -> tuple[bool, int]:
        identity = probe_identity()
        device = getattr(identity, "device", "")
        available = isinstance(device, str) and device.lower().startswith("cuda")
        return (
            available,
            config.available_vram_limit_mb if available else 0,
        )

    def external_renderer_probe() -> tuple[str | None, str | None]:
        identity = probe_identity()
        torch_version = getattr(identity, "torch", None)
        gsplat_version = getattr(identity, "gsplat", None)
        return (
            torch_version if isinstance(torch_version, str) else None,
            gsplat_version if isinstance(gsplat_version, str) else None,
        )

    doctor = EnvironmentDoctor(
        cuda_probe=external_cuda_probe,
        segmentation_backend=config.segmentation_backend,
        worker_prefix=config.segmentation_worker_prefix,
        model_config=config.segmentation_model_config,
        checkpoint=config.segmentation_checkpoint,
        check_renderer=True,
        renderer_probe=external_renderer_probe,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=session_token,
        allowed_origins=_TAURI_ORIGINS,
        task_workers=1,
    )
    services = ApiServices(
        project_repository=repository,
        environment_doctor=doctor,
        pipeline_runner=runner,
        worker_registry=WorkerRegistry(segmentation, renderer),
        preview_service=WorkerPreviewService(
            renderer,
            available_vram_limit_mb=config.available_vram_limit_mb,
        ),
        asset_inspector=AssetInspector(),
    )
    return settings, services


__all__ = [
    "WorkflowRuntimeConfig",
    "WorkerRegistry",
    "assemble_api_services",
    "load_runtime_config",
]
