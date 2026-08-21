from __future__ import annotations

import asyncio
from ipaddress import ip_address
import os
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, cast
from urllib.parse import urlsplit

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
from gs_video.api.workflow import PreviewArtifactStore, WorkerPreviewService
from gs_video.camera.vipe_solver import VipeCameraSolver
from gs_video.domain.contracts import SegmentationBackend
from gs_video.domain.errors import GsVideoError
from gs_video.environment.doctor import EnvironmentDoctor
from gs_video.environment.repair import EnvironmentRepairManager
from gs_video.environment.vram import VramBudgetManager
from gs_video.media.toolchain import executable_name
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
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.gpu import GpuAdmissionGate
from gs_video.pipeline.workflow import WorkflowServices, build_mvp_workflow
from gs_video.project.assets import AssetKind as LibraryAssetKind, AssetLibrary
from gs_video.project.catalog import ProjectCatalog
from gs_video.project.manager import (
    ActivePipelineRunner,
    ActivePreviewArtifactStore,
    ActiveProjectManager,
    ActiveProjectRepository,
    ActiveProjectSession,
)
from gs_video.project.repository import ProjectInstanceLock
from gs_video.scene.worker_client import RendererWorkerClient
from gs_video.scene.preview_session import PreviewSession
from gs_video.segmentation.client import VideoSegmenterClient
from gs_video.segmentation.paths import has_reparse_component, is_wsl_prefix
from gs_video.storage.artifacts import ArtifactStore
from gs_video.storage.legacy import migrate_legacy_project_artifacts
from gs_video.storage.layout import StorageLayoutManager


_TAURI_ORIGINS = (
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
)


def validate_browser_origins(origins: tuple[str, ...]) -> tuple[str, ...]:
    if len(origins) > 16:
        raise ValueError("browser origin allowlist is too large")
    normalized: list[str] = []
    for origin in origins:
        if (
            not origin
            or len(origin) > 2048
            or any(ord(character) < 32 or ord(character) == 127 for character in origin)
        ):
            raise ValueError("browser origin must be a bounded HTTP loopback origin")
        parsed = urlsplit(origin)
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("browser origin port is invalid") from error
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.hostname is None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or (port is not None and port == 0)
        ):
            raise ValueError("browser origin must be an HTTP loopback origin without a path")
        hostname = parsed.hostname.lower()
        if hostname == "localhost":
            rendered_host = hostname
        else:
            try:
                address = ip_address(hostname)
            except ValueError as error:
                raise ValueError("browser origin host must be loopback") from error
            if not address.is_loopback:
                raise ValueError("browser origin host must be loopback")
            rendered_host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        rendered_port = "" if port is None else f":{port}"
        normalized.append(f"{parsed.scheme.lower()}://{rendered_host}{rendered_port}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("browser origins must be unique")
    return tuple(normalized)


class WorkflowRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_root: Path
    model_root: Path
    segmentation_backend: SegmentationBackend
    segmentation_worker_prefix: tuple[str, ...]
    segmentation_model_config: Path
    segmentation_checkpoint: Path
    camera_worker_prefix: tuple[str, ...]
    renderer_worker_prefix: tuple[str, ...]
    renderer_sh_degree: int = Field(default=3, ge=0, le=3)
    available_vram_limit_mb: int = Field(default=8192, ge=1024)

    _workspace_root: Path | None = PrivateAttr(default=None)
    _runtime_path: Path | None = PrivateAttr(default=None)
    _allow_missing_resources: bool = PrivateAttr(default=False)

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

    @field_validator(
        "segmentation_worker_prefix", "camera_worker_prefix", "renderer_worker_prefix"
    )
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

    @field_validator("camera_worker_prefix", "renderer_worker_prefix")
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
    strict: bool = True,
) -> Path:
    try:
        resolved = path.resolve(strict=strict)
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError(f"{label} must remain inside {root_label}")
    return resolved


def _validate_loaded_config(
    config: WorkflowRuntimeConfig,
    *,
    workspace: Path,
    runtime_path: Path,
    allow_missing_resources: bool = False,
) -> None:
    model_root = _resolved_inside(
        config.model_root,
        workspace,
        "model_root",
        strict=not allow_missing_resources,
    )
    if (
        (model_root.exists() and not model_root.is_dir())
        or has_reparse_component(model_root)
        or (not model_root.exists() and not allow_missing_resources)
    ):
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
        resolved = _resolved_inside(
            path,
            model_root,
            label,
            root_label="model_root",
            strict=not allow_missing_resources,
        )
        if resolved.exists():
            _ordinary_file(resolved, label)
        elif not allow_missing_resources:
            raise ValueError(f"{label} is unavailable")
    if not is_wsl_prefix(config.segmentation_worker_prefix):
        executable = _resolved_inside(
            Path(config.segmentation_worker_prefix[0]),
            workspace,
            "segmentation worker",
            strict=not allow_missing_resources,
        )
        if executable.exists():
            _ordinary_file(executable, "segmentation worker")
        elif not allow_missing_resources:
            raise ValueError("segmentation worker is unavailable")
    renderer_executable = _resolved_inside(
        Path(config.renderer_worker_prefix[0]),
        workspace,
        "renderer worker",
        strict=not allow_missing_resources,
    )
    if renderer_executable.exists():
        _ordinary_file(renderer_executable, "renderer worker")
    elif not allow_missing_resources:
        raise ValueError("renderer worker is unavailable")
    camera_executable = _resolved_inside(
        Path(config.camera_worker_prefix[0]),
        workspace,
        "camera worker",
        strict=not allow_missing_resources,
    )
    if camera_executable.exists():
        _ordinary_file(camera_executable, "camera worker")
    elif not allow_missing_resources:
        raise ValueError("camera worker is unavailable")


def load_runtime_config(
    path: Path, *, allow_missing_resources: bool = False
) -> WorkflowRuntimeConfig:
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
    _validate_loaded_config(
        config,
        workspace=workspace,
        runtime_path=resolved,
        allow_missing_resources=allow_missing_resources,
    )
    config._workspace_root = workspace
    config._runtime_path = resolved
    config._allow_missing_resources = allow_missing_resources
    return config


class WorkerRegistry:
    def __init__(
        self,
        *workers: Any,
        gpu_gate: GpuAdmissionGate | None = None,
        project_manager: ActiveProjectManager | None = None,
    ) -> None:
        self._workers = tuple(workers)
        self._gpu_gate = gpu_gate
        self._project_manager = project_manager

    async def terminate_all(self) -> None:
        if self._gpu_gate is not None:
            self._gpu_gate.close()
        try:
            await asyncio.gather(
                *(worker.terminate_all() for worker in self._workers),
                return_exceptions=True,
            )
        finally:
            if self._project_manager is not None:
                self._project_manager.close()


def assemble_api_services(
    config: WorkflowRuntimeConfig,
    session_token: SecretStr,
    *,
    browser_origins: tuple[str, ...] = (),
) -> tuple[ApiSettings, ApiServices]:
    allowed_browser_origins = validate_browser_origins(browser_origins)
    if config._workspace_root is not None:
        assert config._runtime_path is not None
        _validate_loaded_config(
            config,
            workspace=config._workspace_root,
            runtime_path=config._runtime_path,
            allow_missing_resources=config._allow_missing_resources,
        )
    runtime_root = config._workspace_root or config.project_root.parent.parent
    legacy_data_root = config.project_root.parent.parent
    storage_layout = StorageLayoutManager(
        legacy_data_root,
        runtime_root / "user-settings.json",
        protected_roots=(config.model_root,),
    )
    data_root = storage_layout.project_library_root
    cache_library_root = storage_layout.cache_root
    catalog = ProjectCatalog(data_root)
    if not catalog.list():
        try:
            configured_relative = config.project_root.relative_to(legacy_data_root)
        except ValueError:
            configured_relative = Path("projects") / "default"
        adoption_candidates = (
            data_root / configured_relative,
            data_root / "projects" / "default",
            legacy_data_root.parent / "projects" / "default",
        )
        adoption_root = next(
            (
                candidate
                for candidate in adoption_candidates
                if (candidate / "project.json").is_file()
            ),
            None,
        )
        if adoption_root is not None:
            if not adoption_root.is_relative_to(legacy_data_root):
                staged_adoption = data_root / "projects" / "default"
                staged_adoption.parent.mkdir(exist_ok=True)
                if staged_adoption.exists():
                    raise ValueError("legacy project adoption target already exists")
                os.replace(adoption_root, staged_adoption)
                adoption_root = staged_adoption
            catalog.adopt(adoption_root)
        else:
            try:
                config.project_root.rmdir()
            except OSError:
                pass
    asset_library = AssetLibrary(data_root / "assets")
    artifact_store = ArtifactStore(cache_library_root)
    asset_inspector = AssetInspector()

    def migrate_legacy_assets() -> None:
        for summary in catalog.list():
            repository = catalog.repository(summary.project_id)
            project = repository.load()
            imported_paths: list[Path] = []
            updates: dict[str, object] = {}
            legacy_inputs = (
                (
                    "source_video",
                    project.source_video,
                    LibraryAssetKind.VIDEO,
                    "source_video",
                ),
                (
                    "scene_ply",
                    project.scene_ply,
                    LibraryAssetKind.PLY,
                    "scene_ply",
                ),
            )
            for field, relative_value, library_kind, inspector_kind in legacy_inputs:
                asset_field = f"{field}_asset_id"
                if getattr(project, asset_field) is not None or relative_value is None:
                    continue
                relative = Path(relative_value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("legacy project input path is unsafe")
                source = (repository.root / relative).absolute()
                resolved_root = repository.root.resolve(strict=True)
                resolved_source = source.resolve(strict=True)
                if (
                    resolved_source != source
                    or not resolved_source.is_relative_to(resolved_root / "source")
                    or has_reparse_component(resolved_source)
                ):
                    raise ValueError("legacy project input path is unsafe")

                def inspect(
                    path: Path,
                    size: int,
                    sha256: str,
                    *,
                    kind: str = inspector_kind,
                ) -> Any:
                    return asset_inspector.inspect(
                        kind, path, size=size, sha256=sha256
                    )

                record, _ = asset_library.import_file(
                    library_kind, resolved_source, inspect
                )
                updates[asset_field] = record.asset_id
                updates[field] = None
                if library_kind is LibraryAssetKind.VIDEO:
                    updates["source_summary"] = record.video_summary
                else:
                    updates["scene_summary"] = record.scene_summary
                imported_paths.append(resolved_source)
            if not updates:
                continue

            def apply_migration(latest: Any) -> None:
                for key, value in updates.items():
                    if key == "source_summary":
                        latest.workflow.source_summary = value
                    elif key == "scene_summary":
                        latest.workflow.scene_summary = value
                    else:
                        setattr(latest, key, value)

            repository.update(apply_migration)
            for path in imported_paths:
                path.unlink()

    migrate_legacy_assets()
    for summary in catalog.list():
        legacy_repository = catalog.repository(summary.project_id)
        migrate_legacy_project_artifacts(
            legacy_repository.root, summary.project_id, artifact_store
        )
    log_root = legacy_data_root / "logs"
    log_root.mkdir(exist_ok=True)
    gpu_gate = GpuAdmissionGate()
    segmentation = VideoSegmenterClient(
        backend=config.segmentation_backend,
        worker_prefix=config.segmentation_worker_prefix,
        model_config=config.segmentation_model_config,
        checkpoint=config.segmentation_checkpoint,
        log_path=log_root / "segmentation-worker.log",
        gpu_gate=gpu_gate,
        cache_root=config.model_root / ".cache",
    )
    renderer = RendererWorkerClient(
        worker_prefix=config.renderer_worker_prefix,
        log_path=log_root / "renderer-worker.log",
        gpu_gate=gpu_gate,
    )
    camera_solver = VipeCameraSolver(
        config.camera_worker_prefix,
        gpu_gate=gpu_gate,
        cache_root=runtime_root / "camera" / ".cache",
        log_path=log_root / "vipe-worker.log",
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

    def total_vram_probe() -> int:
        identity = probe_identity()
        return int(getattr(identity, "total_vram_mb", 0))

    vram_budget = VramBudgetManager(
        runtime_root / "user-settings.json",
        total_vram_probe=total_vram_probe,
        initial_limit_mb=config.available_vram_limit_mb,
    )
    preview_session = PreviewSession(
        worker_prefix=config.renderer_worker_prefix,
        sh_degree=config.renderer_sh_degree,
        available_vram_limit_mb=config.available_vram_limit_mb,
        vram_limit_provider=vram_budget.current_limit_mb,
        log_path=log_root / "preview-session-worker.log",
        gpu_gate=gpu_gate,
    )
    def resolve_asset(asset_id: str, kind: str) -> Path:
        expected = (
            LibraryAssetKind.VIDEO if kind == "video" else LibraryAssetKind.PLY
        )
        return asset_library.resolve(asset_id, expected)

    def build_project_session(project_id: str) -> ActiveProjectSession:
        repository = catalog.repository(project_id)
        project_lock = ProjectInstanceLock.acquire(repository.root)
        try:
            project = repository.reconcile_interrupted_runs()
            paths = WorkflowPaths(
                repository.root,
                update_project=repository.update,
                resolve_asset=resolve_asset,
                artifact_store=artifact_store,
            )
            workflow_services = WorkflowServices(
                media_ingest=MediaIngestService(paths),
                segmenter=SegmentWorkflowService(paths, segmentation),
                camera_solver=CameraSolveWorkflowService(
                    paths,
                    camera_solver,
                    backend_identity=camera_solver.identity,
                ),
                trajectory_mapper=TrajectoryMapWorkflowService(paths),
                renderer=RendererWorkflowService(
                    paths,
                    renderer,
                    sh_degree=config.renderer_sh_degree,
                    available_vram_limit_mb=config.available_vram_limit_mb,
                    vram_limit_provider=vram_budget.current_limit_mb,
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
            return ActiveProjectSession(
                repository=repository,
                project_lock=project_lock,
                pipeline_runner=runner,
                preview_artifacts=PreviewArtifactStore(
                    artifact_store.project_root(project_id)
                ),
            )
        except BaseException:
            project_lock.close()
            raise

    project_manager = ActiveProjectManager(catalog, build_project_session)
    repository = ActiveProjectRepository(project_manager)
    runner = ActivePipelineRunner(project_manager)

    def external_cuda_probe() -> tuple[bool, int]:
        identity = probe_identity()
        device = getattr(identity, "device", "")
        available = isinstance(device, str) and device.lower().startswith("cuda")
        return (
            available,
            int(getattr(identity, "total_vram_mb", 0)) if available else 0,
        )

    def external_renderer_probe() -> tuple[str | None, str | None]:
        identity = probe_identity()
        torch_version = getattr(identity, "torch", None)
        gsplat_version = getattr(identity, "gsplat", None)
        return (
            torch_version if isinstance(torch_version, str) else None,
            gsplat_version if isinstance(gsplat_version, str) else None,
        )

    def admitted_process_run(command: list[str], **options: object) -> Any:
        cache_root = config.model_root / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HOME": str(cache_root / "huggingface"),
                "HF_HUB_CACHE": str(cache_root / "huggingface" / "hub"),
                "HF_HUB_OFFLINE": "1",
                "TORCH_HOME": str(cache_root / "torch"),
                "XDG_CACHE_HOME": str(cache_root),
            }
        )
        options["env"] = environment
        with gpu_gate.hold(CancellationToken()):
            return cast(Any, subprocess.run)(command, **options)

    def runtime_which(command: str) -> str | None:
        local = (
            legacy_data_root
            / "cache"
            / "ffmpeg"
            / "bin"
            / executable_name(command)
        )
        if local.is_file() and not has_reparse_component(local):
            return str(local)
        return shutil.which(command)

    doctor = EnvironmentDoctor(
        which=runtime_which,
        cuda_probe=external_cuda_probe,
        segmentation_backend=config.segmentation_backend,
        worker_prefix=config.segmentation_worker_prefix,
        model_config=config.segmentation_model_config,
        checkpoint=config.segmentation_checkpoint,
        check_renderer=True,
        renderer_probe=external_renderer_probe,
        process_runner=admitted_process_run,
        vram_limit_mb=config.available_vram_limit_mb,
        vram_limit_provider=vram_budget.current_limit_mb,
    )
    runtime_path = config._runtime_path or (runtime_root / "desktop-runtime.json")
    preview_service = WorkerPreviewService(
        renderer,
        available_vram_limit_mb=config.available_vram_limit_mb,
        vram_limit_provider=vram_budget.current_limit_mb,
        live_session=preview_session,
    )
    repair = EnvironmentRepairManager(
        repo_root=runtime_root.parent,
        runtime_root=runtime_root,
        runtime_config=runtime_path,
        environment_doctor=doctor,
        acquire_runtime=preview_service.suspend_live,
        release_runtime=preview_service.resume_live,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=session_token,
        allowed_origins=(*_TAURI_ORIGINS, *allowed_browser_origins),
        task_workers=1,
    )
    services = ApiServices(
        project_repository=repository,
        environment_doctor=doctor,
        pipeline_runner=runner,
        worker_registry=WorkerRegistry(
            segmentation,
            renderer,
            preview_session,
            gpu_gate=gpu_gate,
            project_manager=project_manager,
        ),
        preview_service=preview_service,
        asset_inspector=asset_inspector,
        environment_repair=repair,
        vram_budget=vram_budget,
        project_catalog=catalog,
        asset_library=asset_library,
        project_manager=project_manager,
        preview_artifacts=ActivePreviewArtifactStore(project_manager),
        upload_root=cache_library_root / "upload-staging",
        artifact_store=artifact_store,
        storage_layout=storage_layout,
    )
    return settings, services


__all__ = [
    "WorkflowRuntimeConfig",
    "WorkerRegistry",
    "assemble_api_services",
    "load_runtime_config",
    "validate_browser_origins",
]
