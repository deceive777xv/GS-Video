import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from gs_video.api.schemas import ApiError
from gs_video.api.workflow import WorkerPreviewService
from gs_video.domain.contracts import SegmentationBackend
from gs_video.domain.models import SceneSummary, StageName, StageState, StageStatus
from gs_video.runtime import (
    WorkflowRuntimeConfig,
    assemble_api_services,
    load_runtime_config,
)
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.worker_client import RendererWorkerClient
from gs_video.project.repository import ProjectRepository


def runtime_payload(workspace: Path) -> dict[str, object]:
    model_root = workspace / "models"
    model_root.mkdir()
    (model_root / "segment.yaml").write_text("model: test", encoding="utf-8")
    (model_root / "segment.pt").write_bytes(b"checkpoint")
    worker = workspace / "python.exe"
    worker.write_bytes(b"worker")
    return {
        "project_root": str(workspace / "projects" / "demo"),
        "model_root": str(model_root),
        "segmentation_backend": "edgetam",
        "segmentation_worker_prefix": [str(worker)],
        "segmentation_model_config": str(model_root / "segment.yaml"),
        "segmentation_checkpoint": str(model_root / "segment.pt"),
        "renderer_worker_prefix": [str(worker)],
        "renderer_sh_degree": 3,
        "available_vram_limit_mb": 8192,
    }


def write_runtime(workspace: Path, payload: dict[str, object]) -> Path:
    path = workspace / "runtime.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_runtime_config_accepts_only_confined_absolute_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    config = load_runtime_config(write_runtime(workspace, runtime_payload(workspace)))

    assert config.segmentation_backend is SegmentationBackend.EDGETAM
    assert config.project_root == (workspace / "projects" / "demo").absolute()
    assert config.model_root == (workspace / "models").absolute()


def test_runtime_config_accepts_wsl_segmentation_worker_prefix(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-wsl"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload["segmentation_worker_prefix"] = [
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--",
        "/opt/edgetam/bin/python",
    ]

    config = load_runtime_config(write_runtime(workspace, payload))

    assert config.segmentation_worker_prefix == (
        "wsl.exe",
        "-d",
        "Ubuntu",
        "--",
        "/opt/edgetam/bin/python",
    )


def test_runtime_config_accepts_vram_budget_above_legacy_eight_gib(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-large-vram"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload["available_vram_limit_mb"] = 24_576

    config = load_runtime_config(write_runtime(workspace, payload))

    assert config.available_vram_limit_mb == 24_576


@pytest.mark.parametrize("field", ["token", "port", "origin", "allowed_origins"])
def test_runtime_config_rejects_launcher_or_secret_fields(
    tmp_path: Path, field: str
) -> None:
    workspace = tmp_path / field
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload[field] = "forbidden"

    with pytest.raises(ValidationError):
        load_runtime_config(write_runtime(workspace, payload))


def test_runtime_config_rejects_relative_config_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        load_runtime_config(Path("runtime.json"))


def test_runtime_config_rejects_project_containing_runtime_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload["project_root"] = str(workspace)

    with pytest.raises(ValueError, match="project_root"):
        load_runtime_config(write_runtime(workspace, payload))


def test_runtime_config_rejects_model_escape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    payload["segmentation_checkpoint"] = str(outside)

    with pytest.raises(ValueError, match="model_root"):
        load_runtime_config(write_runtime(workspace, payload))


def test_runtime_config_rejects_empty_worker_argv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-empty-worker"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload["renderer_worker_prefix"] = [""]
    with pytest.raises(ValidationError, match="worker"):
        load_runtime_config(write_runtime(workspace, payload))


def test_production_assembly_registers_every_stage(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = load_runtime_config(write_runtime(workspace, runtime_payload(workspace)))

    settings, services = assemble_api_services(config, SecretStr("secret"))

    assert isinstance(config, WorkflowRuntimeConfig)
    assert services.project_manager is not None
    services.project_manager.create("Assembly test")
    assert all(services.pipeline_runner.supports(stage) for stage in StageName)
    assert services.preview_service is not None
    assert callable(services.preview_service.render_live)
    assert settings.bind_host == "127.0.0.1"
    assert settings.port == 0
    asyncio.run(services.worker_registry.terminate_all())


def test_production_assembly_locks_then_reconciles_and_reopens_project(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-owned"
    workspace.mkdir()
    config = load_runtime_config(write_runtime(workspace, runtime_payload(workspace)))
    repository = ProjectRepository(config.project_root)
    project = repository.create("restart")
    project.workflow.active_task_id = "lost-task"
    project.stages[StageName.SEGMENT] = StageState(
        status=StageStatus.RUNNING, run_id="lost-run"
    )
    repository.save(project)

    _settings, first = assemble_api_services(config, SecretStr("secret"))
    recovered = first.project_repository.load()
    assert recovered.workflow.active_task_id is None
    assert recovered.stages[StageName.SEGMENT].status is StageStatus.FAILED
    assert recovered.stages[StageName.SEGMENT].error_code == "interrupted"
    with pytest.raises(RuntimeError, match="already open"):
        assemble_api_services(config, SecretStr("second"))

    asyncio.run(first.worker_registry.terminate_all())
    _settings, reopened = assemble_api_services(config, SecretStr("third"))
    claim = reopened.project_repository.claim_stage(
        StageName.SEGMENT, reuse_succeeded=False, run_id="retry"
    )
    assert claim.claimed
    asyncio.run(reopened.worker_registry.terminate_all())


def test_production_assembly_adopts_project_from_previous_runtime_layout(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-upgrade"
    workspace.mkdir()
    payload = runtime_payload(workspace)
    payload["project_root"] = str(workspace / "data" / "projects" / "default")
    config = load_runtime_config(write_runtime(workspace, payload))
    old_repository = ProjectRepository(workspace / "projects" / "default")
    old_repository.save(old_repository.create("Existing project"))

    _settings, services = assemble_api_services(config, SecretStr("secret"))

    assert services.project_manager is not None
    assert services.project_manager.active_project() is not None
    assert services.project_manager.active_project().name == "Existing project"
    asyncio.run(services.worker_registry.terminate_all())


def test_production_assembly_adds_explicit_loopback_browser_origin(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace-browser"
    workspace.mkdir()
    config = load_runtime_config(write_runtime(workspace, runtime_payload(workspace)))

    settings, _services = assemble_api_services(
        config,
        SecretStr("secret"),
        browser_origins=("http://127.0.0.1:4173",),
    )

    assert "http://127.0.0.1:4173" in settings.allowed_origins


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "https://example.com",
        "http://127.0.0.1:4173/path",
        "http://user@127.0.0.1:4173",
    ],
)
def test_production_assembly_rejects_unsafe_browser_origin(
    tmp_path: Path, origin: str
) -> None:
    workspace = tmp_path / f"workspace-browser-{abs(hash(origin))}"
    workspace.mkdir()
    config = load_runtime_config(write_runtime(workspace, runtime_payload(workspace)))

    with pytest.raises(ValueError, match="browser origin"):
        assemble_api_services(
            config,
            SecretStr("secret"),
            browser_origins=(origin,),
        )


def test_worker_preview_applies_conservative_configured_vram_limit(
    tmp_path: Path,
) -> None:
    worker = cast(RendererWorkerClient, cast(Any, object()))
    service = WorkerPreviewService(worker, available_vram_limit_mb=8192)
    summary = SceneSummary(
        filename="scene.ply",
        size=1,
        sha256="0" * 64,
        gaussian_count=1,
        estimated_vram_mb=6554,
    )

    with pytest.raises(ApiError) as error:
        service.render_pick(
            tmp_path,
            "source/scene.ply",
            summary,
            OrbitCamera(
                target=(0.0, 0.0, 0.0),
                distance=4.0,
                yaw=0.0,
                pitch=0.0,
                fov_y_degrees=60.0,
            ),
            960,
            540,
        )

    assert error.value.envelope.code == "scene_vram_limit_exceeded"


def test_worker_preview_reads_dynamic_vram_budget_for_each_admission(
    tmp_path: Path,
) -> None:
    selected = [10_000]
    worker = cast(RendererWorkerClient, cast(Any, object()))
    service = WorkerPreviewService(
        worker,
        available_vram_limit_mb=8192,
        vram_limit_provider=lambda: selected[0],
    )
    summary = SceneSummary(
        filename="scene.ply",
        size=1,
        sha256="0" * 64,
        gaussian_count=1,
        estimated_vram_mb=7_000,
    )

    service._admit(summary, 960, 540)
    selected[0] = 8_192
    with pytest.raises(ApiError) as error:
        service._admit(summary, 960, 540)

    assert error.value.envelope.code == "scene_vram_limit_exceeded"


def test_worker_preview_suspension_drains_inflight_render_and_blocks_admission(
    tmp_path: Path,
) -> None:
    class BlockingSession:
        def __init__(self) -> None:
            self.started = Event()
            self.release = Event()
            self.close_calls = 0

        def render_live(self, *_args: object) -> bytes:
            self.started.set()
            assert self.release.wait(1.0)
            return b"jpeg"

        def render_preview_pick(self, *_args: object) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            self.close_calls += 1

    session = BlockingSession()
    worker = cast(RendererWorkerClient, cast(Any, object()))
    service = WorkerPreviewService(
        worker,
        available_vram_limit_mb=8192,
        live_session=cast(Any, session),
    )
    summary = SceneSummary(
        filename="scene.ply",
        size=1,
        sha256="0" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    camera = OrbitCamera(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=0.0,
        pitch=0.0,
        fov_y_degrees=60.0,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        render = pool.submit(
            service.render_live,
            tmp_path,
            "source/scene.ply",
            summary,
            1,
            camera,
            16,
            9,
        )
        assert session.started.wait(1.0)
        suspend = pool.submit(service.suspend_live)
        assert suspend.done() is False
        session.release.set()
        assert render.result(timeout=1.0) == b"jpeg"
        token = suspend.result(timeout=1.0)

    with pytest.raises(ApiError) as error:
        service.render_live(
            tmp_path,
            "source/scene.ply",
            summary,
            2,
            camera,
            16,
            9,
        )

    assert error.value.envelope.code == "preview_suspended"
    assert session.close_calls == 1
    service.resume_live(token)
    assert service.render_live(
        tmp_path,
        "source/scene.ply",
        summary,
        3,
        camera,
        16,
        9,
    ) == b"jpeg"
