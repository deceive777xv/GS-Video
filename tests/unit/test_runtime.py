import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import SecretStr, ValidationError

from gs_video.api.schemas import ApiError
from gs_video.api.workflow import WorkerPreviewService
from gs_video.domain.contracts import SegmentationBackend
from gs_video.domain.models import SceneSummary, StageName
from gs_video.runtime import (
    WorkflowRuntimeConfig,
    assemble_api_services,
    load_runtime_config,
)
from gs_video.scene.camera import OrbitCamera
from gs_video.scene.worker_client import RendererWorkerClient


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


def test_runtime_config_rejects_model_escape_and_empty_worker_argv(
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
    assert all(services.pipeline_runner.supports(stage) for stage in StageName)
    assert services.preview_service is not None
    assert settings.bind_host == "127.0.0.1"
    assert settings.port == 0


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
