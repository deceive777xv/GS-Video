import json
from pathlib import Path

from pydantic import SecretStr

from gs_video.app import create_app
from gs_video.domain.models import StageName
from gs_video.runtime import assemble_api_services, load_runtime_config


def test_production_assembly_builds_runnable_api_without_spawning_workers(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    models.mkdir(parents=True)
    config_file = models / "segment.yaml"
    checkpoint = models / "segment.pt"
    executable = workspace / "worker.exe"
    config_file.write_text("model: test", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    executable.write_bytes(b"worker")
    runtime_path = workspace / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "project_root": str(workspace / "projects" / "demo"),
                "model_root": str(models),
                "segmentation_backend": "edgetam",
                "segmentation_worker_prefix": [str(executable)],
                "segmentation_model_config": str(config_file),
                "segmentation_checkpoint": str(checkpoint),
                "renderer_worker_prefix": [str(executable)],
            }
        ),
        encoding="utf-8",
    )

    settings, services = assemble_api_services(
        load_runtime_config(runtime_path.absolute()), SecretStr("session-secret")
    )
    app = create_app(settings, services)

    assert app.state.preview_service is services.preview_service
    assert all(services.pipeline_runner.supports(stage) for stage in StageName)
    assert services.project_repository.load().name == "GS Video project"
