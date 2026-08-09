import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    CameraPose,
    Project,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
)
from gs_video.project.repository import ProjectRepository


PROJECT_DIRECTORIES: set[str] = set()


def test_create_makes_exact_project_directory_layout(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "demo"

    project = ProjectRepository(root).create("demo")

    assert project.name == "demo"
    assert {path.name for path in root.iterdir()} == PROJECT_DIRECTORIES
    assert all((root / name).is_dir() for name in PROJECT_DIRECTORIES)


def test_repository_round_trips_project(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    project = repo.create("demo")
    reference = ArtifactRef(
        project_id=project.project_id,
        category=ArtifactCategory.FRAMES,
        cache_key="a" * 64,
        member="input.mp4",
    )
    project.stages[StageName.INGEST] = StageState(output_paths=[reference])

    repo.save(project)
    loaded = repo.load()

    assert loaded == project
    assert loaded.project_id == project.project_id
    assert loaded.schema_version == 5
    assert (tmp_path / "project.json").exists()


def test_repository_load_migrates_v2_pick_authority_and_round_trips_v4(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path)
    artifact_id = "a" * 32
    raw = Project(name="legacy").model_dump(mode="json")
    raw["schema_version"] = 2
    raw["workflow"]["confirmed_camera_revision"] = 2
    raw["workflow"].pop("confirmed_preview_artifact_id", None)
    raw["workflow"]["preview"] = {
        "artifact_id": artifact_id,
        "artifact_size": 12,
        "artifact_sha256": "b" * 64,
        "generation": 1,
        "width": 16,
        "height": 9,
        "camera_revision": 2,
        "pick_buffer_revision": 3,
    }
    raw["workflow"]["foot_point"] = {
        "image": [8, 4],
        "world": [0.0, 0.0, 2.0],
        "camera_revision": 2,
        "pick_buffer_revision": 3,
    }
    repository.path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = repository.load()
    repository.save(loaded)
    round_tripped = repository.load()

    assert round_tripped.schema_version == 5
    assert round_tripped.workflow.confirmed_preview_artifact_id == artifact_id
    assert round_tripped.workflow.foot_point is not None
    assert round_tripped.workflow.foot_point.preview_artifact_id == artifact_id


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "demo", "unexpected": True},
        {
            "name": "demo",
            "stages": {"ingest": {"status": "pending", "unexpected": True}},
        },
    ],
)
def test_project_models_reject_extra_fields(raw: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Project.model_validate(raw)


def test_save_replaces_project_with_complete_valid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = ProjectRepository(tmp_path)
    original = repo.create("original")
    repo.save(original)
    previous_document = repo.path.read_text(encoding="utf-8")
    replacement = original.model_copy(update={"name": "replacement"})
    real_replace = os.replace
    observations: list[tuple[Path, Path]] = []

    def observe_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent == tmp_path
        assert json.loads(source_path.read_text(encoding="utf-8"))["name"] == "replacement"
        assert destination_path.read_text(encoding="utf-8") == previous_document
        observations.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr(os, "replace", observe_replace)

    repo.save(replacement)

    assert len(observations) == 1
    temporary, destination = observations[0]
    assert temporary.parent == tmp_path
    assert temporary.name.startswith(".project.json.")
    assert temporary.name.endswith(".tmp")
    assert destination == tmp_path / "project.json"
    assert json.loads(repo.path.read_text(encoding="utf-8"))["name"] == "replacement"
    assert not temporary.exists()


def test_repository_round_trips_workflow_authority(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    project = repo.create("workflow")
    project.workflow.subject_prompt = SubjectPromptState(
        frame_index=12, x=100, y=120
    )
    project.workflow.target_camera = CameraPose(
        target=(0.0, 0.0, 0.0),
        distance=4.0,
        yaw=20.0,
        pitch=-5.0,
        fov_y_degrees=48.0,
        revision=3,
    )

    repo.save(project)

    loaded = repo.load()
    assert loaded.workflow.subject_prompt == project.workflow.subject_prompt
    assert loaded.workflow.target_camera == project.workflow.target_camera


def test_repository_rejects_cross_project_artifact_authority(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "project")
    project = repository.create("authority")
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key="a" * 64,
        output_paths=[
            ArtifactRef(
                project_id="00000000-0000-4000-8000-000000000000",
                category=ArtifactCategory.FRAMES,
                cache_key="a" * 64,
            )
        ],
    )

    with pytest.raises(ValueError, match="another project"):
        repository.save(project)


def test_reconcile_interrupted_runs_clears_owner_and_allows_retry(tmp_path: Path) -> None:
    repository = ProjectRepository(tmp_path / "project")
    project = repository.create("interrupted")
    project.workflow.active_task_id = "lost-task"
    project.stages[StageName.RENDER] = StageState(
        status=StageStatus.RUNNING,
        run_id="lost-run",
        cache_key="must-not-survive",
    )
    repository.save(project)

    recovered = repository.reconcile_interrupted_runs()
    state = recovered.stages[StageName.RENDER]

    assert recovered.workflow.active_task_id is None
    assert state.status is StageStatus.FAILED
    assert state.error_code == "interrupted"
    assert state.cache_key is None
    assert state.run_id is None
    assert repository.claim_stage(
        StageName.RENDER, reuse_succeeded=False, run_id="retry"
    ).claimed
