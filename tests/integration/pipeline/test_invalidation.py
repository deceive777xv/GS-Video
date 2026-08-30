import pytest
import hashlib

from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    Project,
    StageName,
    StageState,
    StageStatus,
)
import gs_video.pipeline.workflow as workflow


def completed_project() -> Project:
    project = Project(name="completed")
    project.stages = {
            name: StageState(
                status=StageStatus.SUCCEEDED,
                cache_key=hashlib.sha256(name.value.encode()).hexdigest(),
                output_paths=[
                    ArtifactRef(
                        project_id=project.project_id,
                        category=ArtifactCategory.RENDERS,
                        cache_key=hashlib.sha256(name.value.encode()).hexdigest(),
                        member=f"{name.value}.bin",
                    )
                ],
            )
            for name in StageName
        }
    return project


@pytest.mark.parametrize(
    ("change", "expected_stale"),
    [
        ("SOURCE_VIDEO", set(StageName)),
        (
            "SUBJECT_PROMPT",
            {
                StageName.SEGMENT,
                StageName.SOLVE_CAMERA,
                StageName.MAP_TRAJECTORY,
                    StageName.RENDER,
                    StageName.COMPOSITE,
                    StageName.POST_PROCESS,
                    StageName.EXPORT,
            },
        ),
        (
            "TARGET_CAMERA",
            {
                StageName.MAP_TRAJECTORY,
                    StageName.RENDER,
                    StageName.COMPOSITE,
                    StageName.POST_PROCESS,
                    StageName.EXPORT,
            },
        ),
        (
            "GS_ALIGNMENT",
            {
                StageName.MAP_TRAJECTORY,
                    StageName.RENDER,
                    StageName.COMPOSITE,
                    StageName.POST_PROCESS,
                    StageName.EXPORT,
            },
        ),
        (
            "OUTPUT_CROP",
            {
                    StageName.RENDER,
                    StageName.COMPOSITE,
                    StageName.POST_PROCESS,
                    StageName.EXPORT,
            },
        ),
            (
                "EDGE_SETTINGS",
                {StageName.COMPOSITE, StageName.POST_PROCESS, StageName.EXPORT},
            ),
            (
                "POST_PROCESS_SETTINGS",
                {StageName.POST_PROCESS, StageName.EXPORT},
            ),
        ("EXPORT_SETTINGS", {StageName.EXPORT}),
    ],
)
def test_change_invalidates_only_its_root_and_downstream_stages(
    change: str, expected_stale: set[StageName]
) -> None:
    project = completed_project()

    result = workflow.invalidate_for_change(project, workflow.ChangeKind[change])

    assert result is project
    for name, state in project.stages.items():
        if name in expected_stale:
            assert state.status is StageStatus.STALE
            assert state.cache_key is None
        else:
            assert state.status is StageStatus.SUCCEEDED
            assert state.cache_key == hashlib.sha256(name.value.encode()).hexdigest()
        assert state.output_paths[0].member == f"{name.value}.bin"


def test_changing_target_camera_keeps_ingest_segment_and_solver_cached() -> None:
    project = completed_project()

    workflow.invalidate_for_change(project, workflow.ChangeKind.TARGET_CAMERA)

    assert project.stages[StageName.INGEST].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.SEGMENT].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.SOLVE_CAMERA].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.MAP_TRAJECTORY].status is StageStatus.STALE
