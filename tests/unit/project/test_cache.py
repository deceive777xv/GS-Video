from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    Project,
    StageName,
    StageState,
    StageStatus,
)
from gs_video.pipeline.workflow import invalidate_from
from gs_video.project.cache import cache_key


def test_cache_key_is_canonical_for_nested_mappings_and_sequences() -> None:
    left = cache_key(
        "render",
        {"frames": [{"path": "一.png", "index": 1}], "metadata": {"b": 2, "a": 1}},
        {"quality": {"scale": 1.0, "samples": [4, 8]}},
        "1",
    )
    right = cache_key(
        "render",
        {"metadata": {"a": 1, "b": 2}, "frames": [{"index": 1, "path": "一.png"}]},
        {"quality": {"samples": [4, 8], "scale": 1.0}},
        "1",
    )

    assert left == right


def test_cache_key_preserves_sequence_order() -> None:
    left = cache_key("render", {"frames": ["a", "b"]}, {}, "1")
    right = cache_key("render", {"frames": ["b", "a"]}, {}, "1")

    assert left != right


def test_invalidate_from_marks_present_transitive_dependents_stale() -> None:
    project = Project(name="demo")

    def state(
        name: StageName,
        status: StageStatus,
        member: str,
        error_code: str | None = None,
    ) -> StageState:
        key = cache_key(name.value, {}, {}, "test")
        return StageState(
            status=status,
            cache_key=key,
            output_paths=[
                ArtifactRef(
                    project_id=project.project_id,
                    category=ArtifactCategory.RENDERS,
                    cache_key=key,
                    member=member,
                )
            ],
            error_code=error_code,
        )

    project.stages = {
            StageName.INGEST: StageState(
                **state(
                    StageName.INGEST,
                    StageStatus.SUCCEEDED,
                    "meta.json",
                    "old-ingest-error",
                ).model_dump()
            ),
            StageName.SOLVE_CAMERA: state(StageName.SOLVE_CAMERA, StageStatus.SUCCEEDED, "source.json"),
            StageName.MAP_TRAJECTORY: state(StageName.MAP_TRAJECTORY, StageStatus.FAILED, "target.json", "repairable"),
            StageName.RENDER: state(StageName.RENDER, StageStatus.SUCCEEDED, "frame.png"),
            StageName.EXPORT: state(StageName.EXPORT, StageStatus.SUCCEEDED, "final.mp4"),
        }

    returned = invalidate_from(project, StageName.SOLVE_CAMERA)

    assert returned is project
    for name in (StageName.SOLVE_CAMERA, StageName.MAP_TRAJECTORY, StageName.RENDER):
        state = project.stages[name]
        assert state.status is StageStatus.STALE
        assert state.cache_key is None
        assert state.error_code is None
    assert project.stages[StageName.SOLVE_CAMERA].output_paths[0].member == "source.json"
    assert project.stages[StageName.MAP_TRAJECTORY].output_paths[0].member == "target.json"
    assert project.stages[StageName.RENDER].output_paths[0].member == "frame.png"
    assert project.stages[StageName.INGEST].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.EXPORT].status is StageStatus.STALE
    assert project.stages[StageName.EXPORT].cache_key is None
    assert project.stages[StageName.EXPORT].output_paths[0].member == "final.mp4"
    assert StageName.COMPOSITE not in project.stages
