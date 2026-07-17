import pytest

from gs_video.project.migrations import migrate_project_dict


def test_migration_adds_stage_map() -> None:
    migrated = migrate_project_dict({"schema_version": 0, "name": "legacy"})

    assert migrated["schema_version"] == 3
    assert migrated["stages"] == {}
    assert migrated["workflow"] == {}


def test_v0_migration_preserves_existing_stage_map() -> None:
    stages = {"ingest": {"status": "succeeded"}}

    migrated = migrate_project_dict(
        {"schema_version": 0, "name": "legacy", "stages": stages}
    )

    assert migrated["stages"] == stages


def test_v1_migration_adds_persisted_workflow_state() -> None:
    migrated = migrate_project_dict(
        {
            "schema_version": 1,
            "name": "legacy",
            "source_video": None,
            "scene_ply": None,
            "stages": {},
        }
    )

    assert migrated["schema_version"] == 3
    assert migrated["workflow"] == {}


def test_migration_rejects_future_schema_version() -> None:
    with pytest.raises(ValueError, match="高于应用支持版本"):
        migrate_project_dict({"schema_version": 4, "name": "future"})


def test_v2_migration_binds_matching_legacy_pick_authority() -> None:
    artifact_id = "a" * 32
    migrated = migrate_project_dict(
        {
            "schema_version": 2,
            "name": "legacy-pick",
            "workflow": {
                "confirmed_camera_revision": 4,
                "preview": {
                    "artifact_id": artifact_id,
                    "camera_revision": 4,
                    "pick_buffer_revision": 7,
                },
                "foot_point": {
                    "image": [8, 4],
                    "world": [0.0, 0.0, 2.0],
                    "camera_revision": 4,
                    "pick_buffer_revision": 7,
                },
            },
        }
    )

    assert migrated["schema_version"] == 3
    workflow = migrated["workflow"]
    assert workflow["confirmed_preview_artifact_id"] == artifact_id
    assert workflow["foot_point"]["preview_artifact_id"] == artifact_id


@pytest.mark.parametrize(
    "workflow",
    [
        {
            "confirmed_camera_revision": 3,
            "preview": {
                "artifact_id": "a" * 32,
                "camera_revision": 4,
                "pick_buffer_revision": 7,
            },
            "foot_point": {
                "image": [8, 4],
                "world": [0.0, 0.0, 2.0],
                "camera_revision": 4,
                "pick_buffer_revision": 7,
            },
        },
        {
            "confirmed_camera_revision": 4,
            "preview": {
                "artifact_id": "a" * 32,
                "camera_revision": 4,
                "pick_buffer_revision": 7,
            },
            "foot_point": {
                "image": [8, 4],
                "world": [0.0, 0.0, 2.0],
                "camera_revision": 4,
                "pick_buffer_revision": 6,
            },
        },
    ],
)
def test_v2_migration_clears_unbound_legacy_pick_authority(
    workflow: dict[str, object],
) -> None:
    migrated = migrate_project_dict(
        {"schema_version": 2, "name": "legacy-pick", "workflow": workflow}
    )

    migrated_workflow = migrated["workflow"]
    assert migrated_workflow["foot_point"] is None
    if workflow["confirmed_camera_revision"] != 4:
        assert migrated_workflow["confirmed_camera_revision"] is None
        assert migrated_workflow["confirmed_preview_artifact_id"] is None
