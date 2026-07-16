import pytest

from gs_video.project.migrations import migrate_project_dict


def test_migration_adds_stage_map() -> None:
    migrated = migrate_project_dict({"schema_version": 0, "name": "legacy"})

    assert migrated["schema_version"] == 2
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

    assert migrated["schema_version"] == 2
    assert migrated["workflow"] == {}


def test_migration_rejects_future_schema_version() -> None:
    with pytest.raises(ValueError, match="高于应用支持版本"):
        migrate_project_dict({"schema_version": 3, "name": "future"})
