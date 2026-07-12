import pytest

from gs_video.project.migrations import migrate_project_dict


def test_migration_adds_stage_map() -> None:
    migrated = migrate_project_dict({"schema_version": 0, "name": "legacy"})

    assert migrated["schema_version"] == 1
    assert migrated["stages"] == {}


def test_v0_migration_preserves_existing_stage_map() -> None:
    stages = {"ingest": {"status": "succeeded"}}

    migrated = migrate_project_dict(
        {"schema_version": 0, "name": "legacy", "stages": stages}
    )

    assert migrated["stages"] == stages


def test_migration_rejects_future_schema_version() -> None:
    with pytest.raises(ValueError, match="高于应用支持版本"):
        migrate_project_dict({"schema_version": 2, "name": "future"})
