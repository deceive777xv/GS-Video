import pytest

from gs_video.project.migrations import migrate_project_dict


def test_migration_adds_current_stage_and_ground_alignment_state() -> None:
    migrated = migrate_project_dict({"schema_version": 0, "name": "legacy"})

    assert migrated["schema_version"] == 9
    assert migrated["stages"]["post_process"]["status"] == "pending"
    assert migrated["workflow"] == {
        "target_ground_generation": 0,
        "target_ground": None,
        "gs_scale": 1.0,
        "scene_azimuth": 0.0,
        "output_crop": None,
        "source_color_interpretation": None,
        "matte_refinement": {
            "enabled": True,
            "edge_offset": -1.0,
            "feather_radius": 1.0,
            "decontaminate_strength": 0.0,
            "decontaminate_radius": 3.0,
        },
        "effect_chain": [],
        "effect_chain_revision": 0,
        "export_settings": {
            "codec": "h264",
            "rate_control": "constant_quality",
            "quality": 70,
            "target_bitrate_mbps": 12.0,
            "compression_preset": "balanced",
        },
        "export_result": None,
    }


def test_v0_migration_preserves_existing_stage_map() -> None:
    stages = {"ingest": {"status": "succeeded"}}

    migrated = migrate_project_dict(
        {"schema_version": 0, "name": "legacy", "stages": stages}
    )

    assert migrated["stages"]["ingest"] == {
        "status": "succeeded",
        "output_paths": [],
        "artifacts": {},
    }
    assert migrated["stages"]["post_process"]["status"] == "pending"


def test_migration_rejects_future_schema_version() -> None:
    with pytest.raises(ValueError, match="高于应用支持版本"):
        migrate_project_dict({"schema_version": 10, "name": "future"})


def test_v8_migration_revokes_legacy_composite_and_export_authority() -> None:
    migrated = migrate_project_dict(
        {
            "schema_version": 8,
            "name": "legacy-composite",
            "workflow": {"export_result": {"verified": True}},
            "stages": {
                "render": {"status": "succeeded"},
                "composite": {"status": "succeeded", "input_generation": 2},
                "export": {"status": "succeeded", "input_generation": 4},
            },
        }
    )

    assert migrated["stages"]["render"]["status"] == "succeeded"
    assert migrated["stages"]["composite"]["status"] == "stale"
    assert migrated["stages"]["composite"]["input_generation"] == 3
    assert migrated["stages"]["post_process"]["status"] == "pending"
    assert migrated["stages"]["export"]["status"] == "stale"
    assert migrated["stages"]["export"]["input_generation"] == 5
    assert migrated["workflow"]["export_result"] is None


def test_v6_migration_deletes_legacy_person_placement_authority() -> None:
    migrated = migrate_project_dict(
        {
            "schema_version": 6,
            "name": "legacy-placement",
            "workflow": {
                "confirmed_camera_revision": 4,
                "confirmed_preview_artifact_id": "a" * 32,
                "foot_point": {"image": [8, 4]},
                "subject_visibility_audit": {"revision": 1},
                "source_perspective_calibration": {"revision": 1},
                "local_ground_anchor": {"revision": 1},
                "subject_contact_constraint": {"revision": 1},
                "synthesis_placement": {"revision": 1},
                "confirmed_synthesis_placement_revision": 1,
                "motion_scale": 0.5,
            },
        }
    )

    workflow = migrated["workflow"]
    for key in (
        "confirmed_camera_revision",
        "confirmed_preview_artifact_id",
        "foot_point",
        "subject_visibility_audit",
        "source_perspective_calibration",
        "local_ground_anchor",
        "subject_contact_constraint",
        "synthesis_placement",
        "confirmed_synthesis_placement_revision",
        "motion_scale",
    ):
        assert key not in workflow
    assert workflow["target_ground"] is None
    assert workflow["gs_scale"] == 1.0
    assert workflow["scene_azimuth"] == 0.0
