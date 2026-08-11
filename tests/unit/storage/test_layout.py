import json
import os
from pathlib import Path

import pytest

import gs_video.storage.layout as storage_layout_module
from gs_video.storage.layout import (
    CacheAction,
    CacheCleanupMode,
    ProjectLibraryAction,
    StorageLayoutManager,
)
from gs_video.domain.models import (
    ArtifactCategory,
    ArtifactRef,
    ArtifactRole,
    Project,
    StageName,
    StageState,
    StageStatus,
)
from gs_video.project.repository import ProjectRepository


def manager(container: Path, settings: Path) -> StorageLayoutManager:
    return StorageLayoutManager(
        container,
        settings,
        drive_type_probe=lambda _path: 3,
    )


def test_default_layout_adopts_legacy_catalog_projects_and_assets(
    tmp_path: Path,
) -> None:
    container = tmp_path / "data"
    (container / "projects" / "one").mkdir(parents=True)
    (container / "projects" / "one" / "project.json").write_text("{}")
    (container / "assets").mkdir()
    (container / "catalog.json").write_text("{}")

    layout = manager(container, tmp_path / "runtime" / "user-settings.json")

    assert (layout.project_library_root / "catalog.json").is_file()
    assert (layout.project_library_root / "projects" / "one").is_dir()
    assert (layout.project_library_root / "assets").is_dir()
    assert layout.cache_root == container / "cache-library"
    assert not (container / "projects").exists()


def test_default_layout_adoption_resumes_after_top_level_move_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = tmp_path / "data"
    (container / "projects").mkdir(parents=True)
    (container / "catalog.json").write_text("{}")
    settings = tmp_path / "runtime" / "user-settings.json"
    real_replace = os.replace
    interrupted = False

    def interrupt_projects(source: Path | str, destination: Path | str) -> None:
        nonlocal interrupted
        if Path(source) == container / "projects" and not interrupted:
            interrupted = True
            raise OSError("simulated interruption")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupt_projects)
    with pytest.raises(OSError, match="simulated interruption"):
        manager(container, settings)

    monkeypatch.setattr(os, "replace", real_replace)
    layout = manager(container, settings)

    assert (layout.project_library_root / "catalog.json").is_file()
    assert (layout.project_library_root / "projects").is_dir()
    assert not (container / "catalog.json").exists()
    assert not (container / "projects").exists()


def test_layout_preserves_legacy_vram_preference_when_initializing(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "runtime" / "user-settings.json"
    settings.parent.mkdir()
    settings.write_text(
        json.dumps({"mode": "custom", "selected_vram_mb": 12_288})
    )

    manager(tmp_path / "data", settings)

    saved = json.loads(settings.read_text())
    assert saved["vram_budget"] == {
        "mode": "custom",
        "selected_vram_mb": 12_288,
    }
    assert "storage_layout" in saved


def test_layout_rejects_non_fixed_drive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed drive"):
        StorageLayoutManager(
            tmp_path / "data",
            tmp_path / "settings.json",
            drive_type_probe=lambda _path: 2,
        )


def test_storage_status_does_not_measure_storage_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")

    def reject_measurement(_root: Path) -> int:
        raise AssertionError("status must not recursively measure storage")

    monkeypatch.setattr(storage_layout_module, "_tree_size", reject_measurement)

    status = layout.status(editable=False, blocked_reason="runtime_busy")

    assert status.project_library_root == str(layout.project_library_root)
    assert status.cache_root == str(layout.cache_root)
    assert status.editable is False
    assert status.blocked_reason == "runtime_busy"


def test_layout_switch_rejects_overlapping_roots(tmp_path: Path) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    project = tmp_path / "new"

    with pytest.raises(ValueError, match="must not overlap"):
        layout.switch(
            project,
            project / "cache",
            project_action=ProjectLibraryAction.MIGRATE,
            cache_action=CacheAction.START_FRESH,
        )


def test_layout_switch_copies_project_library_and_starts_fresh_cache(
    tmp_path: Path,
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    (layout.project_library_root / "catalog.json").write_text('{"projects":[]}')
    target_project = tmp_path / "target-projects"
    target_cache = tmp_path / "target-cache"

    snapshot = layout.switch(
        target_project,
        target_cache,
        project_action=ProjectLibraryAction.MIGRATE,
        cache_action=CacheAction.START_FRESH,
    )

    assert (target_project / "catalog.json").read_text() == '{"projects":[]}'
    assert snapshot.restart_required is True
    assert snapshot.editable is False
    assert snapshot.blocked_reason == "restart_required"


def test_layout_switch_resumes_after_activation_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    (layout.project_library_root / "catalog.json").write_text('{"projects":[]}')
    target_project = tmp_path / "target-projects"
    target_cache = tmp_path / "target-cache"
    real_initialize = layout._initialize_root
    interrupted = False

    def interrupt_once(root: Path, kind):  # type: ignore[no-untyped-def]
        nonlocal interrupted
        if root == target_project and not interrupted:
            interrupted = True
            raise OSError("simulated interruption")
        return real_initialize(root, kind)

    monkeypatch.setattr(layout, "_initialize_root", interrupt_once)
    with pytest.raises(OSError, match="simulated interruption"):
        layout.switch(
            target_project,
            target_cache,
            project_action=ProjectLibraryAction.MIGRATE,
            cache_action=CacheAction.START_FRESH,
        )

    monkeypatch.setattr(layout, "_initialize_root", real_initialize)
    snapshot = layout.switch(
        target_project,
        target_cache,
        project_action=ProjectLibraryAction.MIGRATE,
        cache_action=CacheAction.START_FRESH,
    )

    assert snapshot.restart_required is True
    assert (target_project / "catalog.json").is_file()


def test_layout_switch_resumes_after_preference_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    (layout.project_library_root / "catalog.json").write_text('{"projects":[]}')
    target_project = tmp_path / "target-projects"
    target_cache = tmp_path / "target-cache"
    real_save = layout._save_preference

    def fail_save(_preference) -> None:  # type: ignore[no-untyped-def]
        raise OSError("settings unavailable")

    monkeypatch.setattr(layout, "_save_preference", fail_save)
    with pytest.raises(OSError, match="settings unavailable"):
        layout.switch(
            target_project,
            target_cache,
            project_action=ProjectLibraryAction.MIGRATE,
            cache_action=CacheAction.START_FRESH,
        )

    monkeypatch.setattr(layout, "_save_preference", real_save)
    snapshot = layout.switch(
        target_project,
        target_cache,
        project_action=ProjectLibraryAction.MIGRATE,
        cache_action=CacheAction.START_FRESH,
    )

    assert snapshot.restart_required is True


def test_layout_switch_opens_marker_owned_existing_project_library(
    tmp_path: Path,
) -> None:
    first = manager(tmp_path / "first-data", tmp_path / "first-settings.json")
    existing = first.project_library_root
    (existing / "catalog.json").write_text(
        '{"schema_version":1,"project_ids":[],"active_project_id":null}'
    )
    (existing / "assets").mkdir()
    (existing / "assets" / "index.json").write_text(
        '{"schema_version":1,"assets":{}}'
    )
    second = manager(tmp_path / "second-data", tmp_path / "second-settings.json")

    snapshot = second.switch(
        existing,
        second.cache_root,
        project_action=ProjectLibraryAction.OPEN_EXISTING,
        cache_action=CacheAction.START_FRESH,
    )

    assert snapshot.project_library_root == str(existing)


def test_layout_switch_rejects_corrupt_existing_project_library(
    tmp_path: Path,
) -> None:
    first = manager(tmp_path / "first-data", tmp_path / "first-settings.json")
    existing = first.project_library_root
    (existing / "catalog.json").write_text('{"projects":[]}', encoding="utf-8")
    second = manager(tmp_path / "second-data", tmp_path / "second-settings.json")

    with pytest.raises(ValueError, match="catalog"):
        second.switch(
            existing,
            second.cache_root,
            project_action=ProjectLibraryAction.OPEN_EXISTING,
            cache_action=CacheAction.START_FRESH,
        )


def test_safe_cache_cleanup_removes_only_unreferenced_cache_entries(
    tmp_path: Path,
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    project = Project(name="cleanup")
    referenced_key = "a" * 64
    orphan_key = "b" * 64
    reference = ArtifactRef(
        project_id=project.project_id,
        category=ArtifactCategory.FRAMES,
        cache_key=referenced_key,
    )
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=referenced_key,
        output_paths=[reference],
        artifacts={ArtifactRole.SOURCE_FRAMES: reference},
    )
    repository = ProjectRepository(
        layout.project_library_root / "projects" / project.project_id
    )
    repository.create(project.name)
    repository.save(project)
    cache_project = layout.cache_root / "projects" / project.project_id / "frames"
    (cache_project / referenced_key).mkdir(parents=True)
    (cache_project / referenced_key / "frame.png").write_bytes(b"keep")
    (cache_project / orphan_key).mkdir()
    (cache_project / orphan_key / "frame.png").write_bytes(b"remove")
    staging = cache_project / f".staging-{'d' * 32}"
    staging.mkdir()
    (staging / "partial.png").write_bytes(b"partial")

    plan = layout.plan_cache_cleanup(CacheCleanupMode.SAFE)
    result = layout.cleanup_cache(CacheCleanupMode.SAFE, plan.plan_token)

    assert result.removed_entries == 2
    assert (cache_project / referenced_key / "frame.png").is_file()
    assert not (cache_project / orphan_key).exists()
    assert not staging.exists()
    assert repository.load().stages[StageName.INGEST].status is StageStatus.SUCCEEDED


def test_deep_cache_cleanup_invalidates_rebuildable_stage_authority(
    tmp_path: Path,
) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    project = Project(name="deep")
    cache_key = "c" * 64
    reference = ArtifactRef(
        project_id=project.project_id,
        category=ArtifactCategory.PROXIES,
        cache_key=cache_key,
    )
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=cache_key,
        output_paths=[reference],
        artifacts={ArtifactRole.PROXY_FRAMES: reference},
    )
    repository = ProjectRepository(
        layout.project_library_root / "projects" / project.project_id
    )
    repository.create(project.name)
    repository.save(project)
    cache_entry = layout.cache_root / "projects" / project.project_id / "proxies" / cache_key
    cache_entry.mkdir(parents=True)
    (cache_entry / "frame.jpg").write_bytes(b"cache")

    plan = layout.plan_cache_cleanup(CacheCleanupMode.DEEP)
    result = layout.cleanup_cache(CacheCleanupMode.DEEP, plan.plan_token)

    state = repository.load().stages[StageName.INGEST]
    assert result.removed_entries == 1
    assert state.status is StageStatus.STALE
    assert state.cache_key is None
    assert state.output_paths == []
    assert state.artifacts == {}


def test_cache_cleanup_rejects_plan_after_cache_changes(tmp_path: Path) -> None:
    layout = manager(tmp_path / "data", tmp_path / "settings.json")
    plan = layout.plan_cache_cleanup(CacheCleanupMode.SAFE)
    changed = layout.cache_root / "changed.bin"
    changed.write_bytes(b"changed")

    with pytest.raises(ValueError, match="stale or invalid"):
        layout.cleanup_cache(CacheCleanupMode.SAFE, plan.plan_token)
