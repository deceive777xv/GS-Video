from datetime import datetime, timezone
import os

import pytest

import gs_video.project.catalog as catalog_module
from gs_video.project.catalog import ProjectCatalog
from gs_video.project.repository import ProjectRepository


def test_catalog_creates_lists_renames_activates_and_deletes_projects(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")

    first = catalog.create(" First project ")
    second = catalog.create("Second", activate=False)

    assert first.name == "First project"
    assert catalog.active_project_id() == first.project_id
    assert {item.project_id for item in catalog.list()} == {
        first.project_id,
        second.project_id,
    }
    renamed = catalog.rename(second.project_id, "Renamed")
    assert renamed.name == "Renamed"
    catalog.activate(second.project_id)
    assert catalog.active_project_id() == second.project_id

    catalog.delete(second.project_id)

    assert catalog.active_project_id() is None
    assert [item.project_id for item in catalog.list()] == [first.project_id]
    assert not (catalog.projects_root / second.project_id).exists()


def test_catalog_rejects_case_insensitive_duplicate_names(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    catalog.create("Demo")

    with pytest.raises(ValueError, match="already in use"):
        catalog.create(" demo ")


def test_catalog_sorts_by_authoritative_updated_time(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    first = catalog.create("First", activate=False)
    second = catalog.create("Second", activate=False)
    old = datetime(2020, 1, 1, tzinfo=timezone.utc)
    new = datetime(2021, 1, 1, tzinfo=timezone.utc)
    first_repository = catalog.repository(first.project_id)
    second_repository = catalog.repository(second.project_id)
    first_project = first_repository.load()
    second_project = second_repository.load()
    first_project.updated_at = old
    second_project.updated_at = new
    first_repository.path.write_text(first_project.model_dump_json(), encoding="utf-8")
    second_repository.path.write_text(second_project.model_dump_json(), encoding="utf-8")

    assert [item.name for item in catalog.list()] == ["Second", "First"]


def test_delete_stays_committed_when_trash_cleanup_needs_startup_retry(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "data"
    catalog = ProjectCatalog(root)
    project = catalog.create("Disposable")
    original_rmtree = catalog_module.shutil.rmtree

    def fail_cleanup(path) -> None:  # type: ignore[no-untyped-def]
        raise PermissionError(path)

    monkeypatch.setattr(catalog_module.shutil, "rmtree", fail_cleanup)

    catalog.delete(project.project_id)

    assert catalog.list() == ()
    assert any(catalog.trash_root.iterdir())
    monkeypatch.setattr(catalog_module.shutil, "rmtree", original_rmtree)
    ProjectCatalog(root)
    assert list(catalog.trash_root.iterdir()) == []


def test_startup_restores_project_staged_before_catalog_delete_commit(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "data"
    catalog = ProjectCatalog(root)
    project = catalog.create("Must survive")
    source = catalog.projects_root / project.project_id
    staged = catalog.trash_root / f"{project.project_id}-{'a' * 32}"
    os.replace(source, staged)

    reopened = ProjectCatalog(root)

    assert reopened.active_project_id() == project.project_id
    assert [item.name for item in reopened.list()] == ["Must survive"]
    assert source.exists()
    assert not staged.exists()


def test_startup_recovers_valid_uuid_project_left_before_catalog_commit(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "data"
    catalog = ProjectCatalog(root)
    project_root = catalog.projects_root / "33333333-3333-4333-8333-333333333333"
    repository = ProjectRepository(project_root)
    project = repository.create("Recovered")
    project.project_id = project_root.name
    repository.save(project)

    reopened = ProjectCatalog(root)

    assert [item.name for item in reopened.list()] == ["Recovered"]
    assert reopened.active_project_id() == project_root.name
