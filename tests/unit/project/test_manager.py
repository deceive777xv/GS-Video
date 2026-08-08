from dataclasses import dataclass

import pytest

from gs_video.project.catalog import ProjectCatalog
from gs_video.project.manager import (
    ActivePipelineRunner,
    ActiveProjectManager,
    ActiveProjectRequiredError,
    ActiveProjectSession,
)
from gs_video.domain.models import StageName
from gs_video.project.repository import ProjectInstanceLock


@dataclass
class Closable:
    closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_active_manager_switches_only_after_candidate_is_ready(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    first = catalog.create("First")
    second = catalog.create("Second", activate=False)
    sessions: dict[str, ActiveProjectSession] = {}

    def build(project_id: str) -> ActiveProjectSession:
        repository = catalog.repository(project_id)
        session = ActiveProjectSession(
            repository=repository,
            project_lock=ProjectInstanceLock.acquire(repository.root),
            pipeline_runner=object(),
            preview_artifacts=object(),
        )
        sessions[project_id] = session
        return session

    manager = ActiveProjectManager(catalog, build)
    first_lock = sessions[first.project_id].project_lock

    activated = manager.activate(second.project_id)

    assert activated.project_id == second.project_id
    assert manager.active_project().project_id == second.project_id
    assert first_lock._closed is True
    manager.close()


def test_active_manager_preserves_current_session_when_candidate_fails(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    first = catalog.create("First")
    second = catalog.create("Second", activate=False)

    def build(project_id: str) -> ActiveProjectSession:
        if project_id == second.project_id:
            raise RuntimeError("candidate failed")
        repository = catalog.repository(project_id)
        return ActiveProjectSession(
            repository=repository,
            project_lock=ProjectInstanceLock.acquire(repository.root),
            pipeline_runner=object(),
            preview_artifacts=object(),
        )

    manager = ActiveProjectManager(catalog, build)

    with pytest.raises(RuntimeError, match="candidate failed"):
        manager.activate(second.project_id)

    assert manager.active_project().project_id == first.project_id
    assert catalog.active_project_id() == first.project_id
    manager.close()


def test_active_manager_can_delete_the_active_project(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    project = catalog.create("First")

    def build(project_id: str) -> ActiveProjectSession:
        repository = catalog.repository(project_id)
        return ActiveProjectSession(
            repository=repository,
            project_lock=ProjectInstanceLock.acquire(repository.root),
            pipeline_runner=object(),
            preview_artifacts=object(),
        )

    manager = ActiveProjectManager(catalog, build)
    manager.delete(project.project_id)

    assert manager.active_project() is None
    with pytest.raises(ActiveProjectRequiredError):
        manager.session()


def test_pipeline_capabilities_are_available_before_first_project(tmp_path) -> None:
    catalog = ProjectCatalog(tmp_path / "data")
    manager = ActiveProjectManager(catalog, lambda project_id: pytest.fail(project_id))
    runner = ActivePipelineRunner(manager)

    assert all(runner.supports(stage) for stage in StageName)
