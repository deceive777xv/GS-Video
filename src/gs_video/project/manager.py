from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, cast

from gs_video.domain.models import Project, StageName, StageState
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.catalog import ProjectCatalog, ProjectSummary, project_summary
from gs_video.project.repository import ProjectInstanceLock, ProjectRepository


class ActiveProjectRequiredError(RuntimeError):
    pass


@dataclass
class ActiveProjectSession:
    repository: ProjectRepository
    project_lock: ProjectInstanceLock
    pipeline_runner: Any
    preview_artifacts: Any

    def close(self) -> None:
        self.project_lock.close()


SessionFactory = Callable[[str], ActiveProjectSession]


class ActiveProjectManager:
    def __init__(
        self,
        catalog: ProjectCatalog,
        session_factory: SessionFactory,
    ) -> None:
        self.catalog = catalog
        self._session_factory = session_factory
        self._lock = RLock()
        self._session: ActiveProjectSession | None = None
        active_id = catalog.active_project_id()
        if active_id is not None:
            self._session = session_factory(active_id)

    def session(self) -> ActiveProjectSession:
        with self._lock:
            if self._session is None:
                raise ActiveProjectRequiredError("an active project is required")
            return self._session

    def active_project(self) -> Project | None:
        with self._lock:
            return None if self._session is None else self._session.repository.load()

    def list(self) -> tuple[ProjectSummary, ...]:
        return self.catalog.list()

    def activate(
        self,
        project_id: str,
        before_switch: Callable[[], None] | None = None,
    ) -> Project:
        with self._lock:
            current = self._session
            if current is not None:
                project = current.repository.load()
                if project.project_id == project_id:
                    return project
            candidate = self._session_factory(project_id)
            try:
                if before_switch is not None:
                    before_switch()
                project = self.catalog.activate(project_id)
            except BaseException:
                candidate.close()
                raise
            self._session = candidate
            if current is not None:
                current.close()
            return project

    def create(
        self,
        name: str,
        before_switch: Callable[[], None] | None = None,
    ) -> Project:
        with self._lock:
            project = self.catalog.create(name, activate=False)
            try:
                return self.activate(project.project_id, before_switch)
            except BaseException:
                self.catalog.delete(project.project_id)
                raise

    def rename(self, project_id: str, name: str) -> ProjectSummary:
        with self._lock:
            current = self._session
            repository = None
            if current is not None:
                active = current.repository.load()
                if active.project_id == project_id:
                    repository = current.repository
            return project_summary(
                self.catalog.rename(project_id, name, repository=repository)
            )

    def delete(
        self,
        project_id: str,
        before_delete: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            current = self._session
            current_id = (
                None if current is None else current.repository.load().project_id
            )
            if current_id == project_id:
                assert current is not None
                if before_delete is not None:
                    before_delete()
                current.close()
                self._session = None
                try:
                    self.catalog.delete(project_id)
                except BaseException:
                    self._session = self._session_factory(project_id)
                    raise
                return
            self.catalog.delete(project_id)

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None


class ActiveProjectRepository:
    def __init__(self, manager: ActiveProjectManager) -> None:
        self._manager = manager

    @property
    def root(self):  # type: ignore[no-untyped-def]
        return self._manager.session().repository.root

    def load(self) -> Project:
        return self._manager.session().repository.load()

    def save(self, project: Project) -> None:
        self._manager.session().repository.save(project)

    def update(self, mutation):  # type: ignore[no-untyped-def]
        return self._manager.session().repository.update(mutation)

    def update_stage(self, name, state):  # type: ignore[no-untyped-def]
        return self._manager.session().repository.update_stage(name, state)

    def compare_and_set_stage(self, name, state, guard):  # type: ignore[no-untyped-def]
        return self._manager.session().repository.compare_and_set_stage(
            name, state, guard
        )

    def claim_stage(self, name, *, reuse_succeeded, run_id):  # type: ignore[no-untyped-def]
        return self._manager.session().repository.claim_stage(
            name, reuse_succeeded=reuse_succeeded, run_id=run_id
        )


class ActivePipelineRunner:
    def __init__(
        self,
        manager: ActiveProjectManager,
        supported_stages: tuple[StageName, ...] = tuple(StageName),
    ) -> None:
        self._manager = manager
        self._supported_stages = frozenset(supported_stages)

    def supports(self, name: StageName) -> bool:
        try:
            runner = self._manager.session().pipeline_runner
        except ActiveProjectRequiredError:
            return name in self._supported_stages
        return bool(runner.supports(name))

    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        return cast(
            StageState,
            self._manager.session().pipeline_runner.run(name, token, emit),
        )

    def run_outcome(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> Any:
        return self._manager.session().pipeline_runner.run_outcome(name, token, emit)


class ActivePreviewArtifactStore:
    def __init__(self, manager: ActiveProjectManager) -> None:
        self._manager = manager

    def publish(self, *args: Any, **kwargs: Any) -> Any:
        return self._manager.session().preview_artifacts.publish(*args, **kwargs)

    def read(self, *args: Any, **kwargs: Any) -> Any:
        return self._manager.session().preview_artifacts.read(*args, **kwargs)

    def pick_buffer(self, *args: Any, **kwargs: Any) -> Any:
        return self._manager.session().preview_artifacts.pick_buffer(*args, **kwargs)
