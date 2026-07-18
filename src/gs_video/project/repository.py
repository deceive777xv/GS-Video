import json
import os
from collections.abc import Callable
from pathlib import Path
from threading import RLock

from gs_video.domain.models import (
    Project,
    StageClaimResult,
    StageName,
    StageState,
    StageWriteGuard,
    StageWriteResult,
    StageStatus,
)
from gs_video.project.migrations import migrate_project_dict


PROJECT_DIRECTORIES = (
    "source",
    "frames",
    "proxies",
    "masks",
    "camera",
    "trajectories",
    "renders",
    "composites",
    "previews",
    "exports",
    "logs",
)


class ProjectRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / "project.json"
        self._lock = RLock()

    def create(self, name: str) -> Project:
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in PROJECT_DIRECTORIES:
            (self.root / folder).mkdir(exist_ok=True)
        return Project(name=name)

    def load(self) -> Project:
        with self._lock:
            return self._load_unlocked()

    def save(self, project: Project) -> None:
        with self._lock:
            self._save_unlocked(project)

    def update(self, mutation: Callable[[Project], None]) -> Project:
        with self._lock:
            project = self._load_unlocked()
            mutation(project)
            self._save_unlocked(project)
            return project.model_copy(deep=True)

    def update_stage(self, name: StageName, state: StageState) -> Project:
        return self.update(
            lambda project: project.stages.__setitem__(
                name, state.model_copy(deep=True)
            )
        )

    def compare_and_set_stage(
        self,
        name: StageName,
        state: StageState,
        guard: StageWriteGuard,
    ) -> StageWriteResult:
        with self._lock:
            project = self._load_unlocked()
            current = project.stages.get(name, StageState())
            matches = (
                current.input_generation == guard.input_generation
                and current.status is guard.status
                and current.run_id == guard.run_id
            )
            if matches:
                project.stages[name] = state.model_copy(deep=True)
                self._save_unlocked(project)
            return StageWriteResult(
                project=project.model_copy(deep=True), applied=matches
            )

    def claim_stage(
        self,
        name: StageName,
        *,
        reuse_succeeded: bool,
        run_id: str,
    ) -> StageClaimResult:
        with self._lock:
            project = self._load_unlocked()
            current = project.stages.get(name, StageState())
            if current.status is StageStatus.RUNNING:
                return StageClaimResult(
                    project=project.model_copy(deep=True), claimed=False
                )
            if reuse_succeeded and current.status is StageStatus.SUCCEEDED:
                return StageClaimResult(
                    project=project.model_copy(deep=True), claimed=False
                )
            running = current.model_copy(deep=True)
            running.status = StageStatus.RUNNING
            running.cache_key = None
            running.error_code = None
            running.run_id = run_id
            project.stages[name] = running
            self._save_unlocked(project)
            return StageClaimResult(
                project=project.model_copy(deep=True), claimed=True
            )

    def _load_unlocked(self) -> Project:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return Project.model_validate(migrate_project_dict(raw))

    def _save_unlocked(self, project: Project) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
