import json
import os
from collections.abc import Callable
from pathlib import Path
from threading import RLock

from gs_video.domain.models import Project, StageName, StageState
from gs_video.project.migrations import migrate_project_dict


PROJECT_DIRECTORIES = (
    "source",
    "proxies",
    "masks",
    "camera",
    "renders",
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

    def _load_unlocked(self) -> Project:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return Project.model_validate(migrate_project_dict(raw))

    def _save_unlocked(self, project: Project) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
