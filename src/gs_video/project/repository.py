import json
import os
from pathlib import Path

from gs_video.domain.models import Project
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

    def create(self, name: str) -> Project:
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in PROJECT_DIRECTORIES:
            (self.root / folder).mkdir(exist_ok=True)
        return Project(name=name)

    def load(self) -> Project:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return Project.model_validate(migrate_project_dict(raw))

    def save(self, project: Project) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, self.path)
