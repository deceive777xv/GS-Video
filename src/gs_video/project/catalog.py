from __future__ import annotations

import json
import os
import re
import shutil
import stat
import time
from datetime import datetime
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from gs_video.domain.models import Project, StageName, StageStatus
from gs_video.project.repository import ProjectRepository
from gs_video.segmentation.paths import has_reparse_component


_PROJECT_NAME_LIMIT = 80
_PROJECT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_TRASH_ENTRY = re.compile(
    r"^(?P<project_id>[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12})-[0-9a-f]{32}$"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProjectCatalogState(_StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    project_ids: list[str] = Field(default_factory=list)
    active_project_id: str | None = None


class ProjectSummary(_StrictModel):
    project_id: str
    name: str
    created_at: datetime
    updated_at: datetime
    workflow_step: str
    active_task_id: str | None


def normalize_project_name(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > _PROJECT_NAME_LIMIT:
        raise ValueError("project name must contain between 1 and 80 characters")
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise ValueError("project name contains unsupported control characters")
    return normalized


def _workflow_step(project: Project) -> str:
    ordered: tuple[tuple[str, tuple[StageName, ...]], ...] = (
        ("export", (StageName.COMPOSITE,)),
        ("preview", (StageName.MAP_TRAJECTORY,)),
        ("camera", (StageName.SEGMENT,)),
        ("subject", (StageName.INGEST,)),
    )
    for step, required in ordered:
        if all(
            project.stages.get(stage) is not None
            and project.stages[stage].status is StageStatus.SUCCEEDED
            for stage in required
        ):
            return step
    return "import"


def project_summary(project: Project) -> ProjectSummary:
    return ProjectSummary(
        project_id=project.project_id,
        name=project.name,
        created_at=project.created_at,
        updated_at=project.updated_at,
        workflow_step=_workflow_step(project),
        active_task_id=project.workflow.active_task_id,
    )


def _ordinary_directory(path: Path) -> None:
    metadata = path.lstat()
    if has_reparse_component(path) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("project catalog contains an unsafe directory")


class ProjectCatalog:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self.projects_root = self.root / "projects"
        self.trash_root = self.projects_root / ".trash"
        self.path = self.root / "catalog.json"
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects_root.mkdir(exist_ok=True)
        self.trash_root.mkdir(exist_ok=True)
        for directory in (self.root, self.projects_root, self.trash_root):
            _ordinary_directory(directory)
        if not self.path.exists():
            self._save_state(ProjectCatalogState())
        state = ProjectCatalogState.model_validate_json(
            self.path.read_text(encoding="utf-8")
        )
        self._retry_pending_deletions(state)
        state = self._load_state()
        self._recover_orphan_projects(state)

    def _retry_pending_deletions(self, state: ProjectCatalogState) -> None:
        for entry in self.trash_root.iterdir():
            match = _TRASH_ENTRY.fullmatch(entry.name)
            if match is None:
                continue
            try:
                _ordinary_directory(entry)
                project_id = match.group("project_id")
                if project_id in state.project_ids:
                    destination = self._project_root(
                        project_id, require_exists=False
                    )
                    if destination.exists():
                        raise OSError(
                            "both live and staged project directories exist"
                        )
                    project = ProjectRepository(entry).load()
                    if project.project_id != project_id:
                        raise OSError(
                            "staged project directory does not match its project ID"
                        )
                    os.replace(entry, destination)
                else:
                    shutil.rmtree(entry)
            except OSError:
                continue

    def _recover_orphan_projects(self, state: ProjectCatalogState) -> None:
        recovered: list[str] = []
        for entry in self.projects_root.iterdir():
            if entry == self.trash_root or _PROJECT_ID.fullmatch(entry.name) is None:
                continue
            if entry.name in state.project_ids:
                continue
            _ordinary_directory(entry)
            project = ProjectRepository(entry).load()
            if project.project_id != entry.name:
                raise ValueError("orphan project directory does not match its project ID")
            recovered.append(entry.name)
        if not recovered:
            return
        state.project_ids.extend(sorted(recovered))
        if state.active_project_id is None:
            state.active_project_id = recovered[0]
        self._save_state(state)

    def _load_state(self) -> ProjectCatalogState:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        state = ProjectCatalogState.model_validate(raw)
        if len(set(state.project_ids)) != len(state.project_ids):
            raise ValueError("project catalog contains duplicate project IDs")
        for project_id in state.project_ids:
            self._project_root(project_id, require_exists=True)
        if (
            state.active_project_id is not None
            and state.active_project_id not in state.project_ids
        ):
            raise ValueError("active project is not present in the catalog")
        return state

    def _save_state(self, state: ProjectCatalogState) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        try:
            for attempt in range(4):
                try:
                    os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)

    def _project_root(self, project_id: str, *, require_exists: bool) -> Path:
        try:
            canonical_id = str(UUID(project_id))
        except ValueError as error:
            raise ValueError("invalid project ID") from error
        if canonical_id != project_id or _PROJECT_ID.fullmatch(project_id) is None:
            raise ValueError("invalid project ID")
        root = self.projects_root / project_id
        if root.parent != self.projects_root:
            raise ValueError("project path escaped the catalog root")
        if require_exists:
            _ordinary_directory(root)
        return root

    def _projects(self, state: ProjectCatalogState) -> list[Project]:
        return [
            ProjectRepository(
                self._project_root(project_id, require_exists=True)
            ).load()
            for project_id in state.project_ids
        ]

    def list(self) -> tuple[ProjectSummary, ...]:
        with self._lock:
            projects = self._projects(self._load_state())
            projects.sort(key=lambda item: item.updated_at, reverse=True)
            return tuple(project_summary(project) for project in projects)

    def active_project_id(self) -> str | None:
        with self._lock:
            return self._load_state().active_project_id

    def references(self, asset_id: str) -> tuple[ProjectSummary, ...]:
        with self._lock:
            projects = self._projects(self._load_state())
            return tuple(
                project_summary(project)
                for project in projects
                if asset_id
                in {
                    project.source_video_asset_id,
                    project.scene_ply_asset_id,
                }
            )

    def repository(self, project_id: str) -> ProjectRepository:
        with self._lock:
            state = self._load_state()
            if project_id not in state.project_ids:
                raise KeyError(project_id)
            return ProjectRepository(self._project_root(project_id, require_exists=True))

    def create(self, name: str, *, activate: bool = True) -> Project:
        normalized = normalize_project_name(name)
        with self._lock:
            state = self._load_state()
            projects = self._projects(state)
            if any(project.name.casefold() == normalized.casefold() for project in projects):
                raise ValueError("project name is already in use")
            project = Project(name=normalized)
            root = self._project_root(project.project_id, require_exists=False)
            repository = ProjectRepository(root)
            try:
                repository.create(normalized)
                repository.save(project)
                state.project_ids.append(project.project_id)
                if activate:
                    state.active_project_id = project.project_id
                self._save_state(state)
            except BaseException:
                if root.exists() and root.is_dir() and not has_reparse_component(root):
                    shutil.rmtree(root, ignore_errors=True)
                raise
            return repository.load()

    def adopt(self, legacy_root: Path, *, activate: bool = True) -> Project:
        legacy_root = Path(legacy_root).absolute()
        with self._lock:
            state = self._load_state()
            if state.project_ids:
                raise ValueError("legacy adoption requires an empty catalog")
            migration_root = self.root.parent
            if (
                legacy_root.parent != self.projects_root
                and not legacy_root.is_relative_to(migration_root)
            ):
                raise ValueError("legacy project is outside the managed projects root")
            _ordinary_directory(legacy_root)
            project = ProjectRepository(legacy_root).load()
            destination = self._project_root(
                project.project_id, require_exists=False
            )
            moved = legacy_root != destination
            if moved:
                if destination.exists():
                    raise ValueError("legacy project destination already exists")
                os.replace(legacy_root, destination)
            try:
                state.project_ids.append(project.project_id)
                if activate:
                    state.active_project_id = project.project_id
                self._save_state(state)
            except BaseException:
                if moved:
                    os.replace(destination, legacy_root)
                raise
            return ProjectRepository(destination).load()

    def activate(self, project_id: str) -> Project:
        with self._lock:
            state = self._load_state()
            if project_id not in state.project_ids:
                raise KeyError(project_id)
            repository = ProjectRepository(
                self._project_root(project_id, require_exists=True)
            )
            project = repository.load()
            state.active_project_id = project_id
            self._save_state(state)
            return project

    def rename(
        self,
        project_id: str,
        name: str,
        *,
        repository: ProjectRepository | None = None,
    ) -> Project:
        normalized = normalize_project_name(name)
        with self._lock:
            state = self._load_state()
            projects = self._projects(state)
            if project_id not in state.project_ids:
                raise KeyError(project_id)
            if any(
                project.project_id != project_id
                and project.name.casefold() == normalized.casefold()
                for project in projects
            ):
                raise ValueError("project name is already in use")
            authoritative_repository = repository or ProjectRepository(
                self._project_root(project_id, require_exists=True)
            )
            if authoritative_repository.root != self._project_root(
                project_id, require_exists=True
            ):
                raise ValueError("project repository does not match project ID")
            return authoritative_repository.update(
                lambda project: setattr(project, "name", normalized)
            )

    def delete(self, project_id: str) -> None:
        with self._lock:
            state = self._load_state()
            if project_id not in state.project_ids:
                raise KeyError(project_id)
            source = self._project_root(project_id, require_exists=True)
            destination = self.trash_root / f"{project_id}-{uuid4().hex}"
            os.replace(source, destination)
            try:
                state.project_ids.remove(project_id)
                if state.active_project_id == project_id:
                    state.active_project_id = None
                self._save_state(state)
            except BaseException:
                os.replace(destination, source)
                raise
            try:
                shutil.rmtree(destination)
            except OSError:
                pass
