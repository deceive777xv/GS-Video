import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

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


class ProjectInstanceLock:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._closed = False

    @classmethod
    def acquire(cls, root: Path) -> "ProjectInstanceLock":
        root.mkdir(parents=True, exist_ok=True)
        stream = (root / ".gs-video.lock").open("a+b")
        try:
            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            raise RuntimeError("project is already open in another process") from error
        return cls(stream)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._closed = True
            self._stream.close()

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


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

    def reconcile_interrupted_runs(self) -> Project:
        """Make persisted in-flight work retryable after an unclean process exit."""

        def reconcile(project: Project) -> None:
            project.workflow.active_task_id = None
            for state in project.stages.values():
                if state.status is StageStatus.RUNNING:
                    state.status = StageStatus.FAILED
                    state.cache_key = None
                    state.error_code = "interrupted"
                    state.run_id = None

        return self.update(reconcile)

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
        project.updated_at = datetime.now(timezone.utc)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(project.model_dump_json(indent=2), encoding="utf-8")
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
