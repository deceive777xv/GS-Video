from __future__ import annotations

import asyncio
import secrets
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from uuid import uuid4

from pydantic import ValidationError
from starlette.websockets import WebSocket, WebSocketDisconnect

from gs_video.api.schemas import ApiError, ApiSettings, TaskEvent, TaskSnapshot, TaskStatus
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.pipeline.cancellation import CancellationToken


class PipelineRunnerLike(Protocol):
    def run(self, name: StageName, token: CancellationToken) -> StageState: ...


class EventBus:
    def __init__(self, window_size: int) -> None:
        self._events: deque[TaskEvent] = deque(maxlen=window_size)
        self._revision = 0
        self._condition = asyncio.Condition()

    @property
    def revision(self) -> int:
        return self._revision

    async def publish(
        self,
        *,
        task_id: str,
        stage: StageName,
        progress: float,
        error: dict[str, object] | None = None,
    ) -> TaskEvent:
        async with self._condition:
            self._revision += 1
            event = TaskEvent(
                task_id=task_id,
                revision=self._revision,
                stage=stage.value,
                progress=progress,
                error=error,
            )
            self._events.append(event)
            self._condition.notify_all()
            return event

    def after(self, revision: int) -> list[TaskEvent] | None:
        if self._events and revision < self._events[0].revision - 1:
            return None
        return [event for event in self._events if event.revision > revision]

    async def wait_after(self, revision: int) -> list[TaskEvent] | None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._revision > revision)
            return self.after(revision)


class TaskService:
    def __init__(
        self,
        runner: PipelineRunnerLike,
        events: EventBus,
        *,
        max_tasks: int,
        workers: int,
        shutdown_timeout: float,
    ) -> None:
        self._runner = runner
        self._events = events
        self._max_tasks = max_tasks
        self._shutdown_timeout = shutdown_timeout
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gs-video-task")
        self._snapshots: dict[str, TaskSnapshot] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._jobs: set[asyncio.Task[None]] = set()
        self._started = False

    async def start(self) -> None:
        self._started = True

    async def create(self, target_stage: StageName) -> TaskSnapshot:
        if not self._started:
            raise ApiError(
                503,
                code="task_service_unavailable",
                category="task",
                message="The task service is not running.",
                retryable=True,
            )
        if len(self._snapshots) >= self._max_tasks:
            raise ApiError(
                429,
                code="task_limit_reached",
                category="task",
                message="The in-process task limit has been reached.",
                retryable=True,
            )
        task_id = uuid4().hex
        event = await self._events.publish(
            task_id=task_id, stage=target_stage, progress=0.0
        )
        snapshot = TaskSnapshot(
            id=task_id,
            target_stage=target_stage.value,
            status=TaskStatus.QUEUED.value,
            revision=event.revision,
        )
        self._snapshots[task_id] = snapshot
        self._tokens[task_id] = CancellationToken()
        job = asyncio.create_task(self._run(task_id, target_stage))
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)
        return snapshot

    def get(self, task_id: str) -> TaskSnapshot:
        try:
            return self._snapshots[task_id]
        except KeyError as error:
            raise ApiError(
                404,
                code="task_not_found",
                category="task",
                message="The requested task was not found.",
            ) from error

    async def _update(
        self,
        task_id: str,
        stage: StageName,
        status: TaskStatus,
        progress: float,
        error: str | None = None,
    ) -> TaskSnapshot:
        event_error = (
            None
            if error is None
            else {"code": error, "category": "task", "retryable": False}
        )
        event = await self._events.publish(
            task_id=task_id,
            stage=stage,
            progress=progress,
            error=event_error,
        )
        current = self._snapshots[task_id]
        snapshot = current.model_copy(
            update={"status": status.value, "revision": event.revision, "error": error}
        )
        self._snapshots[task_id] = snapshot
        return snapshot

    async def _run(self, task_id: str, stage: StageName) -> None:
        await self._update(task_id, stage, TaskStatus.RUNNING, 0.0)
        loop = asyncio.get_running_loop()
        token = self._tokens[task_id]
        try:
            result = await loop.run_in_executor(self._executor, self._runner.run, stage, token)
        except Exception:
            if self._snapshots[task_id].status != TaskStatus.CANCELLED.value:
                await self._update(task_id, stage, TaskStatus.FAILED, 1.0, "task_failed")
            return
        if self._snapshots[task_id].status == TaskStatus.CANCELLED.value:
            return
        mapped = {
            StageStatus.CANCELLED: TaskStatus.CANCELLED,
            StageStatus.FAILED: TaskStatus.FAILED,
        }.get(result.status, TaskStatus.SUCCEEDED)
        error = result.error_code if mapped is TaskStatus.FAILED else None
        await self._update(task_id, stage, mapped, 1.0, error)

    async def cancel(self, task_id: str) -> TaskSnapshot:
        snapshot = self.get(task_id)
        if snapshot.status in {
            TaskStatus.SUCCEEDED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        }:
            return snapshot
        self._tokens[task_id].cancel()
        return await self._update(
            task_id,
            StageName(snapshot.target_stage),
            TaskStatus.CANCELLED,
            1.0,
        )

    async def cancel_all(self) -> None:
        for task_id in tuple(self._tokens):
            await self.cancel(task_id)
        if self._jobs:
            await asyncio.wait(self._jobs, timeout=self._shutdown_timeout)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._started = False


async def serve_events(websocket: WebSocket, settings: ApiSettings, events: EventBus) -> None:
    await websocket.accept()
    if websocket.headers.get("origin") not in settings.allowed_origins:
        await websocket.close(code=1008, reason="origin not allowed")
        return
    try:
        raw_auth = await asyncio.wait_for(
            websocket.receive_json(), timeout=settings.websocket_auth_timeout
        )
        if not isinstance(raw_auth, dict) or set(raw_auth) != {"type", "token"}:
            raise ValueError("invalid authentication message")
        token = raw_auth.get("token")
        if (
            raw_auth.get("type") != "authenticate"
            or not isinstance(token, str)
            or not secrets.compare_digest(token, settings.session_token)
        ):
            raise ValueError("invalid authentication message")
        await websocket.send_json({"type": "authenticated", "revision": events.revision})
        raw_resume = await asyncio.wait_for(
            websocket.receive_json(), timeout=settings.websocket_auth_timeout
        )
        if not isinstance(raw_resume, dict) or set(raw_resume) != {"type", "after_revision"}:
            raise ValueError("invalid resume message")
        revision = raw_resume.get("after_revision")
        if (
            raw_resume.get("type") != "resume"
            or type(revision) is not int
            or revision < 0
        ):
            raise ValueError("invalid resume message")
        while True:
            pending = events.after(revision)
            if pending is None:
                await websocket.send_json(
                    {"type": "resync_required", "revision": events.revision}
                )
                return
            if not pending:
                pending = await events.wait_after(revision)
                if pending is None:
                    await websocket.send_json(
                        {"type": "resync_required", "revision": events.revision}
                    )
                    return
            for event in pending:
                await websocket.send_json(event.model_dump(mode="json"))
                revision = event.revision
    except (TimeoutError, ValueError, ValidationError):
        await websocket.close(code=1008, reason="authentication required")
    except WebSocketDisconnect:
        return
