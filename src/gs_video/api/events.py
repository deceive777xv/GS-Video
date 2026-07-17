from __future__ import annotations

import asyncio
import secrets
from collections import deque
from collections.abc import Awaitable, Callable
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


class EventSubscription:
    def __init__(self, window_size: int) -> None:
        self._queue: asyncio.Queue[TaskEvent | None] = asyncio.Queue(
            maxsize=window_size
        )
        self._overflowed = False

    def push(self, event: TaskEvent) -> None:
        if self._overflowed:
            return
        if self._queue.full():
            while not self._queue.empty():
                self._queue.get_nowait()
            self._queue.put_nowait(None)
            self._overflowed = True
            return
        self._queue.put_nowait(event)

    async def next(self) -> TaskEvent | None:
        return await self._queue.get()


class EventBus:
    def __init__(self, window_size: int) -> None:
        self._events: deque[TaskEvent] = deque(maxlen=window_size)
        self._revision = 0
        self._condition = asyncio.Condition()
        self._window_size = window_size
        self._subscriptions: set[EventSubscription] = set()

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def subscriber_count(self) -> int:
        return len(self._subscriptions)

    async def publish(
        self,
        *,
        task_id: str,
        stage: StageName,
        progress: float,
        error: dict[str, object] | None = None,
        commit: Callable[[TaskEvent], Awaitable[bool]] | None = None,
    ) -> TaskEvent | None:
        async with self._condition:
            event = TaskEvent(
                task_id=task_id,
                revision=self._revision + 1,
                stage=stage.value,
                progress=progress,
                error=error,
            )
            if commit is not None and not await commit(event):
                return None
            self._revision = event.revision
            self._events.append(event)
            for subscription in tuple(self._subscriptions):
                subscription.push(event)
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

    async def subscribe(
        self, revision: int
    ) -> tuple[list[TaskEvent] | None, EventSubscription | None]:
        async with self._condition:
            if revision > self._revision:
                return None, None
            pending = self.after(revision)
            if pending is None:
                return None, None
            subscription = EventSubscription(self._window_size)
            self._subscriptions.add(subscription)
            return pending, subscription

    def unsubscribe(self, subscription: EventSubscription) -> None:
        self._subscriptions.discard(subscription)


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
        del workers
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gs-video-task")
        self._snapshots: dict[str, TaskSnapshot] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._jobs: set[asyncio.Task[None]] = set()
        self._started = False
        self._transition_lock = asyncio.Lock()
        self._shutdown = False

    async def start(self) -> None:
        async with self._transition_lock:
            self._started = True

    async def create(self, target_stage: StageName) -> TaskSnapshot:
        task_id = uuid4().hex
        token = CancellationToken()
        committed: TaskSnapshot | None = None

        async def commit(event: TaskEvent) -> bool:
            nonlocal committed
            async with self._transition_lock:
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
                committed = TaskSnapshot(
                    id=task_id,
                    target_stage=target_stage.value,
                    status=TaskStatus.QUEUED.value,
                    revision=event.revision,
                )
                self._snapshots[task_id] = committed
                self._tokens[task_id] = token
                return True

        event = await self._events.publish(
            task_id=task_id,
            stage=target_stage,
            progress=0.0,
            commit=commit,
        )
        assert event is not None
        assert committed is not None
        job = asyncio.create_task(self._run(task_id, target_stage))
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)
        return committed

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
        committed: TaskSnapshot | None = None
        event_error = (
            None
            if error is None
            else {"code": error, "category": "task", "retryable": False}
        )

        async def commit(event: TaskEvent) -> bool:
            nonlocal committed
            async with self._transition_lock:
                current = self._snapshots[task_id]
                legal_sources = {
                    TaskStatus.RUNNING: {TaskStatus.QUEUED.value},
                    TaskStatus.SUCCEEDED: {TaskStatus.RUNNING.value},
                    TaskStatus.FAILED: {TaskStatus.RUNNING.value},
                    TaskStatus.CANCELLED: {
                        TaskStatus.QUEUED.value,
                        TaskStatus.RUNNING.value,
                    },
                }[status]
                if current.status not in legal_sources:
                    committed = current
                    return False
                committed = current.model_copy(
                    update={
                        "status": status.value,
                        "revision": event.revision,
                        "error": error,
                    }
                )
                self._snapshots[task_id] = committed
                return True

        event = await self._events.publish(
            task_id=task_id,
            stage=stage,
            progress=progress,
            error=event_error,
            commit=commit,
        )
        del event
        assert committed is not None
        return committed

    async def _run(self, task_id: str, stage: StageName) -> None:
        running = await self._update(task_id, stage, TaskStatus.RUNNING, 0.0)
        if running.status != TaskStatus.RUNNING.value:
            return
        loop = asyncio.get_running_loop()
        token = self._tokens[task_id]
        try:
            result = await loop.run_in_executor(self._executor, self._runner.run, stage, token)
        except Exception:
            await self._update(task_id, stage, TaskStatus.FAILED, 1.0, "task_failed")
            return
        mapped = {
            StageStatus.SUCCEEDED: TaskStatus.SUCCEEDED,
            StageStatus.FAILED: TaskStatus.FAILED,
            StageStatus.CANCELLED: TaskStatus.CANCELLED,
            StageStatus.STALE: TaskStatus.CANCELLED,
            StageStatus.PENDING: TaskStatus.CANCELLED,
            StageStatus.RUNNING: TaskStatus.CANCELLED,
        }[result.status]
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
        async with self._transition_lock:
            self._started = False
        for task_id in tuple(self._tokens):
            await self.cancel(task_id)
        if self._jobs:
            await asyncio.wait(self._jobs, timeout=self._shutdown_timeout)
        if not self._shutdown:
            await asyncio.to_thread(
                self._executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            self._shutdown = True
        if self._jobs:
            await asyncio.gather(*tuple(self._jobs), return_exceptions=True)


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


async def serve_events(websocket: WebSocket, settings: ApiSettings, events: EventBus) -> None:
    await websocket.accept()
    if websocket.headers.get("origin") not in settings.allowed_origins:
        await websocket.close(code=1008, reason="origin not allowed")
        return
    subscription: EventSubscription | None = None
    disconnect_task: asyncio.Task[None] | None = None
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
            or not secrets.compare_digest(token, settings.session_token.get_secret_value())
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
        pending, subscription = await events.subscribe(revision)
        if pending is None or subscription is None:
            await websocket.send_json(
                {"type": "resync_required", "revision": events.revision}
            )
            return
        for event in pending:
            await websocket.send_json(event.model_dump(mode="json"))
            revision = event.revision
        disconnect_task = asyncio.create_task(_wait_for_disconnect(websocket))
        while True:
            next_event = asyncio.create_task(subscription.next())
            done, _ = await asyncio.wait(
                {disconnect_task, next_event},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                next_event.cancel()
                await asyncio.gather(next_event, return_exceptions=True)
                return
            queued_event = next_event.result()
            if queued_event is None:
                await websocket.send_json(
                    {"type": "resync_required", "revision": events.revision}
                )
                return
            await websocket.send_json(queued_event.model_dump(mode="json"))
            revision = queued_event.revision
    except (TimeoutError, ValueError, ValidationError):
        await websocket.close(code=1008, reason="authentication required")
    except WebSocketDisconnect:
        return
    finally:
        if subscription is not None:
            events.unsubscribe(subscription)
        if disconnect_task is not None:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
