import asyncio
import time
from pathlib import Path
from threading import Event
from typing import Protocol

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from gs_video.api.events import EventBus
from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.domain.errors import CancelledError
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.project.repository import ProjectRepository


TOKEN = "task-session-token"
ORIGIN = "http://localhost:5173"


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=False, vram_mb=0, issues=[])


class SucceedingRunner:
    def run(self, name: StageName, token: CancellationToken) -> StageState:
        token.raise_if_cancelled()
        return StageState(status=StageStatus.SUCCEEDED, cache_key=f"{name.value}-key")


class RunnerLike(Protocol):
    def run(self, name: StageName, token: CancellationToken) -> StageState: ...


class CancelAwareRunner:
    def __init__(self) -> None:
        self.started = Event()
        self.cancelled = Event()

    def run(self, name: StageName, token: CancellationToken) -> StageState:
        del name
        self.started.set()
        while True:
            try:
                token.raise_if_cancelled()
            except CancelledError:
                self.cancelled.set()
                return StageState(status=StageStatus.CANCELLED)
            self.cancelled.wait(0.01)


class SlowShutdownRunner:
    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay
        self.started = Event()
        self.finished = Event()

    def run(self, name: StageName, token: CancellationToken) -> StageState:
        del name, token
        self.started.set()
        self.finished.wait(self.delay)
        self.finished.set()
        return StageState(status=StageStatus.SUCCEEDED)


class SupersededRunner:
    def __init__(self, status: StageStatus = StageStatus.STALE) -> None:
        self.status = status

    def run(self, name: StageName, token: CancellationToken) -> StageState:
        del name, token
        return StageState(status=self.status)


class RecordingWorkerRegistry:
    def __init__(self) -> None:
        self.terminate_calls = 0

    async def terminate_all(self) -> None:
        self.terminate_calls += 1


def make_app(
    tmp_path: Path,
    *,
    event_window: int = 256,
    runner: RunnerLike | None = None,
    shutdown_timeout: float = 5.0,
) -> tuple[object, RecordingWorkerRegistry]:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("tasks"))
    registry = RecordingWorkerRegistry()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=runner or SucceedingRunner(),
        worker_registry=registry,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
        event_window=event_window,
        websocket_auth_timeout=0.05,
        shutdown_timeout=shutdown_timeout,
    )
    return create_app(settings, services), registry


def test_task_state_is_recoverable_without_websocket(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
        )
        task_id = created.json()["id"]
        snapshot = client.get(f"/api/v1/tasks/{task_id}", headers=headers).json()

    assert created.status_code == 202
    assert snapshot["status"] in {"queued", "running", "succeeded"}
    assert snapshot["target_stage"] == "segment"
    assert snapshot["revision"] >= 1


@pytest.mark.parametrize("stage_status", [StageStatus.STALE, StageStatus.RUNNING])
def test_uncommitted_stage_is_not_reported_as_task_success(
    tmp_path: Path, stage_status: StageStatus
) -> None:
    app, _ = make_app(tmp_path, runner=SupersededRunner(stage_status))
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
        )
        task_id = created.json()["id"]
        deadline = time.monotonic() + 1
        while True:
            snapshot = client.get(
                f"/api/v1/tasks/{task_id}", headers=headers
            ).json()
            if snapshot["status"] not in {"queued", "running"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)

    assert snapshot["status"] == "cancelled"


def test_websocket_authenticates_then_resumes_events(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
        ).json()
        with client.websocket_connect(
            "/api/v1/events", headers={"origin": ORIGIN}
        ) as websocket:
            websocket.send_json({"type": "authenticate", "token": TOKEN})
            authenticated = websocket.receive_json()
            websocket.send_json({"type": "resume", "after_revision": 0})
            event = websocket.receive_json()

    assert authenticated["type"] == "authenticated"
    assert event["type"] == "task_event"
    assert event["task_id"] == created["id"]
    assert set(event) == {
        "type",
        "task_id",
        "revision",
        "stage",
        "progress",
        "error",
    }


def test_websocket_rejects_bad_origin_and_missing_authentication(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        try:
            with client.websocket_connect(
                "/api/v1/events", headers={"origin": "https://attacker.invalid"}
            ) as websocket:
                websocket.receive_json()
        except WebSocketDisconnect as error:
            assert error.code == 1008
        else:
            raise AssertionError("disallowed WebSocket origin remained connected")

        try:
            with client.websocket_connect(
                "/api/v1/events", headers={"origin": ORIGIN}
            ) as websocket:
                websocket.receive_json()
        except WebSocketDisconnect as error:
            assert error.code == 1008
        else:
            raise AssertionError("unauthenticated WebSocket remained connected")


def test_lifespan_terminates_registered_worker_processes(tmp_path: Path) -> None:
    app, registry = make_app(tmp_path)

    with TestClient(app):
        assert registry.terminate_calls == 0

    assert registry.terminate_calls == 1


def test_task_cancel_maps_to_pipeline_cancellation_token(tmp_path: Path) -> None:
    runner = CancelAwareRunner()
    app, _ = make_app(tmp_path, runner=runner)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
        ).json()
        assert runner.started.wait(1)

        cancelled = client.delete(
            f"/api/v1/tasks/{created['id']}", headers=headers
        )

        assert cancelled.status_code == 202
        assert cancelled.json()["status"] == "cancelled"
        assert runner.cancelled.wait(1)
        snapshot = client.get(
            f"/api/v1/tasks/{created['id']}", headers=headers
        ).json()
        assert snapshot["status"] == "cancelled"


def test_cancelled_snapshot_cannot_be_overwritten_by_paused_success(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    success_publish_started = Event()
    release_success_publish = Event()
    event_bus = app.state.event_bus
    original_publish = event_bus.publish
    pause_first_terminal = True

    async def pause_success_publish(**kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal pause_first_terminal
        if kwargs.get("progress") == 1.0 and pause_first_terminal:
            pause_first_terminal = False
            success_publish_started.set()
            await asyncio.to_thread(release_success_publish.wait, 2)
        return await original_publish(**kwargs)

    event_bus.publish = pause_success_publish
    headers = {"Authorization": f"Bearer {TOKEN}"}
    cancelled_revision = 0
    task_id = ""
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
            ).json()
            task_id = created["id"]
            assert success_publish_started.wait(1)

            cancelled = client.delete(f"/api/v1/tasks/{task_id}", headers=headers).json()
            cancelled_revision = cancelled["revision"]
            assert cancelled["status"] == "cancelled"
            release_success_publish.set()
    finally:
        release_success_publish.set()

    final = app.state.task_service.get(task_id)
    assert final.status == "cancelled"
    assert final.revision == cancelled_revision


def test_event_window_requires_rest_resync_when_revision_is_too_old(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path, event_window=1)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/api/v1/tasks", json={"target_stage": "segment"}, headers=headers)
        client.post("/api/v1/tasks", json={"target_stage": "render"}, headers=headers)
        with client.websocket_connect(
            "/api/v1/events", headers={"origin": ORIGIN}
        ) as websocket:
            websocket.send_json({"type": "authenticate", "token": TOKEN})
            websocket.receive_json()
            websocket.send_json({"type": "resume", "after_revision": 0})
            event = websocket.receive_json()

    assert event["type"] == "resync_required"
    assert "revision" in event


def test_websocket_future_revision_resyncs_immediately(tmp_path: Path) -> None:
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/events", headers={"origin": ORIGIN}
        ) as websocket:
            websocket.send_json({"type": "authenticate", "token": TOKEN})
            authenticated = websocket.receive_json()
            websocket.send_json(
                {
                    "type": "resume",
                    "after_revision": authenticated["revision"] + 1,
                }
            )
            event = websocket.receive_json()

    assert event == {
        "type": "resync_required",
        "revision": authenticated["revision"],
    }


def test_idle_websocket_disconnect_unsubscribes_without_waiting_for_event(
    tmp_path: Path,
) -> None:
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        with client.websocket_connect(
            "/api/v1/events", headers={"origin": ORIGIN}
        ) as websocket:
            websocket.send_json({"type": "authenticate", "token": TOKEN})
            authenticated = websocket.receive_json()
            websocket.send_json(
                {"type": "resume", "after_revision": authenticated["revision"]}
            )
            deadline = time.monotonic() + 1
            while app.state.event_bus.subscriber_count != 1:
                assert time.monotonic() < deadline
                time.sleep(0.001)

        deadline = time.monotonic() + 1
        while app.state.event_bus.subscriber_count != 0:
            assert time.monotonic() < deadline
            time.sleep(0.001)


def test_event_subscription_queue_is_bounded_and_requests_resync() -> None:
    async def exercise() -> None:
        events = EventBus(window_size=1)
        pending, subscription = await events.subscribe(0)
        assert pending == []
        assert subscription is not None
        await events.publish(
            task_id="one", stage=StageName.SEGMENT, progress=0.5
        )
        await events.publish(
            task_id="two", stage=StageName.RENDER, progress=0.5
        )
        assert await subscription.next() is None
        events.unsubscribe(subscription)

    asyncio.run(exercise())


def test_lifespan_waits_for_task_thread_before_worker_cleanup(tmp_path: Path) -> None:
    runner = SlowShutdownRunner()
    app, registry = make_app(
        tmp_path, runner=runner, shutdown_timeout=0.05
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with TestClient(app) as client:
        client.post(
            "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
        )
        assert runner.started.wait(1)

    assert runner.finished.is_set()
    assert registry.terminate_calls == 1
