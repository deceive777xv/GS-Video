from pathlib import Path
from threading import Event
from typing import Protocol

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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
