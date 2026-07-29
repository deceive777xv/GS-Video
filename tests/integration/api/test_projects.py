from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.pipeline.runner import PipelineRunner
from gs_video.project.repository import ProjectRepository


TOKEN = "integration-session-token"
ORIGIN = "http://127.0.0.1:5173"


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=8192, issues=[])


class BlockingDoctor:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def check(self) -> EnvironmentReport:
        self.started.set()
        assert self.release.wait(2)
        return EnvironmentReport(ready=True, vram_mb=8192, issues=[])


class SucceedingRunner:
    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        del emit
        token.raise_if_cancelled()
        return StageState(status=StageStatus.SUCCEEDED, cache_key=f"{name.value}-key")


class RecordingWorkerRegistry:
    def __init__(self) -> None:
        self.terminate_calls = 0

    async def terminate_all(self) -> None:
        self.terminate_calls += 1


class UnusedPreviewService:
    def render_pick(self, *args: object) -> object:
        del args
        raise AssertionError("preview rendering is outside this test")


@pytest.fixture
def api_client(tmp_path: Path) -> Iterator[TestClient]:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("demo"))
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=RecordingWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    with TestClient(create_app(settings, services)) as client:
        yield client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def test_protected_route_rejects_missing_session_token(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/projects/current")

    assert response.status_code == 401
    assert response.json() == {
        "code": "invalid_session",
        "category": "authentication",
        "message": "A valid Bearer session token is required.",
        "retryable": False,
    }


def test_health_and_bootstrap_return_bounded_session_state(
    api_client: TestClient, auth_headers: dict[str, str]
) -> None:
    health = api_client.get("/healthz", headers=auth_headers)
    bootstrap = api_client.get("/api/v1/bootstrap", headers=auth_headers)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert bootstrap.status_code == 200
    body = bootstrap.json()
    assert body["api_version"] == "v1"
    assert set(body["capabilities"]) == {"projects", "assets", "uploads", "tasks", "events"}
    assert body["project"]["name"] == "demo"
    assert body["environment"] == {
        "ready": True,
        "vram_mb": 8192,
        "vram_limit_mb": 8192,
        "issues": [],
        "renderer_versions": None,
    }
    assert TOKEN not in bootstrap.text


def test_authenticated_shutdown_requests_server_exit(
    api_client: TestClient, auth_headers: dict[str, str]
) -> None:
    calls: list[str] = []
    api_client.app.state.request_shutdown = lambda: calls.append("shutdown")

    rejected = api_client.post("/api/v1/shutdown")
    accepted = api_client.post("/api/v1/shutdown", headers=auth_headers)

    assert rejected.status_code == 401
    assert accepted.status_code == 202
    assert calls == ["shutdown"]


def test_project_patch_is_strict_and_persists(
    api_client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = api_client.patch(
        "/api/v1/projects/current", json={"name": "renamed"}, headers=auth_headers
    )
    snapshot = api_client.get("/api/v1/projects/current", headers=auth_headers)
    invalid = api_client.patch(
        "/api/v1/projects/current",
        json={"name": "ignored", "output_path": "C:/escape.mp4"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert snapshot.json()["name"] == "renamed"
    assert invalid.status_code == 422
    assert set(invalid.json()) == {"code", "category", "message", "retryable"}


def test_local_asset_import_copies_into_project_source(
    api_client: TestClient, auth_headers: dict[str, str], tmp_path: Path
) -> None:
    selected = tmp_path / "selected video.mp4"
    selected.write_bytes(b"video-data")

    response = api_client.post(
        "/api/v1/assets/import",
        json={"path": str(selected), "kind": "source_video"},
        headers=auth_headers,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["path"].startswith("source/")
    assert not Path(body["path"]).is_absolute()
    project = api_client.get("/api/v1/projects/current", headers=auth_headers).json()
    assert project["source_video"] == body["path"]


def test_blocked_environment_probe_does_not_stall_other_rest_requests(
    tmp_path: Path,
) -> None:
    repository = ProjectRepository(tmp_path / "concurrent-project")
    repository.save(repository.create("concurrent"))
    doctor = BlockingDoctor()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=doctor,
        pipeline_runner=SucceedingRunner(),
        worker_registry=RecordingWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with TestClient(create_app(settings, services)) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            bootstrap = executor.submit(
                client.get, "/api/v1/bootstrap", headers=headers
            )
            assert doctor.started.wait(1)
            health = executor.submit(client.get, "/healthz", headers=headers)
            try:
                health_response = health.result(timeout=0.5)
            except FuturesTimeoutError:
                health_response = None
            finally:
                doctor.release.set()
            bootstrap_response = bootstrap.result(timeout=2)

    assert health_response is not None
    assert health_response.status_code == 200
    assert bootstrap_response.status_code == 200


def test_unassembled_production_workflow_is_rejected_before_task_admission(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    repository = ProjectRepository(tmp_path / "unassembled")
    project = repository.create("unassembled")
    repository.save(project)
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=PipelineRunner(
            project,
            {},
            save=repository.save,
            claim_stage=repository.claim_stage,
            compare_and_set_stage=repository.compare_and_set_stage,
        ),
        worker_registry=RecordingWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1", port=0, session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )

    with TestClient(create_app(settings, services)) as client:
        response = client.post(
            "/api/v1/tasks", json={"target_stage": "segment"},
            headers=auth_headers,
        )

    assert response.status_code == 503
    assert response.json()["code"] == "workflow_unavailable"
