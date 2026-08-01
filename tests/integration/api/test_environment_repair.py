from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings, EnvironmentRepairSnapshot, EnvironmentRepairState
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.repository import ProjectRepository


TOKEN = "environment-repair-test-token"
ORIGIN = "http://127.0.0.1:5173"


class ReadyDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=8192, issues=[])


class RepairFake:
    def __init__(self) -> None:
        self.current = EnvironmentRepairSnapshot(state=EnvironmentRepairState.IDLE)
        self.shutdown_calls = 0

    def snapshot(self) -> EnvironmentRepairSnapshot:
        return self.current

    def start(self) -> EnvironmentRepairSnapshot:
        self.current = EnvironmentRepairSnapshot(
            state=EnvironmentRepairState.RUNNING,
            job_id="repair-test",
            step="preflight",
            message="正在准备环境修复…",
        )
        return self.current

    def cancel(self) -> EnvironmentRepairSnapshot:
        self.current = self.current.model_copy(
            update={"state": EnvironmentRepairState.CANCELLED}
        )
        return self.current

    def is_busy(self) -> bool:
        return self.current.state in {
            EnvironmentRepairState.RUNNING,
            EnvironmentRepairState.CANCELLING,
        }

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class PipelineFake:
    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        del name, emit
        token.raise_if_cancelled()
        return StageState(status=StageStatus.SUCCEEDED, cache_key="test")


class WorkerFake:
    async def terminate_all(self) -> None:
        return None


class PreviewFake:
    def render_pick(self, *args: object) -> object:
        del args
        raise AssertionError("preview rendering is outside this test")


@pytest.fixture
def repair_client(tmp_path: Path) -> Iterator[tuple[TestClient, RepairFake]]:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("repair"))
    repair = RepairFake()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=ReadyDoctor(),
        pipeline_runner=PipelineFake(),
        worker_registry=WorkerFake(),
        preview_service=PreviewFake(),
        environment_repair=repair,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    with TestClient(create_app(settings, services)) as client:
        yield client, repair
    assert repair.shutdown_calls == 1


def test_environment_repair_endpoints_expose_task_state(
    repair_client: tuple[TestClient, RepairFake],
) -> None:
    client, repair = repair_client
    headers = {"Authorization": f"Bearer {TOKEN}"}

    bootstrap = client.get("/api/v1/bootstrap", headers=headers)
    started = client.post("/api/v1/environment/repair", headers=headers)
    polled = client.get("/api/v1/environment/repair", headers=headers)
    blocked = client.post(
        "/api/v1/tasks", json={"target_stage": "segment"}, headers=headers
    )
    cancelled = client.delete("/api/v1/environment/repair", headers=headers)

    assert "environment_repair" in bootstrap.json()["capabilities"]
    assert started.status_code == 202
    assert started.json()["state"] == "running"
    assert polled.json()["job_id"] == "repair-test"
    assert blocked.status_code == 503
    assert blocked.json()["code"] == "environment_repair_in_progress"
    assert cancelled.status_code == 202
    assert cancelled.json()["state"] == "cancelled"
