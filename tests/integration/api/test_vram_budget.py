from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.environment.vram import VramBudgetManager
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.repository import ProjectRepository


TOKEN = "vram-budget-test-token"
ORIGIN = "http://127.0.0.1:5173"


class BudgetDoctor:
    def __init__(self, budget: VramBudgetManager) -> None:
        self._budget = budget

    def check(self) -> EnvironmentReport:
        return EnvironmentReport(
            ready=True,
            vram_mb=24_576,
            vram_limit_mb=self._budget.current_limit_mb(),
            issues=[],
        )


class BlockingRunner:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        del name, emit
        self.started.set()
        assert self.release.wait(2)
        token.raise_if_cancelled()
        return StageState(status=StageStatus.SUCCEEDED, cache_key="done")


class WorkerRegistryFake:
    async def terminate_all(self) -> None:
        return None


class PreviewLifecycleFake:
    def __init__(self) -> None:
        self.suspend_calls = 0
        self.resume_calls = 0

    def suspend_live(self) -> object:
        self.suspend_calls += 1
        return object()

    def resume_live(self, _token: object) -> None:
        self.resume_calls += 1

    def render_pick(self, *args: object) -> object:
        del args
        raise AssertionError("preview rendering is outside this test")


@pytest.fixture
def budget_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, VramBudgetManager, PreviewLifecycleFake, BlockingRunner]]:
    repository = ProjectRepository(tmp_path / "project")
    repository.save(repository.create("budget"))
    budget = VramBudgetManager(
        tmp_path / "runtime" / "user-settings.json",
        total_vram_probe=lambda: 24_576,
        initial_limit_mb=8192,
    )
    preview = PreviewLifecycleFake()
    runner = BlockingRunner()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=BudgetDoctor(budget),
        pipeline_runner=runner,
        worker_registry=WorkerRegistryFake(),
        preview_service=preview,
        vram_budget=budget,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    with TestClient(create_app(settings, services)) as client:
        yield client, budget, preview, runner
        runner.release.set()


def test_vram_budget_endpoint_is_authenticated_and_updates_authority_immediately(
    budget_client: tuple[
        TestClient, VramBudgetManager, PreviewLifecycleFake, BlockingRunner
    ],
) -> None:
    client, budget, preview, _runner = budget_client
    headers = {"Authorization": f"Bearer {TOKEN}"}

    unauthorized = client.get("/api/v1/runtime/vram-budget")
    before = client.get("/api/v1/runtime/vram-budget", headers=headers)
    updated = client.patch(
        "/api/v1/runtime/vram-budget",
        json={"mode": "custom", "selected_vram_mb": 24_321},
        headers=headers,
    )
    bootstrap = client.get("/api/v1/bootstrap", headers=headers)

    assert unauthorized.status_code == 401
    assert before.json()["selected_vram_mb"] == 8192
    assert updated.status_code == 200, updated.text
    assert updated.json() == {
        "mode": "custom",
        "minimum_vram_mb": 1024,
        "total_vram_mb": 24_576,
        "selected_vram_mb": 24_321,
        "editable": True,
        "blocked_reason": None,
        "recovered_from_invalid_preference": False,
    }
    assert budget.current_limit_mb() == 24_321
    assert preview.suspend_calls == 2
    assert preview.resume_calls == 2
    assert bootstrap.json()["environment"] == {
        "ready": True,
        "vram_mb": 24_576,
        "vram_limit_mb": 24_321,
        "issues": [],
        "renderer_versions": None,
    }
    assert bootstrap.json()["vram_budget"]["selected_vram_mb"] == 24_321
    assert "vram_budget" in bootstrap.json()["capabilities"]


def test_vram_budget_rejects_out_of_range_and_active_gpu_task(
    budget_client: tuple[
        TestClient, VramBudgetManager, PreviewLifecycleFake, BlockingRunner
    ],
) -> None:
    client, budget, preview, runner = budget_client
    headers = {"Authorization": f"Bearer {TOKEN}"}

    invalid = client.patch(
        "/api/v1/runtime/vram-budget",
        json={"mode": "custom", "selected_vram_mb": 24_577},
        headers=headers,
    )
    task = client.post(
        "/api/v1/tasks",
        json={"target_stage": "segment"},
        headers=headers,
    )
    assert task.status_code == 202
    assert runner.started.wait(1)
    blocked = client.patch(
        "/api/v1/runtime/vram-budget",
        json={"mode": "custom", "selected_vram_mb": 12_288},
        headers=headers,
    )
    snapshot = client.get("/api/v1/runtime/vram-budget", headers=headers)

    assert invalid.status_code == 422, invalid.text
    assert invalid.json()["code"] == "invalid_vram_budget"
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "vram_budget_busy"
    assert snapshot.json()["editable"] is False
    assert snapshot.json()["blocked_reason"] == "gpu_task_active"
    assert budget.current_limit_mb() == 8192
    assert preview.suspend_calls == 2
    runner.release.set()
