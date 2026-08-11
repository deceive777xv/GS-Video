from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.project.repository import ProjectRepository
from gs_video.storage.layout import StorageLayoutManager


TOKEN = "storage-layout-test-token"


class DoctorFake:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(
            ready=True,
            vram_mb=8192,
            vram_limit_mb=8192,
            issues=[],
        )


class RunnerFake:
    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter,
    ) -> StageState:
        del name, token, emit
        return StageState()

    def supports(self, name: StageName) -> bool:
        del name
        return True


class WorkersFake:
    async def terminate_all(self) -> None:
        return None


class PreviewFake:
    def __init__(self) -> None:
        self.suspended = 0
        self.resumed = 0

    def suspend_live(self) -> object:
        self.suspended += 1
        return object()

    def resume_live(self, token: object) -> None:
        del token
        self.resumed += 1


@pytest.fixture
def storage_client(
    tmp_path: Path,
) -> Iterator[tuple[TestClient, StorageLayoutManager, PreviewFake, Path]]:
    repository = ProjectRepository(tmp_path / "repository")
    repository.save(repository.create("storage"))
    layout = StorageLayoutManager(
        tmp_path / "data",
        tmp_path / "runtime" / "user-settings.json",
        drive_type_probe=lambda _path: 3,
    )
    preview = PreviewFake()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=DoctorFake(),
        pipeline_runner=RunnerFake(),
        worker_registry=WorkersFake(),
        preview_service=preview,
        storage_layout=layout,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(),
    )
    with TestClient(create_app(settings, services)) as client:
        yield client, layout, preview, tmp_path


def test_storage_layout_endpoint_switches_roots_and_requires_restart(
    storage_client: tuple[TestClient, StorageLayoutManager, PreviewFake, Path],
) -> None:
    client, layout, preview, root = storage_client
    headers = {"Authorization": f"Bearer {TOKEN}"}
    (layout.project_library_root / "catalog.json").write_text('{"projects":[]}')

    before = client.get("/api/v1/runtime/storage-layout", headers=headers)
    updated = client.patch(
        "/api/v1/runtime/storage-layout",
        headers=headers,
        json={
            "project_library_root": str(root / "new-project-library"),
            "cache_root": str(root / "new-cache"),
            "project_action": "migrate",
            "cache_action": "start_fresh",
        },
    )

    assert before.status_code == 200
    assert updated.status_code == 200
    assert updated.json()["restart_required"] is True
    assert updated.json()["blocked_reason"] == "restart_required"
    assert preview.suspended == 1
    assert preview.resumed == 0

    blocked = client.post(
        "/api/v1/tasks",
        headers=headers,
        json={"target_stage": "ingest"},
    )
    still_readable = client.get("/api/v1/runtime/storage-layout", headers=headers)

    assert blocked.status_code == 409
    assert blocked.json()["code"] == "storage_restart_required"
    assert still_readable.status_code == 200


def test_bootstrap_returns_storage_status_without_measuring_usage(
    storage_client: tuple[TestClient, StorageLayoutManager, PreviewFake, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, layout, _preview, _root = storage_client
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def reject_measurement(**_kwargs: object) -> object:
        raise AssertionError("bootstrap must not measure storage usage")

    monkeypatch.setattr(layout, "snapshot", reject_measurement)

    response = client.get("/api/v1/bootstrap", headers=headers)

    assert response.status_code == 200
    storage = response.json()["storage_layout"]
    assert storage == {
        "project_library_root": str(layout.project_library_root),
        "project_library_id": layout.status().project_library_id,
        "cache_root": str(layout.cache_root),
        "cache_id": layout.status().cache_id,
        "restart_required": False,
        "editable": True,
        "blocked_reason": None,
    }


def test_storage_layout_endpoint_rejects_overlapping_roots(
    storage_client: tuple[TestClient, StorageLayoutManager, PreviewFake, Path],
) -> None:
    client, _layout, preview, root = storage_client
    headers = {"Authorization": f"Bearer {TOKEN}"}
    target = root / "overlap"

    response = client.patch(
        "/api/v1/runtime/storage-layout",
        headers=headers,
        json={
            "project_library_root": str(target),
            "cache_root": str(target / "cache"),
            "project_action": "migrate",
            "cache_action": "start_fresh",
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "storage_layout_change_failed"
    assert preview.suspended == 1
    assert preview.resumed == 1


def test_cache_cleanup_requires_an_unchanged_backend_plan(
    storage_client: tuple[TestClient, StorageLayoutManager, PreviewFake, Path],
) -> None:
    client, layout, _preview, _root = storage_client
    headers = {"Authorization": f"Bearer {TOKEN}"}
    cache_entry = (
        layout.cache_root
        / "projects"
        / str(uuid4())
        / "frames"
        / ("a" * 64)
    )
    cache_entry.mkdir(parents=True)
    artifact = cache_entry / "frame.png"
    artifact.write_bytes(b"first")
    plan = client.post(
        "/api/v1/runtime/storage-layout/cache-cleanup/plan",
        headers=headers,
        json={"mode": "safe"},
    )
    artifact.write_bytes(b"changed")

    stale = client.post(
        "/api/v1/runtime/storage-layout/cache-cleanup",
        headers=headers,
        json={"mode": "safe", "plan_token": plan.json()["plan_token"]},
    )

    assert plan.status_code == 200
    assert stale.status_code == 422
    assert stale.json()["code"] == "storage_cleanup_failed"
