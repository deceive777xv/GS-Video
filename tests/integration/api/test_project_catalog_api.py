from collections.abc import Iterator
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.api.workflow import PreviewArtifactStore
from gs_video.app import create_app
from gs_video.domain.models import SceneSummary, StageName, StageState, VideoSummary
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.assets import AssetLibrary
from gs_video.project.catalog import ProjectCatalog
from gs_video.project.manager import (
    ActivePipelineRunner,
    ActivePreviewArtifactStore,
    ActiveProjectManager,
    ActiveProjectRepository,
    ActiveProjectSession,
)
from gs_video.project.repository import ProjectInstanceLock


TOKEN = "project-catalog-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
DESKTOP_HEADERS = {**HEADERS, "Origin": "http://tauri.localhost"}


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=8192, issues=[])


class Runner:
    def supports(self, name: StageName) -> bool:
        del name
        return True

    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        del name, emit
        token.raise_if_cancelled()
        return StageState()


class WorkerRegistry:
    async def terminate_all(self) -> None:
        pass


class PreviewService:
    def close_live(self) -> None:
        pass


class Inspector:
    def inspect(
        self, kind: str, path: Path, *, size: int, sha256: str
    ) -> VideoSummary | SceneSummary:
        if kind == "source_video":
            return VideoSummary(
                filename=path.name,
                size=size,
                sha256=sha256,
                width=1920,
                height=1080,
                duration_seconds=1.0,
                fps="30",
                has_audio=False,
                frame_count=30,
            )
        return SceneSummary(
            filename=path.name,
            size=size,
            sha256=sha256,
            gaussian_count=1,
            estimated_vram_mb=1,
        )


@pytest.fixture
def catalog_client(tmp_path: Path) -> Iterator[TestClient]:
    data_root = tmp_path / "data"
    catalog = ProjectCatalog(data_root)
    catalog.create("Initial")
    assets = AssetLibrary(data_root / "assets")

    def build(project_id: str) -> ActiveProjectSession:
        repository = catalog.repository(project_id)
        return ActiveProjectSession(
            repository=repository,
            project_lock=ProjectInstanceLock.acquire(repository.root),
            pipeline_runner=Runner(),
            preview_artifacts=PreviewArtifactStore(repository.root),
        )

    manager = ActiveProjectManager(catalog, build)
    services = ApiServices(
        project_repository=ActiveProjectRepository(manager),
        environment_doctor=StaticDoctor(),
        pipeline_runner=ActivePipelineRunner(manager),
        worker_registry=WorkerRegistry(),
        preview_service=PreviewService(),
        asset_inspector=Inspector(),
        project_catalog=catalog,
        asset_library=assets,
        project_manager=manager,
        preview_artifacts=ActivePreviewArtifactStore(manager),
        upload_root=data_root / "upload-staging",
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=("http://tauri.localhost", "http://127.0.0.1:5173"),
    )
    try:
        with TestClient(create_app(settings, services)) as client:
            yield client
    finally:
        manager.close()


def test_project_crud_switches_active_authority(catalog_client: TestClient) -> None:
    created = catalog_client.post(
        "/api/v1/projects", json={"name": "Second"}, headers=HEADERS
    )
    assert created.status_code == 201
    second_id = created.json()["project_id"]
    assert catalog_client.get(
        "/api/v1/projects/current", headers=HEADERS
    ).json()["project_id"] == second_id

    projects = catalog_client.get("/api/v1/projects", headers=HEADERS)
    first_id = next(
        item["project_id"] for item in projects.json() if item["name"] == "Initial"
    )
    activated = catalog_client.post(
        f"/api/v1/projects/{first_id}/activate", headers=HEADERS
    )
    assert activated.status_code == 200
    renamed = catalog_client.patch(
        f"/api/v1/projects/{second_id}",
        json={"name": "Renamed"},
        headers=HEADERS,
    )
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "Renamed"

    deleted = catalog_client.delete(
        f"/api/v1/projects/{second_id}", headers=HEADERS
    )
    assert deleted.status_code == 204
    assert [
        item["name"]
        for item in catalog_client.get("/api/v1/projects", headers=HEADERS).json()
    ] == ["Initial"]


def test_shared_assets_are_separated_referenced_and_protected(
    catalog_client: TestClient, tmp_path: Path
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    imported = catalog_client.post(
        "/api/v1/assets/import",
        json={
            "path": str(source),
            "kind": "source_video",
            "assign_to_current": False,
        },
        headers=DESKTOP_HEADERS,
    )
    assert imported.status_code == 201
    asset_id = imported.json()["asset_id"]
    assert len(catalog_client.get(
        "/api/v1/assets?kind=video", headers=HEADERS
    ).json()) == 1
    assert catalog_client.get(
        "/api/v1/assets?kind=ply", headers=HEADERS
    ).json() == []

    current_id = catalog_client.get(
        "/api/v1/projects/current", headers=HEADERS
    ).json()["project_id"]
    stale_selection = catalog_client.patch(
        "/api/v1/projects/current/assets",
        json={
            "expected_project_id": "stale-project-id",
            "source_video_asset_id": asset_id,
        },
        headers=HEADERS,
    )
    assert stale_selection.status_code == 409
    assert stale_selection.json()["code"] == "project_context_changed"
    assert catalog_client.get(
        "/api/v1/projects/current", headers=HEADERS
    ).json()["source_video_asset_id"] is None

    selected = catalog_client.patch(
        "/api/v1/projects/current/assets",
        json={
            "expected_project_id": current_id,
            "source_video_asset_id": asset_id,
        },
        headers=HEADERS,
    )
    assert selected.status_code == 200
    assert selected.json()["source_video_asset_id"] == asset_id
    assert selected.json()["source_video"] is None
    blocked = catalog_client.delete(f"/api/v1/assets/{asset_id}", headers=HEADERS)
    assert blocked.status_code == 409
    assert blocked.json()["code"] == "asset_in_use"

    cleared = catalog_client.patch(
        "/api/v1/projects/current/assets",
        json={
            "expected_project_id": current_id,
            "source_video_asset_id": None,
        },
        headers=HEADERS,
    )
    assert cleared.status_code == 200
    removed = catalog_client.delete(f"/api/v1/assets/{asset_id}", headers=HEADERS)
    assert removed.status_code == 204


@pytest.mark.parametrize("origin", [None, "http://127.0.0.1:5173"])
def test_non_desktop_request_cannot_submit_an_arbitrary_local_path(
    catalog_client: TestClient, tmp_path: Path, origin: str | None
) -> None:
    source = tmp_path / "browser-path.mp4"
    source.write_bytes(b"not-browser-readable")

    response = catalog_client.post(
        "/api/v1/assets/import",
        json={
            "path": str(source),
            "kind": "source_video",
            "assign_to_current": False,
        },
        headers=HEADERS if origin is None else {**HEADERS, "Origin": origin},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "desktop_import_required"


def test_current_project_patch_obeys_catalog_name_uniqueness(
    catalog_client: TestClient,
) -> None:
    created = catalog_client.post(
        "/api/v1/projects", json={"name": "Second"}, headers=HEADERS
    )
    assert created.status_code == 201

    response = catalog_client.patch(
        "/api/v1/projects/current",
        json={"name": "Initial"},
        headers=HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["code"] == "project_name_conflict"
    assert catalog_client.get(
        "/api/v1/projects/current", headers=HEADERS
    ).json()["name"] == "Second"


def test_browser_upload_can_add_to_library_without_assigning_current_project(
    catalog_client: TestClient,
) -> None:
    payload = b"browser-video"
    created = catalog_client.post(
        "/api/v1/uploads",
        json={
            "kind": "source_video",
            "filename": "library-only.mp4",
            "mime_type": "video/mp4",
            "total_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "assign_to_current": False,
        },
        headers=HEADERS,
    )
    assert created.status_code == 201
    upload_id = created.json()["id"]
    chunk = catalog_client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=payload,
        headers=HEADERS,
    )
    assert chunk.status_code == 204

    completed = catalog_client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=HEADERS
    )
    assert completed.status_code == 201
    assert completed.json()["assign_to_current"] is False
    assert catalog_client.get(
        "/api/v1/projects/current", headers=HEADERS
    ).json()["source_video_asset_id"] is None
    assets = catalog_client.get(
        "/api/v1/assets?kind=video", headers=HEADERS
    ).json()
    assert [item["asset"]["original_filename"] for item in assets] == [
        "library-only.mp4"
    ]


def test_busy_task_blocks_project_changes_and_current_asset_selection(
    catalog_client: TestClient, tmp_path: Path
) -> None:
    created = catalog_client.post(
        "/api/v1/projects", json={"name": "Second"}, headers=HEADERS
    )
    assert created.status_code == 201
    second_id = created.json()["project_id"]
    first_id = next(
        item["project_id"]
        for item in catalog_client.get("/api/v1/projects", headers=HEADERS).json()
        if item["name"] == "Initial"
    )
    source = tmp_path / "busy.mp4"
    source.write_bytes(b"busy-video")
    imported = catalog_client.post(
        "/api/v1/assets/import",
        json={"path": str(source), "kind": "source_video", "assign_to_current": False},
        headers=DESKTOP_HEADERS,
    )
    asset_id = imported.json()["asset_id"]

    task_service = catalog_client.app.state.task_service
    original_is_busy = task_service.is_busy
    task_service.is_busy = lambda: True
    try:
        responses = (
            catalog_client.post(
                "/api/v1/projects", json={"name": "Blocked"}, headers=HEADERS
            ),
            catalog_client.post(
                f"/api/v1/projects/{first_id}/activate", headers=HEADERS
            ),
            catalog_client.delete(
                f"/api/v1/projects/{second_id}", headers=HEADERS
            ),
            catalog_client.patch(
                "/api/v1/projects/current/assets",
                json={
                    "expected_project_id": second_id,
                    "source_video_asset_id": asset_id,
                },
                headers=HEADERS,
            ),
        )
    finally:
        task_service.is_busy = original_is_busy

    assert all(response.status_code == 409 for response in responses)
    assert all(response.json()["code"] == "project_task_active" for response in responses)
