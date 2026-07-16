from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.media.ffmpeg import VideoMetadata
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.project.repository import ProjectRepository
from gs_video.scene.camera import OrbitCamera


TOKEN = "workflow-session-token"
ORIGIN = "http://127.0.0.1:5173"


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=8192, issues=[])


class SucceedingRunner:
    def run(self, name: StageName, token: CancellationToken) -> StageState:
        token.raise_if_cancelled()
        return StageState(status=StageStatus.SUCCEEDED, cache_key=f"{name.value}-key")


class Registry:
    async def terminate_all(self) -> None:
        return None


class PreviewService:
    def __init__(self) -> None:
        self.cameras: list[OrbitCamera] = []

    def render_pick(
        self,
        project_root: Path,
        scene_path: str,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer:
        del project_root, scene_path
        self.cameras.append(camera)
        rgb = np.full((height, width, 3), 96, dtype=np.uint8)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        return PickBuffer(rgb=rgb, expected_depth=depth)


class ExportInspector:
    def probe(self, path: Path) -> VideoMetadata:
        assert path.read_bytes() == b"verified-video"
        return VideoMetadata(
            width=1920,
            height=1080,
            duration=10.0,
            fps="30",
            has_audio=True,
            frame_count=300,
        )


@pytest.fixture
def workflow_client(tmp_path: Path) -> Iterator[TestClient]:
    repository = ProjectRepository(tmp_path / "project")
    project = repository.create("workflow")
    project.source_video = "source/video.mp4"
    project.scene_ply = "source/scene.ply"
    (repository.root / "exports" / "final.mp4").write_bytes(b"verified-video")
    project.stages[StageName.EXPORT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key="export-key",
        output_paths=["exports/final.mp4"],
    )
    repository.save(project)
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=Registry(),
        preview_service=PreviewService(),
        export_inspector=ExportInspector(),
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


def test_targeted_workflow_patch_persists_and_invalidates_only_downstream(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = workflow_client.patch(
        "/api/v1/projects/current",
        json={
            "subject_prompt": {"frame_index": 4, "x": 100, "y": 120},
            "motion_scale": 0.75,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow"]["subject_prompt"] == {
        "frame_index": 4,
        "x": 100,
        "y": 120,
    }
    assert body["workflow"]["motion_scale"] == 0.75
    persisted = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()
    assert persisted["workflow"] == body["workflow"]


def test_preview_and_pick_are_persisted_with_revision_guards(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    preview = workflow_client.post(
        "/api/v1/projects/current/preview",
        json={
            "generation": 1,
            "width": 16,
            "height": 9,
            "camera": {
                "target": [0.0, 0.0, 0.0],
                "distance": 4.0,
                "yaw": 0.0,
                "pitch": 0.0,
                "fov_y_degrees": 60.0,
            },
        },
        headers=auth_headers,
    )

    assert preview.status_code == 201
    descriptor = preview.json()
    assert descriptor["generation"] == 1
    assert descriptor["width"] == 16
    assert descriptor["height"] == 9
    assert descriptor["camera_revision"] == 1
    assert descriptor["pick_buffer_revision"] == 1
    assert "path" not in descriptor

    image = workflow_client.get(
        f"/api/v1/projects/current/previews/{descriptor['artifact_id']}",
        headers=auth_headers,
    )
    assert image.status_code == 200
    assert image.headers["cache-control"] == "no-store"
    assert image.headers["x-content-type-options"] == "nosniff"
    assert image.headers["content-type"] == "image/png"

    unconfirmed = workflow_client.post(
        "/api/v1/projects/current/pick",
        json={
            "x": 8,
            "y": 4,
            "camera_revision": descriptor["camera_revision"],
            "pick_buffer_revision": descriptor["pick_buffer_revision"],
        },
        headers=auth_headers,
    )
    assert unconfirmed.status_code == 409
    assert unconfirmed.json()["code"] == "camera_not_confirmed"

    confirmed = workflow_client.post(
        "/api/v1/projects/current/camera/confirm",
        json={"camera_revision": descriptor["camera_revision"]},
        headers=auth_headers,
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["workflow"]["confirmed_camera_revision"] == 1

    stale = workflow_client.post(
        "/api/v1/projects/current/pick",
        json={
            "x": 8,
            "y": 4,
            "camera_revision": descriptor["camera_revision"],
            "pick_buffer_revision": descriptor["pick_buffer_revision"] - 1,
        },
        headers=auth_headers,
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_pick_buffer"

    picked = workflow_client.post(
        "/api/v1/projects/current/pick",
        json={
            "x": 8,
            "y": 4,
            "camera_revision": descriptor["camera_revision"],
            "pick_buffer_revision": descriptor["pick_buffer_revision"],
        },
        headers=auth_headers,
    )
    assert picked.status_code == 200
    assert picked.json()["camera_revision"] == 1
    assert picked.json()["pick_buffer_revision"] == 1
    assert len(picked.json()["world"]) == 3
    project = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()
    assert project["workflow"]["foot_point"] == picked.json()


def test_stale_preview_generation_cannot_replace_newer_snapshot(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    request = {
        "generation": 2,
        "width": 16,
        "height": 9,
        "camera": {
            "target": [0.0, 0.0, 0.0],
            "distance": 4.0,
            "yaw": 10.0,
            "pitch": 0.0,
            "fov_y_degrees": 50.0,
        },
    }
    assert workflow_client.post(
        "/api/v1/projects/current/preview", json=request, headers=auth_headers
    ).status_code == 201
    request["generation"] = 1

    stale = workflow_client.post(
        "/api/v1/projects/current/preview", json=request, headers=auth_headers
    )

    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_preview_generation"


def test_started_task_id_is_recoverable_from_project_snapshot(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = workflow_client.post(
        "/api/v1/tasks", json={"target_stage": "segment"}, headers=auth_headers
    )
    project = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()

    assert created.status_code == 202
    assert project["workflow"]["active_task_id"] == created.json()["id"]


def test_export_result_is_ffprobe_verified_persisted_and_opaque(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    verified = workflow_client.get(
        "/api/v1/projects/current/export", headers=auth_headers
    )

    assert verified.status_code == 200
    result = verified.json()
    assert result["verified"] is True
    assert result["frame_count"] == 300
    assert result["duration_seconds"] == 10.0
    assert "path" not in result

    artifact = workflow_client.get(
        f"/api/v1/projects/current/exports/{result['artifact_id']}",
        headers=auth_headers,
    )
    assert artifact.status_code == 200
    assert artifact.content == b"verified-video"
    assert artifact.headers["cache-control"] == "no-store"
    persisted = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()
    assert persisted["workflow"]["export_result"]["artifact_id"] == result[
        "artifact_id"
    ]
