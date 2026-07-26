import asyncio
from collections.abc import Iterator
import hashlib
from pathlib import Path
from threading import Event
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from PIL import Image

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.domain.contracts import PickBuffer
from gs_video.domain.models import (
    ArtifactRole,
    CameraPose,
    ExportResultState,
    FootPointState,
    PreviewState,
    StageName,
    StageState,
    StageStatus,
    SubjectPromptState,
    SceneSummary,
)
from gs_video.environment.doctor import EnvironmentReport
from gs_video.media.ffmpeg import VideoMetadata
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.repository import ProjectRepository
from gs_video.scene.camera import OrbitCamera


TOKEN = "workflow-session-token"
ORIGIN = "http://127.0.0.1:5173"
INGEST_CACHE_KEY = "1" * 64
SEGMENT_CACHE_KEY = "2" * 64


class StaticDoctor:
    def check(self) -> EnvironmentReport:
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
        scene_summary: SceneSummary,
        camera: OrbitCamera,
        width: int,
        height: int,
    ) -> PickBuffer:
        del project_root, scene_path, scene_summary
        self.cameras.append(camera)
        rgb = np.full((height, width, 3), 96, dtype=np.uint8)
        depth = np.full((height, width), 2.0, dtype=np.float32)
        return PickBuffer(rgb=rgb, expected_depth=depth)


class ExportInspector:
    def probe(self, path: Path) -> VideoMetadata:
        assert path.read_bytes() in {b"verified-video", b"composite-preview"}
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
    (repository.root / "source" / "scene.ply").write_bytes(b"scene-one")
    project.workflow.scene_summary = SceneSummary(
        filename="scene.ply",
        size=9,
        sha256="1" * 64,
        gaussian_count=1,
        estimated_vram_mb=1,
    )
    (repository.root / "exports" / "final.mp4").write_bytes(b"verified-video")
    composite_cache_key = "composite-key"
    preview_path = (
        repository.root
        / "previews"
        / composite_cache_key
        / "composite-preview.mp4"
    )
    preview_path.parent.mkdir()
    preview_path.write_bytes(b"composite-preview")
    proxy_root = repository.root / "proxies" / INGEST_CACHE_KEY
    mask_root = repository.root / "masks" / SEGMENT_CACHE_KEY
    proxy_root.mkdir()
    mask_root.mkdir()
    for index in range(1, 6):
        Image.new("RGB", (16, 9), (20, 40, 60)).save(
            proxy_root / f"{index:06d}.jpg"
        )
        Image.new("L", (16, 9), 255).save(
            mask_root / f"{index:06d}.png"
        )
    project.workflow.subject_prompt = SubjectPromptState(
        frame_index=4, x=8, y=4
    )
    project.stages[StageName.INGEST] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=INGEST_CACHE_KEY,
        artifacts={ArtifactRole.PROXY_FRAMES: f"proxies/{INGEST_CACHE_KEY}"},
    )
    project.stages[StageName.SEGMENT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=SEGMENT_CACHE_KEY,
        artifacts={ArtifactRole.SUBJECT_MASKS: f"masks/{SEGMENT_CACHE_KEY}"},
    )
    project.stages[StageName.EXPORT] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key="export-key",
        output_paths=["exports/final.mp4"],
        artifacts={ArtifactRole.EXPORT_VIDEO: "exports/final.mp4"},
    )
    project.stages[StageName.COMPOSITE] = StageState(
        status=StageStatus.SUCCEEDED,
        cache_key=composite_cache_key,
        output_paths=[f"previews/{composite_cache_key}/composite-preview.mp4"],
        artifacts={
            ArtifactRole.COMPOSITE_PREVIEW: (
                f"previews/{composite_cache_key}/composite-preview.mp4"
            )
        },
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
            "subject_prompt": {"frame_index": 4, "x": 10, "y": 5},
            "motion_scale": 0.75,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workflow"]["subject_prompt"] == {
        "frame_index": 4,
        "x": 10,
        "y": 5,
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
            "preview_artifact_id": descriptor["artifact_id"],
            "camera_revision": descriptor["camera_revision"],
            "pick_buffer_revision": descriptor["pick_buffer_revision"],
        },
        headers=auth_headers,
    )
    assert unconfirmed.status_code == 409
    assert unconfirmed.json()["code"] == "camera_not_confirmed"

    second = workflow_client.post(
        "/api/v1/projects/current/preview",
        json={
            "generation": 2,
            "width": 16,
            "height": 9,
            "camera": {
                "target": [0.0, 0.0, 0.0],
                "distance": 4.0,
                "yaw": 5.0,
                "pitch": 0.0,
                "fov_y_degrees": 60.0,
            },
        },
        headers=auth_headers,
    ).json()

    confirmed = workflow_client.post(
        "/api/v1/projects/current/camera/confirm",
        json={"camera_revision": second["camera_revision"]},
        headers=auth_headers,
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["workflow"]["confirmed_camera_revision"] == 2

    stale = workflow_client.post(
        "/api/v1/projects/current/pick",
        json={
            "x": 8,
            "y": 4,
            "preview_artifact_id": descriptor["artifact_id"],
            "camera_revision": descriptor["camera_revision"],
            "pick_buffer_revision": descriptor["pick_buffer_revision"],
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
            "preview_artifact_id": second["artifact_id"],
            "camera_revision": second["camera_revision"],
            "pick_buffer_revision": second["pick_buffer_revision"],
        },
        headers=auth_headers,
    )
    assert picked.status_code == 200
    assert picked.json()["camera_revision"] == 2
    assert picked.json()["pick_buffer_revision"] == 2
    assert len(picked.json()["world"]) == 3
    project = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()
    assert project["workflow"]["foot_point"] == picked.json()


def test_pick_rejects_preview_artifact_aba_before_transaction_commit(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {
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
    }
    old_preview = workflow_client.post(
        "/api/v1/projects/current/preview", json=request, headers=auth_headers
    ).json()
    workflow_client.post(
        "/api/v1/projects/current/camera/confirm",
        json={"camera_revision": old_preview["camera_revision"]},
        headers=auth_headers,
    )
    repository = workflow_client.app.state.services.project_repository

    from gs_video.api import routes as workflow_routes

    original_unproject = workflow_routes._unproject

    def replace_authority_after_unproject(
        *args: Any, **kwargs: Any
    ) -> tuple[float, float, float]:
        world = original_unproject(*args, **kwargs)

        def replace(latest: Any) -> None:
            latest.scene_ply = "source/scene-b.ply"
            latest.workflow.scene_summary = SceneSummary(
                filename="scene-b.ply",
                size=7,
                sha256="b" * 64,
                gaussian_count=1,
                estimated_vram_mb=1,
            )
            latest.workflow.preview = PreviewState(
                artifact_id="preview-b",
                artifact_size=1,
                artifact_sha256="b" * 64,
                generation=1,
                width=16,
                height=9,
                camera_revision=1,
                pick_buffer_revision=1,
            )
            latest.workflow.confirmed_camera_revision = 1
            latest.workflow.confirmed_preview_artifact_id = "preview-b"

        repository.update(replace)
        return world

    monkeypatch.setattr(
        workflow_routes, "_unproject", replace_authority_after_unproject
    )

    stale = workflow_client.post(
        "/api/v1/projects/current/pick",
        json={
            "x": 8,
            "y": 4,
            "preview_artifact_id": old_preview["artifact_id"],
            "camera_revision": 1,
            "pick_buffer_revision": 1,
        },
        headers=auth_headers,
    )

    assert stale.status_code == 409
    assert stale.json()["code"] == "stale_pick_buffer"
    assert repository.load().workflow.foot_point is None


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


def test_scene_replacement_discards_preview_rendered_from_old_scene(
    tmp_path: Path, auth_headers: dict[str, str]
) -> None:
    class BlockingPreviewService(PreviewService):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def render_pick(
            self,
            project_root: Path,
            scene_path: str,
            scene_summary: SceneSummary,
            camera: OrbitCamera,
            width: int,
            height: int,
        ) -> PickBuffer:
            self.started.set()
            assert self.release.wait(2)
            return super().render_pick(
                project_root, scene_path, scene_summary, camera, width, height
            )

    repository = ProjectRepository(tmp_path / "scene-race")
    project = repository.create("scene-race")
    project.scene_ply = "source/old.ply"
    (repository.root / "source" / "old.ply").write_bytes(b"old")
    project.workflow.scene_summary = SceneSummary(
        filename="old.ply", size=3, sha256="a" * 64,
        gaussian_count=1, estimated_vram_mb=1
    )
    repository.save(project)
    service = BlockingPreviewService()
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=Registry(),
        preview_service=service,
    )
    settings = ApiSettings(
        bind_host="127.0.0.1", port=0, session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    request = {
        "generation": 1, "width": 16, "height": 9,
        "camera": {"target": [0.0, 0.0, 0.0], "distance": 4.0,
                   "yaw": 0.0, "pitch": 0.0, "fov_y_degrees": 60.0},
    }

    with TestClient(create_app(settings, services)) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                client.post, "/api/v1/projects/current/preview",
                json=request, headers=auth_headers
            )
            assert service.started.wait(1)

            def replace_scene(latest: object) -> None:
                latest.scene_ply = "source/new.ply"  # type: ignore[attr-defined]
                latest.workflow.scene_summary = SceneSummary(  # type: ignore[attr-defined]
                    filename="new.ply", size=3, sha256="b" * 64,
                    gaussian_count=1, estimated_vram_mb=1
                )

            (repository.root / "source" / "new.ply").write_bytes(b"new")
            repository.update(replace_scene)
            service.release.set()
            response = pending.result(timeout=2)

    assert response.status_code == 409
    assert response.json()["code"] == "scene_changed"
    assert repository.load().workflow.preview is None


def test_cancelled_preview_cannot_share_its_buffer_with_conflicting_request(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    class BlockingPreviewService(PreviewService):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def render_pick(
            self,
            project_root: Path,
            scene_path: str,
            scene_summary: SceneSummary,
            camera: OrbitCamera,
            width: int,
            height: int,
        ) -> PickBuffer:
            self.started.set()
            assert self.release.wait(2)
            return super().render_pick(
                project_root, scene_path, scene_summary, camera, width, height
            )

    service = BlockingPreviewService()
    workflow_client.app.state.preview_service = service
    repository = workflow_client.app.state.services.project_repository
    first_request = {
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
    }
    conflicting_request = {
        **first_request,
        "camera": {**first_request["camera"], "yaw": 15.0},
    }

    async def exercise() -> tuple[bool, int, str]:
        transport = ASGITransport(app=workflow_client.app)
        async with AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            first = asyncio.create_task(
                client.post(
                    "/api/v1/projects/current/preview",
                    json=first_request,
                    headers=auth_headers,
                )
            )
            assert await asyncio.to_thread(service.started.wait, 1)
            first.cancel()
            await asyncio.sleep(0)
            conflicting = asyncio.create_task(
                client.post(
                    "/api/v1/projects/current/preview",
                    json=conflicting_request,
                    headers=auth_headers,
                )
            )
            await asyncio.sleep(0.05)
            rejected_before_release = conflicting.done()
            service.release.set()
            outcomes = await asyncio.gather(
                first, conflicting, return_exceptions=True
            )

        assert isinstance(outcomes[0], asyncio.CancelledError)
        response = outcomes[1]
        assert not isinstance(response, BaseException)
        return rejected_before_release, response.status_code, response.json()["code"]

    rejected_before_release, status_code, code = asyncio.run(exercise())

    assert rejected_before_release
    assert status_code == 409
    assert code == "preview_generation_conflict"
    assert len(service.cameras) == 1
    assert service.cameras[0].yaw == 0.0
    assert repository.load().workflow.preview is None
    assert not tuple((repository.root / "previews").glob("*.png"))


def test_subject_prompt_requires_authoritative_proxy_bounds(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    before = workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()["workflow"]["subject_prompt"]

    invalid_frame = workflow_client.patch(
        "/api/v1/projects/current",
        json={"subject_prompt": {"frame_index": 5, "x": 0, "y": 0}},
        headers=auth_headers,
    )
    invalid_point = workflow_client.patch(
        "/api/v1/projects/current",
        json={"subject_prompt": {"frame_index": 4, "x": 16, "y": 9}},
        headers=auth_headers,
    )

    assert invalid_frame.status_code == 422
    assert invalid_frame.json()["code"] == "invalid_subject_prompt"
    assert invalid_point.status_code == 422
    assert invalid_point.json()["code"] == "invalid_subject_prompt"
    assert workflow_client.get(
        "/api/v1/projects/current", headers=auth_headers
    ).json()["workflow"]["subject_prompt"] == before


def test_subject_prompt_can_be_cleared_without_proxy_validation(
    workflow_client: TestClient, auth_headers: dict[str, str], tmp_path: Path
) -> None:
    for proxy in (tmp_path / "project" / "proxies" / INGEST_CACHE_KEY).glob("*.jpg"):
        proxy.unlink()

    response = workflow_client.patch(
        "/api/v1/projects/current",
        json={"subject_prompt": None},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["workflow"]["subject_prompt"] is None


@pytest.mark.parametrize("replacement_path", ["local", "upload"])
def test_replacing_source_video_clears_all_source_derived_authority(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
    replacement_path: str,
) -> None:
    repository = workflow_client.app.state.services.project_repository
    initial_preview_epoch = repository.load().workflow.preview_epoch

    def seed_authority(project: object) -> None:
        workflow = project.workflow  # type: ignore[attr-defined]
        workflow.target_camera = CameraPose(
            target=(0.0, 0.0, 0.0), distance=4.0, yaw=0.0,
            pitch=0.0, fov_y_degrees=60.0, revision=3
        )
        workflow.confirmed_camera_revision = 3
        workflow.confirmed_preview_artifact_id = "f" * 32
        workflow.foot_point = FootPointState(
            image=(8, 4), world=(0.0, 0.0, 2.0),
            preview_artifact_id="f" * 32,
            camera_revision=3, pick_buffer_revision=3
        )
        workflow.preview = PreviewState(
            artifact_id="f" * 32, artifact_size=10,
            artifact_sha256="e" * 64, generation=3, width=16, height=9,
            camera_revision=3, pick_buffer_revision=3
        )
        workflow.export_result = ExportResultState(
            artifact_id="d" * 32, filename="final.mp4", size=10,
            sha256="c" * 64, duration_seconds=1.0, fps="30",
            frame_count=30, has_audio=False
        )
        workflow.active_task_id = "obsolete-task"

    repository.update(seed_authority)
    payload = b"replacement-video"
    if replacement_path == "local":
        selected = tmp_path / "replacement.mp4"
        selected.write_bytes(payload)
        response = workflow_client.post(
            "/api/v1/assets/import",
            json={"path": str(selected), "kind": "source_video"},
            headers=auth_headers,
        )
    else:
        created = workflow_client.post(
            "/api/v1/uploads",
            json={
                "kind": "source_video", "filename": "replacement.mp4",
                "mime_type": "video/mp4", "total_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            headers=auth_headers,
        ).json()
        assert workflow_client.put(
            f"/api/v1/uploads/{created['id']}/chunks/0",
            content=payload, headers=auth_headers
        ).status_code == 204
        response = workflow_client.post(
            f"/api/v1/uploads/{created['id']}/complete", headers=auth_headers
        )

    assert response.status_code == 201
    project = repository.load()
    assert project.workflow.subject_prompt is None
    assert project.workflow.confirmed_camera_revision is None
    assert project.workflow.confirmed_preview_artifact_id is None
    assert project.workflow.foot_point is None
    assert project.workflow.preview_epoch == initial_preview_epoch + 1
    assert project.workflow.preview is None
    assert project.workflow.export_result is None
    assert project.workflow.active_task_id is None
    for name in (StageName.INGEST, StageName.SEGMENT, StageName.EXPORT):
        assert project.stages[name].status is StageStatus.STALE
        assert project.stages[name].input_generation == 1


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


def test_composite_preview_is_opaque_authenticated_and_cache_bound(
    workflow_client: TestClient, auth_headers: dict[str, str]
) -> None:
    descriptor = workflow_client.get(
        "/api/v1/projects/current/composite-preview", headers=auth_headers
    )

    assert descriptor.status_code == 200
    payload = descriptor.json()
    assert set(payload) == {
        "artifact_id",
        "filename",
        "size",
        "sha256",
        "duration_seconds",
        "fps",
        "frame_count",
    }
    assert "path" not in payload
    video = workflow_client.get(
        f"/api/v1/artifacts/composite-previews/{payload['artifact_id']}",
        headers=auth_headers,
    )

    assert video.status_code == 200
    assert video.content == b"composite-preview"
    assert video.headers["content-type"] == "video/mp4"


def test_verified_export_can_be_safely_copied_to_caller_destination(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    verified = workflow_client.get(
        "/api/v1/projects/current/export", headers=auth_headers
    ).json()
    destination = tmp_path / "chosen" / "copied.mp4"

    copied = workflow_client.post(
        f"/api/v1/projects/current/exports/{verified['artifact_id']}/copy",
        json={"destination": str(destination)},
        headers=auth_headers,
    )

    assert copied.status_code == 204
    assert copied.content == b""
    assert destination.read_bytes() == b"verified-video"

    stale = workflow_client.post(
        "/api/v1/projects/current/exports/not-current/copy",
        json={"destination": str(tmp_path / "stale.mp4")},
        headers=auth_headers,
    )
    assert stale.status_code == 404
    assert not (tmp_path / "stale.mp4").exists()


@pytest.mark.parametrize(
    "relative_destination",
    ["project.json", "source/clobber.mp4"],
)
def test_verified_export_copy_cannot_write_inside_project_root(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
    relative_destination: str,
) -> None:
    verified = workflow_client.get(
        "/api/v1/projects/current/export", headers=auth_headers
    ).json()
    project_root = tmp_path / "project"
    project_before = (project_root / "project.json").read_bytes()
    destination = project_root / relative_destination

    response = workflow_client.post(
        f"/api/v1/projects/current/exports/{verified['artifact_id']}/copy",
        json={"destination": str(destination)},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_export_destination"
    assert (project_root / "project.json").read_bytes() == project_before
    if relative_destination != "project.json":
        assert not destination.exists()


def test_verified_export_copy_requires_an_absolute_external_destination(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    verified = workflow_client.get(
        "/api/v1/projects/current/export", headers=auth_headers
    ).json()

    response = workflow_client.post(
        f"/api/v1/projects/current/exports/{verified['artifact_id']}/copy",
        json={"destination": "relative/result.mp4"},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_export_destination"


@pytest.mark.parametrize(
    "stage_mutation",
    [
        {"cache_key": None},
        {"cache_key": "replacement-export-key"},
        {"artifacts": {}},
        {"artifacts": {ArtifactRole.EXPORT_VIDEO: "source/video.mp4"}},
    ],
)
def test_verified_export_copy_requires_the_typed_authoritative_export(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
    stage_mutation: dict[str, object],
) -> None:
    verified = workflow_client.get(
        "/api/v1/projects/current/export", headers=auth_headers
    ).json()
    repository = workflow_client.app.state.services.project_repository

    def mutate(project: object) -> None:
        stage = project.stages[StageName.EXPORT]  # type: ignore[attr-defined]
        for name, value in stage_mutation.items():
            setattr(stage, name, value)

    repository.update(mutate)
    destination = tmp_path / "typed-authority.mp4"

    response = workflow_client.post(
        f"/api/v1/projects/current/exports/{verified['artifact_id']}/copy",
        json={"destination": str(destination)},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert not destination.exists()


@pytest.mark.parametrize(
    ("role", "content_type"),
    [("proxy", "image/jpeg"), ("alpha", "image/png")],
)
def test_subject_media_is_derived_from_typed_stage_artifacts(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    role: str,
    content_type: str,
) -> None:
    descriptor = workflow_client.get(
        f"/api/v1/projects/current/subject-media/{role}", headers=auth_headers
    )

    assert descriptor.status_code == 200
    body = descriptor.json()
    assert body["role"] == role
    assert body["frame_index"] == 4
    assert body["width"] == 16
    assert body["height"] == 9
    assert "path" not in body

    artifact = workflow_client.get(
        f"/api/v1/projects/current/subject-media/{role}/{body['artifact_id']}",
        headers=auth_headers,
    )
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == content_type
    assert artifact.headers["cache-control"] == "no-store"


def test_subject_proxy_defaults_to_zero_based_frame_zero_before_prompt(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    repository = workflow_client.app.state.services.project_repository
    repository.update(lambda project: setattr(project.workflow, "subject_prompt", None))

    descriptor = workflow_client.get(
        "/api/v1/projects/current/subject-media/proxy", headers=auth_headers
    )

    assert descriptor.status_code == 200
    assert descriptor.json()["frame_index"] == 0


def test_subject_alpha_requires_a_successful_prompt_bound_segment(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    repository = workflow_client.app.state.services.project_repository
    repository.update(lambda project: setattr(project.workflow, "subject_prompt", None))

    response = workflow_client.get(
        "/api/v1/projects/current/subject-media/alpha", headers=auth_headers
    )

    assert response.status_code == 409
    assert response.json()["code"] == "subject_media_not_ready"


@pytest.mark.parametrize(
    "stage_mutation",
    [
        {"cache_key": None},
        {"artifacts": {ArtifactRole.PROXY_FRAMES: f"./proxies/{INGEST_CACHE_KEY}"}},
    ],
)
def test_subject_proxy_requires_exact_typed_stage_authority(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    stage_mutation: dict[str, object],
) -> None:
    repository = workflow_client.app.state.services.project_repository

    def mutate(project: object) -> None:
        stage = project.stages[StageName.INGEST]  # type: ignore[attr-defined]
        for name, value in stage_mutation.items():
            setattr(stage, name, value)

    repository.update(mutate)

    response = workflow_client.get(
        "/api/v1/projects/current/subject-media/proxy", headers=auth_headers
    )

    assert response.status_code == 409


def test_subject_proxy_rejects_noncanonical_cache_keys_even_when_target_exists(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    repository = workflow_client.app.state.services.project_repository
    project_root = tmp_path / "project"
    canonical_root = project_root / "proxies" / INGEST_CACHE_KEY
    cases = (
        (str(canonical_root), canonical_root),
        (f"../proxies/{INGEST_CACHE_KEY}", canonical_root),
        ("abc", project_root / "proxies" / "abc"),
        ("A" * 64, project_root / "proxies" / ("A" * 64)),
    )

    for cache_key, target in cases:
        if target != canonical_root:
            target.mkdir(parents=True)
            for source in canonical_root.glob("*.jpg"):
                (target / source.name).write_bytes(source.read_bytes())

        def mutate(project: object) -> None:
            stage = project.stages[StageName.INGEST]  # type: ignore[attr-defined]
            stage.cache_key = cache_key
            stage.artifacts = {
                ArtifactRole.PROXY_FRAMES: f"proxies/{cache_key}"
            }

        repository.update(mutate)
        response = workflow_client.get(
            "/api/v1/projects/current/subject-media/proxy",
            headers=auth_headers,
        )
        assert response.status_code == 409, cache_key
        assert response.json()["code"] == "subject_media_contract_missing"


def test_subject_media_rejects_noncanonical_inventory_and_dimensions(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    proxy_root = project_root / "proxies" / INGEST_CACHE_KEY
    mask_root = project_root / "masks" / SEGMENT_CACHE_KEY
    (proxy_root / "extra.jpg").write_bytes(b"not a frame")

    invalid_inventory = workflow_client.get(
        "/api/v1/projects/current/subject-media/proxy", headers=auth_headers
    )
    assert invalid_inventory.status_code == 409
    assert invalid_inventory.json()["code"] == "subject_media_changed"

    (proxy_root / "extra.jpg").unlink()
    Image.new("L", (8, 8), 255).save(mask_root / "000005.png")
    invalid_dimensions = workflow_client.get(
        "/api/v1/projects/current/subject-media/alpha", headers=auth_headers
    )
    assert invalid_dimensions.status_code == 409
    assert invalid_dimensions.json()["code"] == "subject_media_invalid"


def test_subject_media_descriptor_becomes_stale_when_bytes_change(
    workflow_client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    descriptor = workflow_client.get(
        "/api/v1/projects/current/subject-media/proxy", headers=auth_headers
    ).json()
    Image.new("RGB", (16, 9), (200, 10, 30)).save(
        tmp_path / "project" / "proxies" / INGEST_CACHE_KEY / "000005.jpg"
    )

    stale = workflow_client.get(
        "/api/v1/projects/current/subject-media/proxy/"
        f"{descriptor['artifact_id']}",
        headers=auth_headers,
    )

    assert stale.status_code == 404
    assert stale.json()["code"] == "subject_media_unavailable"
