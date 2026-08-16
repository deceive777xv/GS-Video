from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import shutil
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
from pydantic import SecretStr

from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.app import create_app
from gs_video.camera.classify import CameraKind
from gs_video.camera.opencv_solver import CameraSolution
from gs_video.camera.serialization import read_mapped_trajectory
from gs_video.domain.contracts import (
    MaskSequence,
    PickBuffer,
    Prompt,
    RenderSequence,
    SegmentationBackend,
)
from gs_video.domain.models import (
    ArtifactRole,
    Project,
    SceneSummary,
    StageName,
    StageStatus,
    VideoSummary,
)
from gs_video.environment.doctor import EnvironmentReport
from gs_video.media.export import ExportResult
from gs_video.media.ffmpeg import VideoMetadata
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter
from gs_video.pipeline.services import (
    CameraSolveWorkflowService,
    CompositeWorkflowService,
    ExportWorkflowService,
    MediaIngestService,
    RendererWorkflowService,
    SegmentWorkflowService,
    TrajectoryMapWorkflowService,
    WorkflowPaths,
)
from gs_video.pipeline.workflow import WorkflowServices, build_mvp_workflow
from gs_video.project.repository import ProjectRepository
from gs_video.scene.worker_client import RendererWorkerIdentity
from gs_video.scene.worker_protocol import RenderSequenceRequest


TOKEN = "real-workflow-session-token"
ORIGIN = "http://127.0.0.1:5173"


def _write_rgb(path: Path, size: tuple[int, int], value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


class FakeAssetInspector:
    def inspect(
        self,
        kind: str,
        path: Path,
        *,
        size: int,
        sha256: str,
    ) -> VideoSummary | SceneSummary:
        if kind == "source_video":
            return VideoSummary(
                filename=path.name,
                size=size,
                sha256=sha256,
                width=8,
                height=6,
                duration_seconds=2 / 24,
                fps="24/1",
                has_audio=True,
                frame_count=2,
            )
        return SceneSummary(
            filename=path.name,
            size=size,
            sha256=sha256,
            gaussian_count=1,
            estimated_vram_mb=1,
        )


class FakeMediaBackend:
    identity = "fake-media-v1"

    def extract_source_frames(self, source: Path, output_dir: Path) -> list[Path]:
        assert source.is_file()
        frames: list[Path] = []
        for index, value in enumerate((40, 80), start=1):
            frame = output_dir / f"{index:06d}.png"
            _write_rgb(frame, (8, 6), value)
            frames.append(frame)
        return frames

    def extract_proxy_frames(
        self,
        source: Path,
        output_dir: Path,
        max_height: int,
    ) -> list[Path]:
        assert source.is_file()
        assert max_height > 0
        frames: list[Path] = []
        for index, value in enumerate((40, 80), start=1):
            frame = output_dir / f"{index:06d}.jpg"
            _write_rgb(frame, (4, 3), value)
            frames.append(frame)
        return frames


class FakeSegmenter:
    backend = SegmentationBackend.EDGETAM
    worker_prefix = ("deterministic-fake-segmenter",)

    def __init__(self, root: Path) -> None:
        self.model_config = root / "models" / "segment.yaml"
        self.checkpoint = root / "models" / "segment.pt"
        self.model_config.parent.mkdir(parents=True)
        self.model_config.write_text("model: fake", encoding="utf-8")
        self.checkpoint.write_bytes(b"fake-checkpoint")

    def segment(
        self,
        frames: list[Path],
        prompt: Prompt,
        output_dir: Path,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> MaskSequence:
        assert prompt.frame_index == 0
        output_dir.mkdir()
        for index, frame in enumerate(frames, start=1):
            token.raise_if_cancelled()
            with Image.open(frame) as image:
                Image.new("L", image.size, 255).save(
                    output_dir / f"{index:06d}.png"
                )
            emit(index, len(frames), f"fake segment {index}/{len(frames)}")
        return MaskSequence(output_dir, len(frames))


class FakeCameraSolver:
    def solve(
        self,
        frame_paths: list[Path] | tuple[Path, ...],
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> CameraSolution:
        poses: list[np.ndarray] = []
        for index, _path in enumerate(frame_paths, start=1):
            token.raise_if_cancelled()
            pose = np.eye(4)
            pose[0, 3] = index - 1
            poses.append(pose)
            emit(index, len(frame_paths), f"fake solve {index}/{len(frame_paths)}")
        return CameraSolution(
            np.array([[3.0, 0.0, 2.0], [0.0, 3.0, 1.5], [0.0, 0.0, 1.0]]),
            poses,
            CameraKind.SIX_DOF,
            0.9,
        )


class CpuFakeRendererWorker:
    def __init__(self) -> None:
        self.render_requests: list[RenderSequenceRequest] = []

    def probe(
        self, *, token: CancellationToken | None = None
    ) -> RendererWorkerIdentity:
        if token is not None:
            token.raise_if_cancelled()
        return RendererWorkerIdentity(
            torch="cpu-fake",
            gsplat="cpu-fake-1",
            device="cpu",
        )

    def render_sequence(
        self,
        request: RenderSequenceRequest,
        emit: ProgressEmitter,
        token: CancellationToken,
    ) -> RenderSequence:
        self.render_requests.append(request)
        request.output_dir.mkdir()
        trajectory = read_mapped_trajectory(request.camera_manifest)
        source_indices = tuple(
            range(0, len(trajectory.camera_to_world), request.preview_stride)
        )
        frames: list[Path] = []
        for current, _source_index in enumerate(source_indices, start=1):
            token.raise_if_cancelled()
            frame = request.output_dir / f"{current:06d}.png"
            _write_rgb(frame, (request.width, request.height), 200)
            frames.append(frame)
            emit(
                current,
                len(source_indices),
                f"cpu fake render {current}/{len(source_indices)}",
            )
        return RenderSequence(
            frame_dir=request.output_dir,
            frame_paths=tuple(frames),
            source_frame_indices=source_indices,
            width=request.width,
            height=request.height,
            implementation_version="gsplat-cpu-fake-1",
        )


class FakeExporter:
    def __call__(
        self,
        frames_dir: Path,
        source_video: Path,
        fps: Fraction,
        frame_count: int,
        output: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult:
        if cancellation_check is not None:
            cancellation_check()
        assert source_video.is_file()
        assert len(tuple(frames_dir.glob("*.png"))) == frame_count == 2
        shutil.copyfile(source_video, output)
        return ExportResult(
            output=output,
            fps=fps,
            frame_count=frame_count,
            duration=Fraction(frame_count, 1) / fps,
            has_audio=True,
        )


class FakeVideoProbe:
    def __call__(
        self,
        path: Path,
        *,
        cancellation_check: Callable[[], None] | None = None,
    ) -> ExportResult:
        if cancellation_check is not None:
            cancellation_check()
        return ExportResult(
            output=path,
            fps=Fraction(24, 1),
            frame_count=2,
            duration=Fraction(1, 12),
            has_audio=True,
        )

    def probe(self, path: Path) -> VideoMetadata:
        with path.open("rb") as stream:
            stream.seek(4)
            assert stream.read(4) == b"ftyp"
        return VideoMetadata(
            width=8,
            height=6,
            duration=1 / 12,
            fps="24/1",
            has_audio=True,
            frame_count=2,
        )


class FakePreviewService:
    def render_pick(
        self,
        project_root: Path,
        scene_path: str,
        scene_summary: SceneSummary,
        camera: object,
        width: int,
        height: int,
        *,
        preview_root: Path | None = None,
    ) -> PickBuffer:
        assert preview_root is not None
        assert (project_root / scene_path).is_file()
        assert scene_summary.gaussian_count == 1
        assert camera is not None
        return PickBuffer(
            rgb=np.full((height, width, 3), 96, dtype=np.uint8),
            expected_depth=np.full((height, width), 2.0, dtype=np.float32),
        )


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=0, issues=[])


class FakeWorkerRegistry:
    async def terminate_all(self) -> None:
        return None


def _wait_for_task(
    client: TestClient,
    task_id: str,
    headers: dict[str, str],
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while True:
        response = client.get(f"/api/v1/tasks/{task_id}", headers=headers)
        assert response.status_code == 200
        snapshot = response.json()
        if snapshot["status"] not in {"queued", "running"}:
            return snapshot
        assert time.monotonic() < deadline
        time.sleep(0.001)


@dataclass(frozen=True)
class ProductionHarness:
    root: Path
    source_fixture: Path
    scene_fixture: Path

    def run_with_fake_workers(self) -> Project:
        repository = ProjectRepository(self.root / "project")
        project = repository.create("deterministic production workflow")
        repository.save(project)
        paths = WorkflowPaths(repository.root, update_project=repository.update)
        probe = FakeVideoProbe()
        workflow_services = WorkflowServices(
            media_ingest=MediaIngestService(paths, FakeMediaBackend()),
            segmenter=SegmentWorkflowService(paths, FakeSegmenter(repository.root)),
            camera_solver=CameraSolveWorkflowService(
                paths,
                FakeCameraSolver(),
                backend_identity="fake-camera-solver-v1",
            ),
            trajectory_mapper=TrajectoryMapWorkflowService(paths),
            renderer=RendererWorkflowService(
                paths,
                CpuFakeRendererWorker(),
            ),
            compositor=CompositeWorkflowService(
                paths,
                exporter=FakeExporter(),
                exporter_identity="fake-composite-exporter-v1",
                prober=probe,
            ),
            exporter=ExportWorkflowService(
                paths,
                exporter=FakeExporter(),
                exporter_identity="fake-final-exporter-v1",
                prober=probe,
            ),
        )
        runner = build_mvp_workflow(
            workflow_services,
            project,
            save=repository.save,
            persist_stage=repository.update_stage,
            compare_and_set_stage=repository.compare_and_set_stage,
            claim_stage=repository.claim_stage,
        )
        services = ApiServices(
            project_repository=repository,
            environment_doctor=StaticDoctor(),
            pipeline_runner=runner,
            worker_registry=FakeWorkerRegistry(),
            preview_service=FakePreviewService(),
            asset_inspector=FakeAssetInspector(),
            export_inspector=probe,
        )
        settings = ApiSettings(
            bind_host="127.0.0.1",
            port=0,
            session_token=SecretStr(TOKEN),
            allowed_origins=("http://tauri.localhost", ORIGIN),
            task_workers=1,
        )
        headers = {"Authorization": f"Bearer {TOKEN}"}

        with TestClient(create_app(settings, services)) as client:
            for kind, path in (
                ("source_video", self.source_fixture),
                ("scene_ply", self.scene_fixture),
            ):
                imported = client.post(
                    "/api/v1/assets/import",
                    json={"kind": kind, "path": str(path)},
                    headers={**headers, "Origin": "http://tauri.localhost"},
                )
                assert imported.status_code == 201

            ingest = client.post(
                "/api/v1/tasks",
                json={"target_stage": "ingest"},
                headers=headers,
            )
            assert ingest.status_code == 202
            assert _wait_for_task(client, ingest.json()["id"], headers)[
                "status"
            ] == "succeeded"

            proxy = client.get(
                "/api/v1/projects/current/subject-media/proxy",
                headers=headers,
            )
            assert proxy.status_code == 200, proxy.json()

            prompt = client.patch(
                "/api/v1/projects/current",
                json={"subject_prompt": {"frame_index": 0, "x": 1, "y": 1}},
                headers=headers,
            )
            assert prompt.status_code == 200, prompt.json()

            segmented = client.post(
                "/api/v1/tasks",
                json={"target_stage": "segment"},
                headers=headers,
            )
            assert segmented.status_code == 202
            assert _wait_for_task(client, segmented.json()["id"], headers)[
                "status"
            ] == "succeeded"
            solved_source = client.post(
                "/api/v1/tasks",
                json={"target_stage": "solve_camera"},
                headers=headers,
            )
            assert solved_source.status_code == 202
            assert _wait_for_task(client, solved_source.json()["id"], headers)[
                "status"
            ] == "succeeded"
            authoritative = client.get(
                "/api/v1/projects/current", headers=headers
            ).json()
            calibrated = client.put(
                "/api/v1/projects/current/source-perspective",
                json={
                    "expected_project_id": project.project_id,
                    "expected_segment_cache_key": authoritative["stages"]["segment"][
                        "cache_key"
                    ],
                    "anchor_frame_index": 0,
                    "image_width": 4,
                    "image_height": 3,
                    "vertical_fov": 60.0,
                    "horizon_start": [0.0, 1.5],
                    "horizon_end": [4.0, 1.5],
                    "vertical_bottom": [2.0, 2.0],
                    "vertical_top": [2.0, 0.0],
                },
                headers=headers,
            )
            assert calibrated.status_code == 200, calibrated.text
            calibration = calibrated.json()["workflow"][
                "source_perspective_calibration"
            ]

            preview = client.post(
                "/api/v1/projects/current/preview",
                json={
                    "expected_project_id": project.project_id,
                    "generation": 1,
                    "width": 16,
                    "height": 9,
                    "camera": {
                        "camera_to_world": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, -2.0],
                            [0.0, 0.0, 1.0, -5.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "fov_y_degrees": 60.0,
                    },
                },
                headers=headers,
            )
            assert preview.status_code == 201, preview.json()
            preview_descriptor = preview.json()
            confirmed = client.post(
                "/api/v1/projects/current/camera/confirm",
                json={
                    "expected_project_id": project.project_id,
                    "camera_revision": preview_descriptor["camera_revision"],
                },
                headers=headers,
            )
            assert confirmed.status_code == 200
            anchored = client.put(
                "/api/v1/projects/current/local-ground",
                json={
                    "expected_project_id": project.project_id,
                    "preview_artifact_id": preview_descriptor["artifact_id"],
                    "camera_revision": preview_descriptor["camera_revision"],
                    "pick_buffer_revision": preview_descriptor[
                        "pick_buffer_revision"
                    ],
                    "points": [[4, 7], [11, 7], [8, 5]],
                    "flip_normal": False,
                },
                headers=headers,
            )
            assert anchored.status_code == 200, anchored.text
            ground = anchored.json()["workflow"]["local_ground_anchor"]
            placed = client.put(
                "/api/v1/projects/current/synthesis-placement",
                json={
                    "expected_project_id": project.project_id,
                    "source_calibration_revision": calibration["revision"],
                    "ground_anchor_revision": ground["revision"],
                    "mode": "perspective",
                    "scene_azimuth": 0.0,
                    "subject_to_scene_scale": 1.0,
                    "composition_offset_local": [0.0, 0.0],
                    "foot_pixel": None,
                },
                headers=headers,
            )
            assert placed.status_code == 200, placed.text
            placement = placed.json()["workflow"]["synthesis_placement"]
            placement_confirmed = client.post(
                "/api/v1/projects/current/synthesis-placement/confirm",
                json={
                    "expected_project_id": project.project_id,
                    "placement_revision": placement["revision"],
                },
                headers=headers,
            )
            assert placement_confirmed.status_code == 200

            exported = client.post(
                "/api/v1/tasks",
                json={"target_stage": "export"},
                headers=headers,
            )
            assert exported.status_code == 202
            assert _wait_for_task(client, exported.json()["id"], headers)[
                "status"
            ] == "succeeded"

            composite = client.get(
                "/api/v1/projects/current/composite-preview",
                headers=headers,
            )
            assert composite.status_code == 200
            composite_video = client.get(
                "/api/v1/artifacts/composite-previews/"
                f"{composite.json()['artifact_id']}",
                headers=headers,
            )
            assert composite_video.status_code == 200
            assert composite_video.content[4:8] == b"ftyp"

            verified = client.get(
                "/api/v1/projects/current/export",
                headers=headers,
            )
            assert verified.status_code == 200
            export_video = client.get(
                "/api/v1/projects/current/exports/"
                f"{verified.json()['artifact_id']}",
                headers=headers,
            )
            assert export_video.status_code == 200
            assert export_video.content[4:8] == b"ftyp"

        return repository.load()


@pytest.fixture
def production_harness(tmp_path: Path) -> ProductionHarness:
    fixtures = Path(__file__).parents[2] / "fixtures"
    return ProductionHarness(
        root=tmp_path,
        source_fixture=fixtures / "media" / "source.mp4",
        scene_fixture=fixtures / "scene" / "tiny_gaussians.ply",
    )


def test_assembled_workflow_produces_preview_and_verified_export(
    production_harness: ProductionHarness,
) -> None:
    project = production_harness.run_with_fake_workers()

    assert project.stages[StageName.EXPORT].status is StageStatus.SUCCEEDED
    assert project.stages[StageName.COMPOSITE].artifacts[
        ArtifactRole.COMPOSITE_PREVIEW
    ].member == "composite-preview.mp4"
    assert project.workflow.export_result is not None
    assert project.workflow.export_result.verified is True
