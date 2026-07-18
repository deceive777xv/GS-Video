# Production Workflow Assembly and Composite Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing pipeline protocols into a runnable offline MVP that produces full-resolution composite frames, a verified low-resolution preview MP4, and a verified final MP4 without loading renderer GPU dependencies into the packaged API sidecar.

**Architecture:** Add cache-key-scoped, atomically published stage artifacts and thin project-aware adapters around the already-tested media, segmentation, camera, renderer, compositor, and exporter primitives. Keep segmentation and gsplat rendering in separately configured worker Python environments; the FastAPI process validates authority and files but never imports model or CUDA runtimes. Register the composite preview as an opaque authenticated artifact, then let the shared React UI display it.

**Tech Stack:** Python 3.11, FastAPI, Pydantic 2, FFmpeg/ffprobe, OpenCV, NumPy, Pillow, existing segmentation process isolation, gsplat worker process, React 19, TypeScript, Vitest.

## Global Constraints

- Windows 11, NVIDIA GPU, and 8 GB VRAM remain the MVP target.
- GPU work is serialized and renders one frame at a time; realtime performance is not a release requirement.
- Processing is local and offline. Runtime code must not download models, test assets, or user media.
- Model checkpoints and worker environments are explicit project-local configuration; no framework default cache under the user profile is allowed.
- Original source video and Gaussian PLY files are immutable.
- Every published stage artifact is cache-key scoped, atomically published, confined to the project root, and protected from stale-generation overwrite.
- The packaged API sidecar does not bundle or import torch, gsplat, EdgeTAM, or SAM 2.
- Renderer and segmentation worker commands use argv arrays with `shell=False`, process-tree cleanup, cancellation, bounded diagnostics, and no token in argv, environment, stdout, or logs.
- Browser and Tauri feature code continues to use the versioned HTTP API through `BackendClient`.

---

### Task 1: Cache-scoped artifact authority and camera serialization

**Files:**
- Modify: `src/gs_video/domain/models.py`
- Modify: `src/gs_video/project/repository.py`
- Create: `src/gs_video/pipeline/artifacts.py`
- Create: `src/gs_video/camera/serialization.py`
- Create: `tests/unit/pipeline/test_artifacts.py`
- Create: `tests/unit/camera/test_serialization.py`
- Modify: `tests/integration/api/test_workflow.py`

**Interfaces:**
- Produces: artifact roles `SOURCE_FRAMES`, `CAMERA_SOLUTION`, `MAPPED_TRAJECTORY`, `RENDER_FRAMES`, `COMPOSITE_FRAMES`, and `COMPOSITE_PREVIEW`.
- Produces: `ArtifactPublisher(root).publish_tree(category, cache_key, build) -> Path`.
- Produces: `write_camera_solution(path, solution)`, `read_camera_solution(path)`, `write_mapped_trajectory(path, trajectory)`, and `read_mapped_trajectory(path)`.
- Consumes: existing `StageResult.artifacts`, `CameraSolution`, `CameraKind`, and `OrbitCamera`.

- [ ] **Step 1: Write failing artifact-publication and serialization tests**

```python
def test_artifact_publisher_keeps_old_generation_when_builder_fails(tmp_path: Path) -> None:
    publisher = ArtifactPublisher(tmp_path)
    published = publisher.publish_tree(
        "proxies", "a" * 64, lambda staging: (staging / "000001.jpg").write_bytes(b"old")
    )

    def fail(staging: Path) -> None:
        (staging / "000001.jpg").write_bytes(b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        publisher.publish_tree("proxies", "b" * 64, fail)

    assert (published / "000001.jpg").read_bytes() == b"old"
    assert not (tmp_path / "proxies" / ("b" * 64)).exists()


def test_camera_solution_json_round_trip_preserves_matrices(tmp_path: Path) -> None:
    solution = camera_solution_fixture()
    destination = tmp_path / "camera" / "solution.json"
    write_camera_solution(destination, solution)
    restored = read_camera_solution(destination)
    np.testing.assert_allclose(restored.intrinsics, solution.intrinsics)
    np.testing.assert_allclose(restored.camera_to_world, solution.camera_to_world)
    assert restored.kind is solution.kind
    assert restored.confidence == pytest.approx(solution.confidence)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:UV_CACHE_DIR='E:\Project\GS-Video\.cache\uv'
uv run --offline --frozen --extra dev pytest tests/unit/pipeline/test_artifacts.py tests/unit/camera/test_serialization.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-artifacts-red
```

Expected: collection fails because the publisher, serialization module, and new roles do not exist.

- [ ] **Step 3: Implement confined atomic publication and strict JSON**

`ArtifactPublisher` accepts only a fixed category name and a 64-character lowercase hex cache key. It creates a sibling `.staging-<uuid>` directory with exclusive ownership, calls the builder, fsyncs every regular file and the directory, rejects links/reparse points and non-regular members, then renames the staging directory to `<root>/<category>/<cache_key>`. Existing destinations are reusable only after the same inventory validation; publication never deletes another cache key.

Camera JSON contains only finite numeric lists and these exact keys:

```python
{
    "version": 1,
    "intrinsics": solution.intrinsics.tolist(),
    "camera_to_world": [pose.tolist() for pose in solution.camera_to_world],
    "kind": solution.kind.value,
    "confidence": solution.confidence,
    "diagnostics": dict(solution.diagnostics),
}
```

Mapped trajectory JSON contains:

```python
{
    "version": 1,
    "fov_y_degrees": fov_y_degrees,
    "camera_to_world": [pose.tolist() for pose in camera_to_world],
}
```

All reads use Pydantic models with `extra="forbid"`, finite values, rigid-transform validation, a maximum file size of 64 MiB, and held-handle pre/post identity checks.

- [ ] **Step 4: Generalize subject artifact resolution**

Change subject artifact registration from the canonical bare strings `proxies` and `masks` to cache-scoped relative paths such as `proxies/<cache_key>` and `masks/<cache_key>`. `resolve_subject_media` must require:

```python
registered == f"{definition.directory_name}/{stage.cache_key}"
```

and resolve that exact directory strictly beneath the canonical category root. Preserve the current consecutive filename, link, size, stable-read, hash, image mode, and dimension checks.

- [ ] **Step 5: Run focused GREEN and commit**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/pipeline/test_artifacts.py tests/unit/camera/test_serialization.py tests/integration/api/test_workflow.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-artifacts-green
uv run --offline --frozen --extra dev ruff check src/gs_video/pipeline/artifacts.py src/gs_video/camera/serialization.py src/gs_video/domain/models.py tests/unit/pipeline/test_artifacts.py tests/unit/camera/test_serialization.py tests/integration/api/test_workflow.py
uv run --offline --frozen --extra dev mypy --strict src/gs_video/pipeline/artifacts.py src/gs_video/camera/serialization.py
```

Expected: all focused tests, Ruff, and strict mypy pass.

Commit:

```powershell
git add src/gs_video/domain/models.py src/gs_video/project/repository.py src/gs_video/pipeline/artifacts.py src/gs_video/camera/serialization.py tests/unit/pipeline/test_artifacts.py tests/unit/camera/test_serialization.py tests/integration/api/test_workflow.py
git commit -m "feat: add cache-scoped workflow artifacts"
```

---

### Task 2: Full-resolution ingest and concrete CPU-stage adapters

**Files:**
- Modify: `src/gs_video/media/ingest.py`
- Create: `src/gs_video/pipeline/services.py`
- Create: `tests/unit/pipeline/test_services.py`
- Modify: `tests/integration/pipeline/test_workflow.py`

**Interfaces:**
- Produces: `extract_source_frames(source, output_dir) -> list[Path]`.
- Produces: `MediaIngestService`, `SegmentWorkflowService`, `CameraSolveWorkflowService`, `TrajectoryMapWorkflowService`, `CompositeWorkflowService`, and `ExportWorkflowService`.
- Consumes: `ArtifactPublisher`, `VideoSegmenterClient`, `OpenCvCameraSolver`, `map_trajectory`, `composite_frame`, and `export_mp4`.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_ingest_registers_proxy_and_full_resolution_frames(tmp_path: Path) -> None:
    service = MediaIngestService(paths(tmp_path), fake_media_backend())
    result = service.run(project_with_inputs(), CancellationToken(), discard_progress)
    assert result.artifacts[ArtifactRole.PROXY_FRAMES] == Path(
        f"proxies/{result.cache_key}"
    )
    assert result.artifacts[ArtifactRole.SOURCE_FRAMES] == Path(
        f"frames/{result.cache_key}"
    )


def test_compositor_uses_source_dimensions_and_registers_preview(
    tmp_path: Path,
) -> None:
    service = CompositeWorkflowService(paths(tmp_path), preview_frame_limit=150)
    result = service.run(completed_render_project(), CancellationToken(), discard_progress)
    assert result.artifacts[ArtifactRole.COMPOSITE_FRAMES] == Path(
        f"composites/{result.cache_key}"
    )
    assert result.artifacts[ArtifactRole.COMPOSITE_PREVIEW] == Path(
        f"previews/{result.cache_key}/composite-preview.mp4"
    )
    with Image.open(tmp_path / result.artifacts[ArtifactRole.COMPOSITE_FRAMES] / "000001.png") as image:
        assert image.size == (1920, 1080)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/pipeline/test_services.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-services-red
```

Expected: collection fails because the concrete services do not exist.

- [ ] **Step 3: Add frame-extraction backend**

Add `source_frame_command()` using:

```python
[
    ffmpeg, "-y", "-i", str(source), "-map", "0:v:0",
    "-vsync", "0", str(output_dir / "%06d.png"),
]
```

The implementation writes only into an `ArtifactPublisher` staging directory, validates a consecutive `000001.png` inventory, checks every frame is RGB with the probed source width and height, and requires the inventory count to equal `VideoSummary.frame_count`. If ffprobe cannot provide a frame count, ingest sets the persisted summary count from the validated inventory in one repository mutation before stage success.

- [ ] **Step 4: Implement project-aware stage adapters**

Every adapter:

1. validates required project fields and upstream succeeded stage cache keys;
2. computes a deterministic `cache_key()` from immutable input summaries, exact parameters, backend identity, and an implementation version;
3. builds only inside `ArtifactPublisher` staging paths;
4. calls `token.raise_if_cancelled()` before publication;
5. returns project-relative artifact paths.

Exact stage inputs are:

| Stage | Required inputs |
|---|---|
| ingest | source summary SHA/size/frame metadata |
| segment | ingest cache key, prompt, segmentation backend/config/checkpoint identity |
| solve_camera | ingest cache key, proxy inventory |
| map_trajectory | solve cache key, confirmed camera revision/artifact, foot-point authority, motion scale |
| composite | segment cache key, render cache key, source frame inventory, `edge_px=1`, preview height, preview frame limit |
| export | composite cache key, source summary FPS/frame count/audio identity |

`TrajectoryMapWorkflowService` converts the persisted target camera to `OrbitCamera.camera_to_world()`, calls `map_trajectory`, and persists the mapped matrices plus target vertical FOV. It requires the confirmed camera, confirmed preview artifact, and foot point to reference the same authority even though MVP depth is used only for picking.

`CompositeWorkflowService` reads source RGB PNG, rendered background PNG, and proxy-resolution L-mode mask inventories. It resizes masks to source dimensions with `cv2.INTER_NEAREST`, composites full-resolution frames one at a time, then downsizes at most the first 150 composite frames to the configured even preview dimensions and invokes `export_mp4` to create `composite-preview.mp4` with the corresponding leading source audio.

`ExportWorkflowService` invokes `export_mp4` over the full composite inventory and registers `EXPORT_VIDEO`.

- [ ] **Step 5: Run focused GREEN and commit**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/pipeline/test_services.py tests/integration/pipeline/test_workflow.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-services-green
uv run --offline --frozen --extra dev ruff check src/gs_video/media/ingest.py src/gs_video/pipeline/services.py tests/unit/pipeline/test_services.py tests/integration/pipeline/test_workflow.py
uv run --offline --frozen --extra dev mypy --strict src/gs_video/media/ingest.py src/gs_video/pipeline/services.py
```

Expected: all focused tests, Ruff, and strict mypy pass.

Commit:

```powershell
git add src/gs_video/media/ingest.py src/gs_video/pipeline/services.py tests/unit/pipeline/test_services.py tests/integration/pipeline/test_workflow.py
git commit -m "feat: implement concrete CPU workflow stages"
```

---

### Task 3: Isolated gsplat renderer worker and client

**Files:**
- Create: `src/gs_video/scene/worker_protocol.py`
- Create: `src/gs_video/scene/worker.py`
- Create: `src/gs_video/scene/worker_client.py`
- Create: `tests/unit/scene/test_worker_protocol.py`
- Create: `tests/unit/scene/test_worker_client.py`
- Create: `tests/integration/scene/test_worker_smoke.py`

**Interfaces:**
- Produces: `RendererWorkerClient.render_sequence(...) -> RenderSequence`.
- Produces: `RendererWorkerClient.render_pick(...) -> PickBuffer`.
- Produces: `RendererWorkerClient.probe() -> RendererWorkerIdentity`.
- Consumes: an explicit renderer worker argv prefix and the existing `GsplatRenderer`.

- [ ] **Step 1: Write protocol and process-lifecycle RED tests**

```python
def test_render_client_sends_argv_without_shell_or_token(tmp_path: Path) -> None:
    process = RecordingProcess(terminal_render_event())
    client = RendererWorkerClient(
        worker_prefix=("renderer-python",),
        process_factory=lambda command, **options: process.record(command, options),
    )
    client.render_sequence(request_fixture(tmp_path), discard_progress, CancellationToken())
    assert process.options["shell"] is False
    assert process.command[:4] == (
        "renderer-python", "-m", "gs_video.scene.worker", "--request"
    )
    assert "token" not in " ".join(process.command).lower()


def test_cancel_terminates_renderer_process_tree(tmp_path: Path) -> None:
    token = CancellationToken()
    process, tree = blocking_process_tree()
    client = RendererWorkerClient(
        worker_prefix=("renderer-python",),
        process_factory=lambda *_args, **_kwargs: process,
        tree_guard_factory=lambda _process: tree,
    )
    timer = threading.Timer(0.05, token.cancel)
    timer.start()
    with pytest.raises(CancelledError):
        client.render_sequence(request_fixture(tmp_path), discard_progress, token)
    timer.join()
    assert tree.terminated is True
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/scene/test_worker_protocol.py tests/unit/scene/test_worker_client.py -q -p no:cacheprovider --basetemp=.tmp/pytest-render-worker-red
```

Expected: collection fails because the renderer worker modules do not exist.

- [ ] **Step 3: Implement strict request/event schemas**

The request file is UTF-8 JSON, maximum 16 MiB, with `extra="forbid"` and one of:

```python
RenderSequenceRequest(
    type="render_sequence",
    scene_path=absolute_scene,
    camera_manifest=absolute_camera_json,
    output_dir=absolute_staging_dir,
    width=width,
    height=height,
    sh_degree=sh_degree,
    background=(0.0, 0.0, 0.0),
    preview_stride=1,
)

RenderPickRequest(
    type="render_pick",
    scene_path=absolute_scene,
    output_npz=absolute_owned_npz,
    camera=OrbitCameraPayload(...),
    width=width,
    height=height,
)
```

Stdout is newline-delimited JSON with only:

```python
ProgressEvent(type="progress", current=current, total=total, message=message)
TerminalEvent(type="complete", implementation_version=version, outputs=[...])
ErrorEvent(type="error", code=stable_code, message=bounded_message)
ProbeEvent(type="probe", torch=version, gsplat=version, device="cuda")
```

Reject unknown keys, non-finite values, lines over 64 KiB, duplicate terminal events, progress after terminal, output paths outside the requested owned root, and output inventories that do not match the terminal event.

- [ ] **Step 4: Implement worker and process isolation**

`scene.worker` loads `GaussianScene` and `GsplatRenderer` only after validating a request. Sequence requests load mapped camera JSON into immutable matrix-backed cameras, call the existing one-frame-at-a-time renderer, and emit progress. Pick requests call `render_pick` and atomically write `rgb` (`uint8 H×W×3`) plus `expected_depth` (`float32 H×W`) into an owned NPZ.

`RendererWorkerClient` mirrors the segmentation client safeguards:

- `shell=False`;
- `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` on Windows;
- process-tree guard established before releasing an exclusive startup gate;
- concurrent stdout/stderr draining;
- 64 KiB stdout line and 1 MiB diagnostic log bounds;
- cancellation polling and graceful-then-force tree termination;
- exact output-root, reparse-point, ordinary-file, dimension, dtype, and finite-value validation;
- worker registration so API shutdown can terminate live trees.

- [ ] **Step 5: Run unit GREEN and opt-in GPU smoke**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/scene/test_worker_protocol.py tests/unit/scene/test_worker_client.py -q -p no:cacheprovider --basetemp=.tmp/pytest-render-worker-green
uv run --offline --frozen --extra dev pytest tests/integration/scene/test_worker_smoke.py -m gpu -q -p no:cacheprovider --basetemp=.tmp/pytest-render-worker-gpu
uv run --offline --frozen --extra dev ruff check src/gs_video/scene/worker_protocol.py src/gs_video/scene/worker.py src/gs_video/scene/worker_client.py tests/unit/scene/test_worker_protocol.py tests/unit/scene/test_worker_client.py tests/integration/scene/test_worker_smoke.py
uv run --offline --frozen --extra dev mypy --strict src/gs_video/scene/worker_protocol.py src/gs_video/scene/worker_client.py
```

Expected: unit tests pass. GPU smoke passes only when an explicitly configured project-local renderer worker environment is available; otherwise it is collected and reports the documented fixture/configuration failure rather than silently skipping the release gate.

Commit:

```powershell
git add src/gs_video/scene/worker_protocol.py src/gs_video/scene/worker.py src/gs_video/scene/worker_client.py tests/unit/scene/test_worker_protocol.py tests/unit/scene/test_worker_client.py tests/integration/scene/test_worker_smoke.py
git commit -m "feat: isolate Gaussian rendering in a worker"
```

---

### Task 4: Renderer workflow adapter and production API assembly

**Files:**
- Modify: `src/gs_video/pipeline/services.py`
- Create: `src/gs_video/runtime.py`
- Modify: `src/gs_video/api/workflow.py`
- Modify: `src/gs_video/api/events.py`
- Modify: `src/gs_video/api/schemas.py`
- Modify: `src/gs_video/pipeline/runner.py`
- Modify: `src/gs_video/app.py`
- Modify: `src/gs_video/__main__.py`
- Create: `tests/unit/test_runtime.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `tests/security/test_local_api.py`
- Modify: `tests/integration/api/test_tasks.py`
- Create: `tests/integration/api/test_production_assembly.py`
- Modify: `apps/web/src/api/types.ts`

**Interfaces:**
- Produces: `WorkflowRuntimeConfig`.
- Produces: `load_runtime_config(path) -> WorkflowRuntimeConfig`.
- Produces: `assemble_api_services(config, session_token) -> tuple[ApiSettings, ApiServices]`.
- Produces: `RendererWorkflowService` and `WorkerPreviewService`.
- Produces: frame-aware `TaskSnapshot`/`TaskEvent` progress with elapsed time and ETA.
- Consumes: cache-scoped stage adapters, `RendererWorkerClient`, repository claim/CAS callbacks, and explicit worker/model paths.

- [ ] **Step 1: Write production-assembly RED tests**

```python
def test_production_assembly_registers_every_stage(tmp_path: Path) -> None:
    settings, services = assemble_api_services(runtime_config(tmp_path), SecretStr("secret"))
    assert all(services.pipeline_runner.supports(stage) for stage in StageName)
    assert services.preview_service is not None
    assert settings.bind_host == "127.0.0.1"


def test_serve_refuses_missing_runtime_configuration(tmp_path: Path) -> None:
    missing = tmp_path / "runtime.json"
    with pytest.raises(SystemExit, match="runtime configuration"):
        cli.main(["--serve", "--runtime-config", str(missing)])


def test_stage_progress_is_recoverable_from_rest(
    api_client: TestClient, auth_headers: dict[str, str],
) -> None:
    task_id = start_recording_task(api_client, auth_headers)
    snapshot = wait_for_task_progress(api_client, auth_headers, task_id)
    assert snapshot["current"] == 12
    assert snapshot["total"] == 30
    assert snapshot["progress"] == pytest.approx(0.4)
    assert snapshot["message"] == "渲染背景 12/30"
    assert snapshot["elapsed_seconds"] >= 0
    assert snapshot["eta_seconds"] is None or snapshot["eta_seconds"] >= 0
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/test_runtime.py tests/unit/test_cli.py tests/integration/api/test_production_assembly.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-assembly-red
```

Expected: collection fails because `WorkflowRuntimeConfig` and `assemble_api_services` do not exist.

- [ ] **Step 3: Implement explicit non-secret runtime configuration**

`WorkflowRuntimeConfig` is loaded from a user-selected project-local JSON file and contains:

```python
class WorkflowRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_root: Path
    model_root: Path
    segmentation_backend: SegmentationBackend
    segmentation_worker_prefix: tuple[str, ...]
    segmentation_model_config: Path
    segmentation_checkpoint: Path
    renderer_worker_prefix: tuple[str, ...]
    renderer_sh_degree: int = Field(default=3, ge=0, le=3)
    available_vram_limit_mb: int = Field(default=8192, ge=1024, le=8192)
```

`load_runtime_config(path)` treats the configuration file's parent as the explicit application workspace. It rejects relative paths, links/reparse points, a project root that contains the runtime configuration itself through an unexpected alias, empty argv items, segmentation model paths outside `model_root`, a `model_root` outside that workspace, and any token/port/origin fields. It never writes configuration to the user profile.

- [ ] **Step 4: Assemble real services and external preview rendering**

`assemble_api_services` must:

1. create or load `ProjectRepository(config.project_root)`;
2. create `VideoSegmenterClient` and `RendererWorkerClient`;
3. construct all seven concrete `WorkflowServices`;
4. call `build_mvp_workflow` with `save`, `persist_stage`, `compare_and_set_stage`, and `claim_stage`;
5. inject a `WorkerPreviewService` that preserves the existing scene path/hash/identity verification but delegates pick rendering to `RendererWorkerClient`;
6. inject an `EnvironmentDoctor` configured to probe FFmpeg, ffprobe, the segmentation worker, and the renderer worker;
7. inject a registry that terminates both worker clients on API shutdown.

`create_app` must no longer silently instantiate in-process `GsplatPreviewService` for a production service set. A missing preview service is a startup configuration error. Tests may continue to inject a fake preview service explicitly.

- [ ] **Step 5: Bridge stage progress into WebSocket and REST authority**

Change `PipelineRunner.run` to accept an optional per-run `ProgressEmitter`; the existing constructor emitter remains the test/default fallback. `TaskService` supplies a thread-safe relay for the current task. Each stage callback updates the authoritative `TaskSnapshot` and publishes a matching event containing:

```python
{
    "progress": current / total,
    "current": current,
    "total": total,
    "message": message[:512],
    "elapsed_seconds": monotonic() - started_at,
    "eta_seconds": (
        elapsed_seconds * (total - current) / current
        if current > 0 and current < total
        else 0.0 if current == total
        else None
    ),
}
```

Reject non-positive totals, out-of-range current values, non-finite times, and messages containing control characters other than tab. Preserve event ordering by awaiting the event-bus commit from the worker thread with `asyncio.run_coroutine_threadsafe`; cancellation interrupts the next stage callback. REST `GET /tasks/{id}` exposes the same last committed progress fields, so WebSocket loss never resets a long render to an indeterminate state. Extend the TypeScript `TaskDto` and `TaskProgressEvent` fields without making the UI fabricate ETA when the backend has not reported one.

- [ ] **Step 6: Remove the empty-runner production path**

`--serve` requires `--runtime-config <absolute-json>`. `run_api` receives the loaded config and a secret supplied by the current launcher boundary; until Task 15 adds the private stdin handshake, a developer-only `--session-token-stdin` reads exactly one bounded line from inherited stdin and never prints or logs it. The CLI must not generate a temporary project or construct `PipelineRunner(project, {})`.

- [ ] **Step 7: Run GREEN, security gate, and commit**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/unit/test_runtime.py tests/unit/test_cli.py tests/integration/api/test_tasks.py tests/integration/api/test_production_assembly.py tests/security/test_local_api.py -q -p no:cacheprovider --basetemp=.tmp/pytest-production-assembly-green
uv run --offline --frozen --extra dev ruff check src/gs_video/runtime.py src/gs_video/pipeline/services.py src/gs_video/api/workflow.py src/gs_video/api/events.py src/gs_video/api/schemas.py src/gs_video/pipeline/runner.py src/gs_video/app.py src/gs_video/__main__.py tests/unit/test_runtime.py tests/unit/test_cli.py tests/integration/api/test_tasks.py tests/integration/api/test_production_assembly.py tests/security/test_local_api.py
uv run --offline --frozen --extra dev mypy --strict src/gs_video/runtime.py src/gs_video/pipeline/services.py src/gs_video/api/workflow.py src/gs_video/api/events.py src/gs_video/pipeline/runner.py src/gs_video/app.py
```

Expected: focused tests, security tests, Ruff, and strict mypy pass; no production code path creates an empty runner.

Commit:

```powershell
git add src/gs_video/runtime.py src/gs_video/pipeline/services.py src/gs_video/api/workflow.py src/gs_video/api/events.py src/gs_video/api/schemas.py src/gs_video/pipeline/runner.py src/gs_video/app.py src/gs_video/__main__.py tests/unit/test_runtime.py tests/unit/test_cli.py tests/integration/api/test_tasks.py tests/integration/api/test_production_assembly.py tests/security/test_local_api.py apps/web/src/api/types.ts
git commit -m "feat: assemble the production processing workflow"
```

---

### Task 5: Authenticated composite-preview artifact API

**Files:**
- Modify: `src/gs_video/api/schemas.py`
- Create: `src/gs_video/api/preview_routes.py`
- Modify: `src/gs_video/api/routes.py`
- Modify: `src/gs_video/api/assets.py`
- Modify: `tests/integration/api/test_workflow.py`
- Modify: `tests/security/test_local_api.py`
- Modify: `apps/web/src/api/types.ts`
- Modify: `apps/web/src/api/backend-client.ts`
- Modify: `apps/web/src/api/http-backend-client.ts`
- Modify: `apps/web/src/api/http-backend-client.test.ts`

**Interfaces:**
- Produces: `GET /api/v1/projects/current/composite-preview`.
- Produces: `GET /api/v1/artifacts/composite-previews/{artifact_id}`.
- Produces: `BackendClient.getCompositePreview()` and `fetchCompositePreviewArtifact()`.
- Consumes: succeeded `COMPOSITE` stage registration with `COMPOSITE_PREVIEW`.

- [ ] **Step 1: Write API/client RED tests**

```python
def test_composite_preview_is_opaque_authenticated_and_cache_bound(
    api_client: TestClient, auth_headers: dict[str, str], completed_project: Project,
) -> None:
    descriptor = api_client.get(
        "/api/v1/projects/current/composite-preview", headers=auth_headers
    )
    assert descriptor.status_code == 200
    payload = descriptor.json()
    assert set(payload) == {
        "artifact_id", "filename", "size", "sha256",
        "duration_seconds", "fps", "frame_count",
    }
    assert "path" not in payload
    video = api_client.get(
        f"/api/v1/artifacts/composite-previews/{payload['artifact_id']}",
        headers=auth_headers,
    )
    assert video.headers["content-type"] == "video/mp4"
```

```ts
it('fetches the current opaque composite preview', async () => {
  const descriptor = await client.getCompositePreview()
  await client.fetchCompositePreviewArtifact(descriptor.artifact_id)
  expect(fetchImpl).toHaveBeenNthCalledWith(
    2,
    expect.stringContaining(`/artifacts/composite-previews/${descriptor.artifact_id}`),
    expect.objectContaining({ headers: expect.any(Headers) }),
  )
})
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/integration/api/test_workflow.py tests/security/test_local_api.py -q -p no:cacheprovider --basetemp=.tmp/pytest-composite-preview-api-red
$env:npm_config_cache='E:\Project\GS-Video\.cache\npm'
npm run test:web -- --run apps/web/src/api/http-backend-client.test.ts
```

Expected: Python routes and TypeScript client methods are missing.

- [ ] **Step 3: Implement stable opaque preview resolution**

The resolver accepts only the exact `previews/<composite-cache-key>/composite-preview.mp4` registration from a succeeded composite stage. It enforces:

- canonical confinement beneath `<project>/previews`;
- no reparse component or hard link;
- ordinary non-empty MP4 no larger than 256 MiB;
- held-handle pre/post/path identity stability;
- current file hash and ffprobe metadata;
- artifact ID bound to project ID, composite cache key, file identity, size, and SHA-256.

The descriptor exposes no filesystem path. The blob endpoint revalidates current project/stage authority and the same held identity before returning `video/mp4`.

- [ ] **Step 4: Run GREEN and commit**

Run:

```powershell
uv run --offline --frozen --extra dev pytest tests/integration/api/test_workflow.py tests/security/test_local_api.py -q -p no:cacheprovider --basetemp=.tmp/pytest-composite-preview-api-green
npm run test:web -- --run apps/web/src/api/http-backend-client.test.ts
npm run typecheck:web
```

Expected: API/security tests, client tests, and typecheck pass.

Commit:

```powershell
git add src/gs_video/api/schemas.py src/gs_video/api/preview_routes.py src/gs_video/api/routes.py src/gs_video/api/assets.py tests/integration/api/test_workflow.py tests/security/test_local_api.py apps/web/src/api/types.ts apps/web/src/api/backend-client.ts apps/web/src/api/http-backend-client.ts apps/web/src/api/http-backend-client.test.ts
git commit -m "feat: expose verified composite previews"
```

---

### Task 6: Display the real composite preview and verify the complete flow

**Files:**
- Modify: `apps/web/src/features/preview/preview-page.tsx`
- Modify: `apps/web/src/features/preview/preview-page.test.tsx`
- Modify: `apps/web/src/features/workflow/guided-workflow.test.tsx`
- Modify: `apps/web/src/app/app.tsx`
- Modify: `apps/web/src/app/app.css`
- Create: `tests/integration/api/test_real_workflow.py`
- Modify: `.superpowers/sdd/task-14-report.md` (ignored ledger)
- Modify: `.superpowers/sdd/progress.md` (ignored ledger)

**Interfaces:**
- Consumes: `BackendClient.getCompositePreview()` and `fetchCompositePreviewArtifact()`.
- Produces: a real `<video controls>` preview bound to the succeeded composite cache.
- Produces: a mock-worker integration proof from imported assets through verified preview and final export.

- [ ] **Step 1: Write UI and integration RED tests**

```tsx
it('shows only the composite video registered by the succeeded stage', async () => {
  render(<PreviewPage {...props({ compositeStatus: 'succeeded' })} />)
  const video = await screen.findByLabelText('低分辨率合成预览')
  expect(video).toHaveAttribute('controls')
  expect(backend.getCompositePreview).toHaveBeenCalledOnce()
})

it('shows backend current-frame timing without inventing missing ETA', async () => {
  render(<App {...propsWithRunningRender({
    current: 12,
    total: 30,
    message: '渲染背景 12/30',
    elapsed_seconds: 8,
    eta_seconds: 12,
  })} />)
  expect(screen.getByText('渲染背景 12/30')).toBeVisible()
  expect(screen.getByText(/已用时 8 秒/)).toBeVisible()
  expect(screen.getByText(/预计剩余 12 秒/)).toBeVisible()
})
```

```python
def test_assembled_workflow_produces_preview_and_verified_export(
    production_harness: ProductionHarness,
) -> None:
    project = production_harness.run_with_fake_workers()
    assert project.stages[StageName.EXPORT].status is StageStatus.SUCCEEDED
    assert (
        project.stages[StageName.COMPOSITE].artifacts[ArtifactRole.COMPOSITE_PREVIEW]
        .endswith("/composite-preview.mp4")
    )
    assert project.workflow.export_result is not None
    assert project.workflow.export_result.verified is True
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
$env:npm_config_cache='E:\Project\GS-Video\.cache\npm'
npm run test:web -- --run apps/web/src/features/preview/preview-page.test.tsx apps/web/src/features/workflow/guided-workflow.test.tsx
$env:UV_CACHE_DIR='E:\Project\GS-Video\.cache\uv'
uv run --offline --frozen --extra dev pytest tests/integration/api/test_real_workflow.py -q -p no:cacheprovider --basetemp=.tmp/pytest-real-workflow-red
```

Expected: the UI still shows the static scene frame and the assembled mock-worker flow does not exist.

- [ ] **Step 3: Implement preview lifecycle**

When composite status is succeeded, fetch the descriptor then the MP4 Blob, create one object URL, and render:

```tsx
<video
  aria-label="低分辨率合成预览"
  controls
  playsInline
  preload="metadata"
  src={previewUrl}
/>
```

Abort stale fetches, compare the descriptor artifact ID before publication, revoke URLs on replacement/null authority/unmount, and retain the static Gaussian frame only as a clearly labelled camera-reference fallback before composite success.

The task footer displays the authoritative `message`, `current/total`, elapsed time, and ETA from the latest matching event or REST snapshot. It omits the ETA label when `eta_seconds` is null and never derives a second estimate in the browser.

- [ ] **Step 4: Run complete verification**

Run:

```powershell
$env:UV_CACHE_DIR='E:\Project\GS-Video\.cache\uv'
uv run --offline --frozen --extra dev pytest -m "not gpu" --import-mode=importlib -q -p no:cacheprovider --basetemp=.tmp/pytest-production-workflow-full
uv run --offline --frozen --extra dev ruff check src tests
uv run --offline --frozen --extra dev mypy --strict src/gs_video/runtime.py src/gs_video/pipeline/services.py src/gs_video/scene/worker_protocol.py src/gs_video/scene/worker_client.py src/gs_video/api/preview_routes.py
uv run --offline --frozen --extra dev python -m compileall -q src
$env:npm_config_cache='E:\Project\GS-Video\.cache\npm'
npm run test:web -- --run --reporter=dot
npm run typecheck:web
npm run build:web
git diff --check
```

Expected: full non-GPU Python, Ruff, strict mypy, compileall, all frontend tests, typecheck, production build, and diff checks pass.

- [ ] **Step 5: Perform browser QA and 8 GB worker smoke**

Use the real FastAPI app with fake workers for deterministic desktop and narrow-browser E2E. Verify the real `<video>` preview, refresh recovery, progress, cancellation, explicit browser save, and no console errors. Then run one configured 64×36 real renderer-worker pick and a short 540p preview render; record peak CUDA allocation and require it to stay below 8192 MiB. All model paths and outputs remain under the project.

- [ ] **Step 6: Review, record, and commit**

Update the ignored Task 14 report/ledger with exact commands, test counts, browser screenshots, and GPU environment status. Request independent code review across the entire production-assembly range and fix every Critical or Important finding before marking Task 14 complete.

Commit:

```powershell
git add apps/web/src/features/preview/preview-page.tsx apps/web/src/features/preview/preview-page.test.tsx apps/web/src/features/workflow/guided-workflow.test.tsx apps/web/src/app/app.tsx apps/web/src/app/app.css tests/integration/api/test_real_workflow.py docs/superpowers/plans/2026-07-17-production-workflow-assembly.md
git commit -m "feat: complete the runnable Gaussian video workflow"
```
