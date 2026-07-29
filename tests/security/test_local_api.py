import errno
import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gs_video import __main__ as cli
from gs_video import app as local_app
from gs_video.api import uploads
from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiError, ApiSettings
from gs_video.api.uploads import CHUNK_LIMIT
from gs_video.app import create_app
from gs_video.domain.models import ArtifactRole, StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.media.ffmpeg import VideoMetadata
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.pipeline.events import ProgressEmitter, discard_progress
from gs_video.project.repository import ProjectRepository


TOKEN = "security-session-token"
ORIGIN = "http://127.0.0.1:4173"


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=0, issues=[])


class SucceedingRunner:
    def run(
        self,
        name: StageName,
        token: CancellationToken,
        emit: ProgressEmitter = discard_progress,
    ) -> StageState:
        del name, token, emit
        return StageState(status=StageStatus.SUCCEEDED)


class NoopWorkerRegistry:
    async def terminate_all(self) -> None:
        return None


class UnusedPreviewService:
    def render_pick(self, *args: object) -> object:
        del args
        raise AssertionError("preview rendering is outside this test")


def make_security_app(root: Path, *, max_active_uploads: int = 64) -> FastAPI:
    repository = ProjectRepository(root)
    repository.save(repository.create("security"))
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=NoopWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
        max_active_uploads=max_active_uploads,
    )
    return create_app(settings, services)


@pytest.fixture
def client_and_root(tmp_path: Path):  # type: ignore[no-untyped-def]
    root = tmp_path / "project"
    with TestClient(make_security_app(root)) as client:
        yield client, root


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def assert_stable_error(response: object, *, status_code: int, code: str) -> None:
    assert hasattr(response, "status_code")
    assert response.status_code == status_code  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert set(body) == {"code", "category", "message", "retryable"}
    assert body["code"] == code


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.2", "localhost", "example.com"])
def test_settings_reject_non_ip_or_non_loopback_bind_hosts(host: str) -> None:
    with pytest.raises(ValidationError):
        ApiSettings(
            bind_host=host, port=0, session_token=TOKEN, allowed_origins=(ORIGIN,)
        )


def test_settings_reject_wildcard_origin() -> None:
    with pytest.raises(ValidationError):
        ApiSettings(
            bind_host="127.0.0.1", port=0, session_token=TOKEN, allowed_origins=("*",)
        )


def test_rest_authentication_accepts_only_bearer_header(client_and_root) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    query = client.get(f"/healthz?token={TOKEN}")
    wrong_scheme = client.get("/healthz", headers={"Authorization": f"Basic {TOKEN}"})
    correct = client.get("/healthz", headers=auth_headers())

    assert query.status_code == 401
    assert wrong_scheme.status_code == 401
    assert correct.status_code == 200


def test_composite_preview_blob_revalidates_current_stage_authority(
    client_and_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    cache_key = "composite-security-key"
    relative = f"previews/{cache_key}/composite-preview.mp4"
    preview = root / relative
    preview.parent.mkdir()
    preview.write_bytes(b"composite-preview")
    repository = client.app.state.services.project_repository
    repository.update(
        lambda project: project.stages.__setitem__(
            StageName.COMPOSITE,
            StageState(
                status=StageStatus.SUCCEEDED,
                cache_key=cache_key,
                output_paths=[relative],
                artifacts={ArtifactRole.COMPOSITE_PREVIEW: relative},
            ),
        )
    )

    class Inspector:
        def probe(self, path: Path):  # type: ignore[no-untyped-def]
            assert path == preview
            from gs_video.media.ffmpeg import VideoMetadata

            return VideoMetadata(16, 9, 1.0, "30", frame_count=30)

    monkeypatch.setattr(client.app.state, "export_inspector", Inspector())
    descriptor = client.get(
        "/api/v1/projects/current/composite-preview", headers=auth_headers()
    )
    assert descriptor.status_code == 200
    repository.update(
        lambda project: setattr(
            project.stages[StageName.COMPOSITE], "status", StageStatus.STALE
        )
    )

    stale = client.get(
        "/api/v1/artifacts/composite-previews/"
        f"{descriptor.json()['artifact_id']}",
        headers=auth_headers(),
    )

    assert_stable_error(stale, status_code=409, code="composite_preview_changed")


@pytest.mark.parametrize("mutation", ["empty", "oversize", "hard_link", "reparse"])
def test_composite_preview_rejects_unsafe_registered_files(
    client_and_root,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    cache_key = f"composite-{mutation}"
    relative = f"previews/{cache_key}/composite-preview.mp4"
    preview = root / relative
    preview.parent.mkdir()
    preview.write_bytes(b"composite-preview")
    repository = client.app.state.services.project_repository
    repository.update(
        lambda project: project.stages.__setitem__(
            StageName.COMPOSITE,
            StageState(
                status=StageStatus.SUCCEEDED,
                cache_key=cache_key,
                output_paths=[relative],
                artifacts={ArtifactRole.COMPOSITE_PREVIEW: relative},
            ),
        )
    )

    if mutation == "empty":
        preview.write_bytes(b"")
    elif mutation == "oversize":
        with preview.open("r+b") as stream:
            stream.truncate(256 * 1024 * 1024 + 1)
    elif mutation == "hard_link":
        try:
            os.link(preview, root / "outside-preview.mp4")
        except OSError:
            pytest.skip("the filesystem does not support hard-link coverage")
    else:
        outside = root.parent / "outside-preview.mp4"
        outside.write_bytes(b"outside")
        preview.unlink()
        try:
            preview.symlink_to(outside)
        except OSError:
            pytest.skip("the filesystem does not permit reparse-point coverage")

    class Inspector:
        def probe(self, path: Path) -> VideoMetadata:
            raise AssertionError(f"unsafe preview reached ffprobe: {path}")

    monkeypatch.setattr(client.app.state, "export_inspector", Inspector())
    response = client.get(
        "/api/v1/projects/current/composite-preview", headers=auth_headers()
    )

    assert_stable_error(response, status_code=409, code="composite_preview_changed")


def test_composite_preview_rejects_cache_registered_path_outside_previews_root(
    client_and_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    cache_key = "../escaped-preview"
    relative = f"previews/{cache_key}/composite-preview.mp4"
    escaped_preview = root / "escaped-preview" / "composite-preview.mp4"
    escaped_preview.parent.mkdir()
    escaped_preview.write_bytes(b"outside-previews-root")
    repository = client.app.state.services.project_repository
    repository.update(
        lambda project: project.stages.__setitem__(
            StageName.COMPOSITE,
            StageState(
                status=StageStatus.SUCCEEDED,
                cache_key=cache_key,
                output_paths=[relative],
                artifacts={ArtifactRole.COMPOSITE_PREVIEW: relative},
            ),
        )
    )

    class Inspector:
        def probe(self, path: Path) -> VideoMetadata:
            raise AssertionError(f"escaped preview reached ffprobe: {path}")

    monkeypatch.setattr(client.app.state, "export_inspector", Inspector())
    descriptor = client.get(
        "/api/v1/projects/current/composite-preview", headers=auth_headers()
    )
    blob = client.get(
        "/api/v1/artifacts/composite-previews/" + "a" * 32,
        headers=auth_headers(),
    )

    assert_stable_error(descriptor, status_code=409, code="composite_preview_changed")
    assert_stable_error(blob, status_code=409, code="composite_preview_changed")
    assert descriptor.content != b"outside-previews-root"
    assert blob.content != b"outside-previews-root"


def test_composite_preview_descriptor_revalidates_authority_after_probe(
    client_and_root,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    cache_key = "composite-race"
    relative = f"previews/{cache_key}/composite-preview.mp4"
    preview = root / relative
    preview.parent.mkdir()
    preview.write_bytes(b"composite-preview")
    repository = client.app.state.services.project_repository
    repository.update(
        lambda project: project.stages.__setitem__(
            StageName.COMPOSITE,
            StageState(
                status=StageStatus.SUCCEEDED,
                cache_key=cache_key,
                output_paths=[relative],
                artifacts={ArtifactRole.COMPOSITE_PREVIEW: relative},
            ),
        )
    )
    probe_started = Event()
    release_probe = Event()

    class BlockingInspector:
        def probe(self, path: Path) -> VideoMetadata:
            assert path == preview
            probe_started.set()
            assert release_probe.wait(2)
            return VideoMetadata(16, 9, 1.0, "30", frame_count=30)

    monkeypatch.setattr(client.app.state, "export_inspector", BlockingInspector())
    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(
            client.get,
            "/api/v1/projects/current/composite-preview",
            headers=auth_headers(),
        )
        assert probe_started.wait(1)
        repository.update(
            lambda project: setattr(
                project.stages[StageName.COMPOSITE], "status", StageStatus.STALE
            )
        )
        release_probe.set()
        response = request.result(timeout=2)

    assert_stable_error(response, status_code=409, code="composite_preview_changed")


def test_outer_boundary_rejects_unauthorized_body_without_consuming_it(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root
    consumed = 0

    def body() -> Iterator[bytes]:
        nonlocal consumed
        consumed += 1
        yield b"{" + b"x" * (64 * 1024)

    response = client.patch(
        "/api/v1/projects/current",
        content=body(),
        headers={"Content-Type": "application/json", "Origin": ORIGIN},
    )

    assert_stable_error(response, status_code=401, code="invalid_session")
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert consumed == 0


def test_outer_boundary_rejects_authorized_oversized_json_before_validation(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    response = client.patch(
        "/api/v1/projects/current",
        content=b"{" + b"x" * (64 * 1024),
        headers={
            **auth_headers(),
            "Content-Type": "application/json",
            "Origin": ORIGIN,
        },
    )

    assert_stable_error(response, status_code=413, code="request_body_too_large")
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_outer_boundary_bounds_chunked_body_without_content_length(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    def body() -> Iterator[bytes]:
        yield b"{" + b"x" * (32 * 1024)
        yield b"y" * (32 * 1024)

    response = client.patch(
        "/api/v1/projects/current",
        content=body(),
        headers={
            **auth_headers(),
            "Content-Type": "application/json",
            "Origin": ORIGIN,
        },
    )

    assert "content-length" not in response.request.headers
    assert_stable_error(response, status_code=413, code="request_body_too_large")
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_outer_boundary_invalid_content_length_has_browser_readable_error(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    response = client.patch(
        "/api/v1/projects/current",
        content=b"{}",
        headers={
            **auth_headers(),
            "Content-Type": "application/json",
            "Content-Length": "invalid",
            "Origin": ORIGIN,
        },
    )

    assert_stable_error(response, status_code=400, code="invalid_content_length")
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_cors_preflight_is_stable_and_does_not_require_bearer(client_and_root) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root
    preflight = {
        "Origin": ORIGIN,
        "Access-Control-Request-Method": "PATCH",
        "Access-Control-Request-Headers": "authorization,content-type",
    }

    allowed = client.options("/api/v1/projects/current", headers=preflight)
    bad_origin = client.options(
        "/api/v1/projects/current",
        headers={**preflight, "Origin": "https://attacker.invalid"},
    )
    bad_method = client.options(
        "/api/v1/projects/current",
        headers={**preflight, "Access-Control-Request-Method": "TRACE"},
    )
    bad_header = client.options(
        "/api/v1/projects/current",
        headers={**preflight, "Access-Control-Request-Headers": "x-evil"},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert_stable_error(bad_origin, status_code=400, code="cors_forbidden")
    assert_stable_error(bad_method, status_code=400, code="cors_forbidden")
    assert_stable_error(bad_header, status_code=400, code="cors_forbidden")


def test_api_settings_never_repr_or_serialize_session_token() -> None:
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )

    assert TOKEN not in repr(settings)
    assert TOKEN not in str(settings.model_dump())
    assert settings.session_token.get_secret_value() == TOKEN


def test_lifespan_closes_upload_handles_even_when_prior_cleanup_fails(
    tmp_path: Path,
) -> None:
    app = make_security_app(tmp_path / "shutdown-cleanup")
    uploads_closed = Event()
    workers_terminated = Event()

    with pytest.raises(RuntimeError, match="simulated task cleanup failure"):
        with TestClient(app):
            manager = app.state.upload_manager
            original_close = manager.close

            def recording_close() -> None:
                uploads_closed.set()
                original_close()

            async def fail_task_cleanup() -> None:
                raise RuntimeError("simulated task cleanup failure")

            async def recording_worker_cleanup() -> None:
                workers_terminated.set()

            manager.close = recording_close
            app.state.task_service.request_cancel_all = fail_task_cleanup
            app.state.services.worker_registry.terminate_all = recording_worker_cleanup

    assert workers_terminated.is_set()
    assert uploads_closed.is_set()


def test_lifespan_closes_upload_handles_when_task_service_start_fails(
    tmp_path: Path,
) -> None:
    app = make_security_app(tmp_path / "startup-cleanup")
    uploads_closed = Event()
    original_close = app.state.upload_manager.close

    def recording_close() -> None:
        uploads_closed.set()
        original_close()

    async def fail_start() -> None:
        raise RuntimeError("simulated task startup failure")

    app.state.upload_manager.close = recording_close
    app.state.task_service.start = fail_start

    with pytest.raises(RuntimeError, match="simulated task startup failure"):
        with TestClient(app):
            pass

    assert uploads_closed.is_set()


def create_upload(client: TestClient, content: bytes, filename: str = "clip.mp4") -> dict[str, object]:
    response = client.post(
        "/api/v1/uploads",
        json={
            "filename": filename,
            "mime_type": "video/mp4",
            "total_size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        headers=auth_headers(),
    )
    assert response.status_code == 201
    return response.json()


def test_browser_upload_resumes_idempotently_and_completes_below_project_root(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"recoverable upload"
    created = create_upload(client, content)
    upload_id = created["id"]

    first = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0", content=content, headers=auth_headers()
    )
    duplicate = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0", content=content, headers=auth_headers()
    )
    resumed = client.get(f"/api/v1/uploads/{upload_id}", headers=auth_headers())
    completed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )

    assert first.status_code == duplicate.status_code == 204
    assert resumed.json()["uploaded_chunks"] == [0]
    assert completed.status_code == 201
    relative = Path(completed.json()["path"])
    assert not relative.is_absolute()
    destination = (root / relative).resolve()
    assert destination.is_relative_to(root.resolve())
    assert destination.read_bytes() == content


def test_browser_upload_rejects_conflicting_duplicate(client_and_root) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root
    content = b"original"
    upload_id = create_upload(client, content)["id"]
    client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0", content=content, headers=auth_headers()
    )

    conflict = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=b"changed!",
        headers=auth_headers(),
    )

    assert conflict.status_code == 409
    assert conflict.json()["code"] == "chunk_conflict"


def test_browser_upload_rejects_path_traversal_and_oversized_chunk(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"x"
    upload_id = create_upload(client, content)["id"]

    traversal = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/../../project.json",
        content=b"x",
        headers=auth_headers(),
    )
    oversized = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=b"x" * (CHUNK_LIMIT + 1),
        headers=auth_headers(),
    )

    assert traversal.status_code in {400, 413}
    assert oversized.status_code == 413
    assert (root / "project.json").read_text(encoding="utf-8").startswith("{")


def test_browser_upload_rejects_malformed_index_and_wrong_expected_size(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root
    upload_id = create_upload(client, b"two-bytes")["id"]

    malformed = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/-1",
        content=b"two-bytes",
        headers=auth_headers(),
    )
    wrong_size = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=b"short",
        headers=auth_headers(),
    )

    assert malformed.status_code == 400
    assert malformed.json()["code"] == "invalid_chunk_index"
    assert wrong_size.status_code == 400
    assert wrong_size.json()["code"] == "chunk_size_mismatch"


@pytest.mark.parametrize("filename", ["../clip.mp4", "folder/clip.mp4", "C:\\clip.mp4"])
def test_browser_upload_requires_basename_filename(client_and_root, filename: str) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    response = client.post(
        "/api/v1/uploads",
        json={
            "filename": filename,
            "mime_type": "video/mp4",
            "total_size": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        },
        headers=auth_headers(),
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_browser_upload_declarations_are_bounded(tmp_path: Path) -> None:
    with TestClient(
        make_security_app(tmp_path / "bounded", max_active_uploads=1)
    ) as client:
        first = create_upload(client, b"first")
        second = client.post(
            "/api/v1/uploads",
            json={
                "filename": "second.mp4",
                "mime_type": "video/mp4",
                "total_size": 6,
                "sha256": hashlib.sha256(b"second").hexdigest(),
            },
            headers=auth_headers(),
        )

    assert first["id"]
    assert second.status_code == 429
    assert second.json()["code"] == "upload_limit_reached"


def test_upload_validates_hash_and_cancellation_cleans_only_owned_directory(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"expected"
    created = create_upload(client, content)
    upload_id = created["id"]
    client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=b"differs!",
        headers=auth_headers(),
    )

    mismatch = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )
    cancelled = client.delete(f"/api/v1/uploads/{upload_id}", headers=auth_headers())

    assert mismatch.status_code in {400, 409}
    assert mismatch.json()["code"] == "upload_hash_mismatch"
    assert cancelled.status_code == 204
    assert not (root / ".uploads" / str(upload_id)).exists()
    assert (root / "project.json").exists()


def test_upload_rejects_allocated_directory_identity_swap_without_mutating_replacement(
    client_and_root, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"identity-bound"
    upload_id = str(create_upload(client, content)["id"])
    allocated = root / ".uploads" / upload_id
    moved = root.parent / f"{upload_id}.moved"
    replacement_target = root.parent / "replacement-target"
    replacement_target.mkdir()
    mutation_started = Event()
    release_mutation = Event()
    original_write = uploads._write_spool

    def pausing_write(
        stream, *, offset: int, content: bytes
    ) -> None:  # type: ignore[no-untyped-def]
        mutation_started.set()
        assert release_mutation.wait(2)
        original_write(stream, offset=offset, content=content)

    monkeypatch.setattr(uploads, "_write_spool", pausing_write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        request = executor.submit(
            client.put,
            f"/api/v1/uploads/{upload_id}/chunks/0",
            content=content,
            headers=auth_headers(),
        )
        assert mutation_started.wait(1)
        try:
            allocated.rename(moved)
        except OSError:
            swap_succeeded = False
        else:
            swap_succeeded = True
            try:
                allocated.symlink_to(replacement_target, target_is_directory=True)
            except OSError:
                allocated.mkdir()
        release_mutation.set()
        written = request.result(timeout=2)

    if swap_succeeded:
        assert_stable_error(written, status_code=409, code="upload_path_changed")
        assert not (moved / "0.chunk").exists()
        assert not (replacement_target / "0.chunk").exists()
    else:
        assert written.status_code == 204
        cancelled = client.delete(
            f"/api/v1/uploads/{upload_id}", headers=auth_headers()
        )
        assert cancelled.status_code == 204


def test_upload_rejects_chunk_identity_swap_and_hard_link(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"identity-bound-chunk"
    upload_id = str(create_upload(client, content)["id"])
    outside = root / "outside-owned.bin"
    destination = next((root / "source").glob(f"{upload_id}-*"))
    try:
        os.link(destination, outside)
    except OSError:
        hard_link_created = False
    else:
        hard_link_created = True
    chunked = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content,
        headers=auth_headers(),
    )

    completed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )
    cancelled = client.delete(
        f"/api/v1/uploads/{upload_id}", headers=auth_headers()
    )

    assert chunked.status_code == 204
    if hard_link_created:
        assert_stable_error(completed, status_code=409, code="upload_path_changed")
        assert cancelled.status_code == 404
        assert outside.read_bytes() == b""
    else:
        assert completed.status_code == 201
        assert cancelled.status_code == 204


def test_upload_manager_rejects_non_windows_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uploads, "_IS_WINDOWS", False, raising=False)

    with pytest.raises(RuntimeError, match="Windows 11"):
        uploads.UploadManager(tmp_path / "unsupported", CHUNK_LIMIT, 4)


def test_upload_manager_rolls_back_partially_acquired_root_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "partial-init"
    original_acquire = uploads._DirectoryLease.acquire
    calls = 0

    def fail_second_acquire(path: Path, expected=None):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EACCES, "simulated lease failure")
        return original_acquire(path, expected)

    monkeypatch.setattr(
        uploads._DirectoryLease, "acquire", staticmethod(fail_second_acquire)
    )

    with pytest.raises(ValueError, match="ordinary directories"):
        uploads.UploadManager(root, CHUNK_LIMIT, 4)

    moved = tmp_path / "partial-init-moved"
    root.rename(moved)
    assert moved.is_dir()


def test_upload_manager_close_attempts_every_root_lease_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "close-all-roots"
    repository = ProjectRepository(root)
    repository.save(repository.create("close-all-roots"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    source_close = manager._source_root_lease.close
    upload_close = manager._upload_root_lease.close
    root_close = manager._root_lease.close
    calls: list[str] = []

    def close_source_then_fail() -> None:
        calls.append("source")
        source_close()
        raise OSError(errno.EACCES, "simulated source lease close failure")

    def close_upload() -> None:
        calls.append("upload")
        upload_close()

    def close_root() -> None:
        calls.append("root")
        root_close()

    monkeypatch.setattr(manager._source_root_lease, "close", close_source_then_fail)
    monkeypatch.setattr(manager._upload_root_lease, "close", close_upload)
    monkeypatch.setattr(manager._root_lease, "close", close_root)

    manager.close()

    assert calls == ["source", "upload", "root"]


def test_upload_allocation_failure_removes_untracked_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "allocation-failure"
    repository = ProjectRepository(root)
    repository.save(repository.create("allocation-failure"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    original_create = uploads._OwnedFile.create
    calls = 0

    def fail_destination_create(
        lease, name: str, *, delete_on_close: bool
    ):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOSPC, "simulated allocation failure")
        return original_create(lease, name, delete_on_close=delete_on_close)

    monkeypatch.setattr(
        uploads._OwnedFile, "create", staticmethod(fail_destination_create)
    )
    request = uploads.UploadCreateRequest(
        filename="clip.mp4",
        mime_type="video/mp4",
        total_size=1,
        sha256=hashlib.sha256(b"x").hexdigest(),
    )

    with pytest.raises(ApiError) as error:
        manager.create(request)

    assert error.value.envelope.code == "storage_full"
    assert not list((root / ".uploads").iterdir())
    assert not list((root / "source").iterdir())
    manager.close()


def test_unsafe_prepare_abort_closes_and_removes_scratch_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unsafe-abort"
    repository = ProjectRepository(root)
    repository.save(repository.create("unsafe-abort"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    content = b"unsafe-abort"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    manager.put_chunk(created.id, 0, content)
    record = manager._records[created.id]

    def reject_destination_link() -> None:
        raise uploads._UnsafePathError("simulated unsafe link")

    monkeypatch.setattr(record.destination, "verify_link", reject_destination_link)

    with pytest.raises(ApiError) as error:
        manager.complete(created.id, lambda _: None)

    assert error.value.envelope.code == "upload_path_changed"
    assert created.id not in manager._records
    assert record.directory_lease is None
    assert not (root / ".uploads" / created.id).exists()
    assert not list((root / "source").glob(f"{created.id}-*"))
    manager.close()


def test_cancel_retries_failed_handle_delete_without_leaving_source_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "delete-retry"
    repository = ProjectRepository(root)
    repository.save(repository.create("delete-retry"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    content = b"delete-retry"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    destination = next((root / "source").glob(f"{created.id}-*"))
    original_mark_delete = uploads._mark_windows_delete
    delete_attempts = 0

    def fail_first_delete(stream) -> None:  # type: ignore[no-untyped-def]
        nonlocal delete_attempts
        delete_attempts += 1
        if delete_attempts == 1:
            raise OSError(errno.EACCES, "simulated delete-mark failure")
        original_mark_delete(stream)

    monkeypatch.setattr(uploads, "_mark_windows_delete", fail_first_delete)

    with pytest.raises(OSError, match="simulated delete-mark failure"):
        manager.cancel(created.id)
    assert destination.exists()

    manager.cancel(created.id)

    assert delete_attempts == 2
    assert not destination.exists()
    assert not (root / ".uploads" / created.id).exists()
    manager.close()


def test_upload_completion_can_retry_after_project_update_failure(tmp_path: Path) -> None:
    root = tmp_path / "retry-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("retry"))
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=NoopWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    app = create_app(settings, services)
    content = b"retry-completion"
    original_update = repository.update
    update_attempts = 0
    successful_updates = 0

    def fail_once(mutation):  # type: ignore[no-untyped-def]
        nonlocal update_attempts, successful_updates
        update_attempts += 1
        if update_attempts == 1:
            raise OSError("simulated persistence failure")
        project = original_update(mutation)
        successful_updates += 1
        return project

    with TestClient(app, raise_server_exceptions=False) as client:
        upload_id = str(create_upload(client, content)["id"])
        chunked = client.put(
            f"/api/v1/uploads/{upload_id}/chunks/0",
            content=content,
            headers=auth_headers(),
        )
        repository.update = fail_once  # type: ignore[method-assign]

        first = client.post(
            f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
        )
        retried = client.post(
            f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
        )
        duplicate = client.post(
            f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
        )

    assert chunked.status_code == 204
    assert_stable_error(first, status_code=500, code="internal_error")
    assert retried.status_code == duplicate.status_code == 201
    assert retried.json() == duplicate.json()
    destination = root / retried.json()["path"]
    assert destination.read_bytes() == content
    assert repository.load().source_video == retried.json()["path"]
    assert len(list((root / "source").glob(f"{upload_id}-*"))) == 1
    assert update_attempts == 2
    assert successful_updates == 1


def test_multichunk_completion_recovers_after_partial_cleanup_failure(
    client_and_root, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"a" * CHUNK_LIMIT + b"tail"
    upload_id = str(create_upload(client, content)["id"])
    assert client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content[:CHUNK_LIMIT],
        headers=auth_headers(),
    ).status_code == 204
    assert client.put(
        f"/api/v1/uploads/{upload_id}/chunks/1",
        content=content[CHUNK_LIMIT:],
        headers=auth_headers(),
    ).status_code == 204
    manager = client.app.state.upload_manager
    record = manager._records[upload_id]
    lease = record.directory_lease
    assert lease is not None
    original_delete = lease.delete_if_empty
    delete_calls = 0

    def fail_first_delete() -> None:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise OSError(errno.EACCES, "simulated cleanup failure")
        original_delete()

    monkeypatch.setattr(lease, "delete_if_empty", fail_first_delete)
    failed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )
    monkeypatch.setattr(lease, "delete_if_empty", original_delete)
    retried = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )
    duplicate = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )

    assert_stable_error(failed, status_code=500, code="storage_error")
    assert retried.status_code == duplicate.status_code == 201
    assert retried.json() == duplicate.json()
    assert (root / retried.json()["path"]).read_bytes() == content
    assert not (root / ".uploads" / upload_id).exists()


def test_scratch_cleanup_never_uses_pathname_delete_after_releasing_lease(
    client_and_root, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"handle-bound-cleanup"
    upload_id = str(create_upload(client, content)["id"])
    assert client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content,
        headers=auth_headers(),
    ).status_code == 204
    manager = client.app.state.upload_manager
    allocated = root / ".uploads" / upload_id
    moved = root.parent / f"{upload_id}-cleanup-moved"
    pathname_delete_calls = 0
    original_rmdir = manager._upload_root_lease.rmdir_child

    def swap_before_pathname_delete(name: str) -> None:
        nonlocal pathname_delete_calls
        pathname_delete_calls += 1
        allocated.rename(moved)
        allocated.mkdir()
        original_rmdir(name)

    monkeypatch.setattr(
        manager._upload_root_lease, "rmdir_child", swap_before_pathname_delete
    )

    completed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )

    assert completed.status_code == 201
    assert pathname_delete_calls == 0
    assert not allocated.exists()
    assert not moved.exists()


def test_scratch_cleanup_commits_deletion_before_nonfatal_identity_refresh(
    client_and_root, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"post-delete-refresh"
    upload_id = str(create_upload(client, content)["id"])
    assert client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content,
        headers=auth_headers(),
    ).status_code == 204
    manager = client.app.state.upload_manager
    record = manager._records[upload_id]
    lease = record.directory_lease
    assert lease is not None
    original_delete = lease.delete_if_empty
    original_read_identity = uploads._read_identity
    directory_deleted = False

    def recording_delete() -> None:
        nonlocal directory_deleted
        original_delete()
        directory_deleted = True

    def fail_post_delete_refresh(
        path: Path, *, directory: bool, single_link_file: bool = False
    ):  # type: ignore[no-untyped-def]
        if directory_deleted and path == root / ".uploads":
            raise OSError(errno.EACCES, "simulated identity refresh failure")
        return original_read_identity(
            path,
            directory=directory,
            single_link_file=single_link_file,
        )

    monkeypatch.setattr(lease, "delete_if_empty", recording_delete)
    monkeypatch.setattr(uploads, "_read_identity", fail_post_delete_refresh)

    completed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )

    assert completed.status_code == 201
    assert record.scratch_removed is True
    assert not (root / ".uploads" / upload_id).exists()


def test_multichunk_cancel_recovers_after_partial_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cancel-retry-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("cancel-retry"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT * 2, 4)
    content = b"a" * CHUNK_LIMIT + b"tail"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    manager.put_chunk(created.id, 0, content[:CHUNK_LIMIT])
    manager.put_chunk(created.id, 1, content[CHUNK_LIMIT:])
    completed = manager.complete(created.id)
    record = manager._records[created.id]
    lease = record.directory_lease
    assert lease is not None
    original_delete = lease.delete_if_empty
    delete_calls = 0

    def fail_first_delete() -> None:
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise OSError(errno.EACCES, "simulated cleanup failure")
        original_delete()

    monkeypatch.setattr(lease, "delete_if_empty", fail_first_delete)
    with pytest.raises(ApiError) as cleanup_error:
        manager.cancel(created.id)
    assert cleanup_error.value.envelope.code == "storage_error"
    monkeypatch.setattr(lease, "delete_if_empty", original_delete)
    manager.cancel(created.id)
    manager.close()

    assert not (root / completed.path).exists()
    assert not (root / ".uploads" / created.id).exists()


def test_source_directory_swap_at_publish_cannot_leave_external_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source-race-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("source-race"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    content = b"source-race"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    manager.put_chunk(created.id, 0, content)
    original_copy = uploads._copy_spool_to_destination
    mutation_started = Event()
    release_mutation = Event()

    def pausing_copy(record):  # type: ignore[no-untyped-def]
        mutation_started.set()
        assert release_mutation.wait(2)
        return original_copy(record)

    monkeypatch.setattr(uploads, "_copy_spool_to_destination", pausing_copy)
    moved = root.parent / "source-moved"
    source = root / "source"
    replacement_target = root.parent / "source-replacement-target"
    replacement_target.mkdir()
    with ThreadPoolExecutor(max_workers=1) as executor:
        completion = executor.submit(manager.complete, created.id, lambda _: None)
        assert mutation_started.wait(1)
        try:
            source.rename(moved)
        except OSError:
            swap_succeeded = False
        else:
            swap_succeeded = True
            try:
                source.symlink_to(replacement_target, target_is_directory=True)
            except OSError:
                source.mkdir()
        release_mutation.set()
        if swap_succeeded:
            with pytest.raises(ApiError) as error:
                completion.result(timeout=2)
            assert error.value.envelope.code == "upload_path_changed"
            assert not list(moved.glob(f"{created.id}-*"))
            assert not list(replacement_target.glob(f"{created.id}-*"))
        else:
            completed = completion.result(timeout=2)
            assert (root / completed.path).read_bytes() == content
    manager.close()


def test_source_directory_swap_at_unlink_cannot_remove_replacement_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "unlink-race-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("unlink-race"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    content = b"unlink-race"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    manager.put_chunk(created.id, 0, content)
    record = manager._records[created.id]
    original_delete = record.destination.delete
    mutation_started = Event()
    release_mutation = Event()

    def pausing_delete() -> None:
        mutation_started.set()
        assert release_mutation.wait(2)
        original_delete()

    monkeypatch.setattr(record.destination, "delete", pausing_delete)
    allocated = root / "source"
    moved = root.parent / "source-unlink-moved"
    replacement_target = root.parent / "unlink-replacement-target"
    replacement_target.mkdir()
    with ThreadPoolExecutor(max_workers=1) as executor:
        cancellation = executor.submit(manager.cancel, created.id)
        assert mutation_started.wait(1)
        try:
            allocated.rename(moved)
        except OSError:
            swap_succeeded = False
        else:
            swap_succeeded = True
            try:
                allocated.symlink_to(replacement_target, target_is_directory=True)
            except OSError:
                allocated.mkdir()
            sentinel = replacement_target / "do-not-remove.txt"
            sentinel.write_bytes(b"replacement")
        release_mutation.set()
        if swap_succeeded:
            with pytest.raises(ApiError) as error:
                cancellation.result(timeout=2)
            assert error.value.envelope.code == "upload_path_changed"
            assert sentinel.read_bytes() == b"replacement"
            assert not list(moved.glob(f"{created.id}-*"))
        else:
            assert cancellation.result(timeout=2) is None
            assert allocated.is_dir()
            assert not list(allocated.glob(f"{created.id}-*"))
    manager.close()


def test_upload_complete_and_cancel_are_serialized_per_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "serialized-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("serialized"))
    manager = uploads.UploadManager(root, CHUNK_LIMIT, 4)
    content = b"serialized-completion"
    created = manager.create(
        uploads.UploadCreateRequest(
            filename="clip.mp4",
            mime_type="video/mp4",
            total_size=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    )
    manager.put_chunk(created.id, 0, content)
    prepare_started = Event()
    release_prepare = Event()
    original_prepare = manager._prepare

    def pausing_prepare(record):  # type: ignore[no-untyped-def]
        prepare_started.set()
        assert release_prepare.wait(2)
        return original_prepare(record)

    monkeypatch.setattr(manager, "_prepare", pausing_prepare)
    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = executor.submit(manager.complete, created.id)
        assert prepare_started.wait(1)
        cancelled = executor.submit(manager.cancel, created.id)
        assert not cancelled.done()
        release_prepare.set()
        result = completed.result(timeout=2)
        assert cancelled.result(timeout=2) is None

    assert result.path.startswith("source/")
    assert not (root / result.path).exists()
    assert not (root / ".uploads" / created.id).exists()


def test_disk_full_has_stable_non_retryable_storage_error(
    client_and_root, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root
    upload_id = create_upload(client, b"x")["id"]

    def disk_full(
        stream: object, *, offset: int, content: bytes
    ) -> None:
        del stream, offset, content
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(uploads, "_write_spool", disk_full)
    response = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0", content=b"x", headers=auth_headers()
    )

    assert response.status_code == 507
    assert response.json() == {
        "code": "storage_full",
        "category": "storage",
        "message": "Insufficient storage for upload chunk.",
        "retryable": False,
    }


def test_run_api_rejects_empty_launcher_token_before_invoking_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run_server(app: FastAPI, **kwargs: object) -> None:
        del app, kwargs
        nonlocal calls
        calls += 1

    monkeypatch.setattr(local_app, "run_server", fake_run_server)

    with pytest.raises(ValueError, match="session token"):
        local_app.run_api(object(), "")  # type: ignore[arg-type]

    assert calls == 0


def test_run_api_uses_launcher_token_and_bounded_uvicorn_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    repository = ProjectRepository(tmp_path / "run-api")
    repository.save(repository.create("run-api"))
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=NoopWorkerRegistry(),
        preview_service=UnusedPreviewService(),
    )

    def fake_run_server(app: FastAPI, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(local_app, "run_server", fake_run_server)
    monkeypatch.setattr(
        local_app,
        "assemble_api_services",
        lambda config, session_token, *, browser_origins=(): (settings, services),
    )

    exit_code = local_app.run_api(object(), TOKEN)  # type: ignore[arg-type]

    assert exit_code == 0
    api = captured.pop("app")
    assert isinstance(api, FastAPI)
    token = api.state.settings.session_token.get_secret_value()
    assert token == TOKEN
    assert captured == {
        "bind_host": "127.0.0.1",
        "port": 0,
        "startup_handshake": False,
    }
    assert token not in repr(captured)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_cli_rejects_non_loopback_serve_host() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--serve", "--host", "0.0.0.0"])

    assert error.value.code == 2
