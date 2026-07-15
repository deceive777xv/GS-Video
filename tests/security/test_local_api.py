import errno
import hashlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gs_video import __main__ as cli
from gs_video import app as local_app
from gs_video.api import uploads
from gs_video.api.routes import ApiServices
from gs_video.api.schemas import ApiSettings
from gs_video.api.uploads import CHUNK_LIMIT
from gs_video.app import create_app
from gs_video.domain.models import StageName, StageState, StageStatus
from gs_video.environment.doctor import EnvironmentReport
from gs_video.pipeline.cancellation import CancellationToken
from gs_video.project.repository import ProjectRepository


TOKEN = "security-session-token"
ORIGIN = "http://127.0.0.1:4173"


class StaticDoctor:
    def check(self) -> EnvironmentReport:
        return EnvironmentReport(ready=True, vram_mb=0, issues=[])


class SucceedingRunner:
    def run(self, name: StageName, token: CancellationToken) -> StageState:
        return StageState(status=StageStatus.SUCCEEDED)


class NoopWorkerRegistry:
    async def terminate_all(self) -> None:
        return None


def make_security_app(root: Path, *, max_active_uploads: int = 64) -> FastAPI:
    repository = ProjectRepository(root)
    repository.save(repository.create("security"))
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=NoopWorkerRegistry(),
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
        headers={"Content-Type": "application/json"},
    )

    assert_stable_error(response, status_code=401, code="invalid_session")
    assert consumed == 0


def test_outer_boundary_rejects_authorized_oversized_json_before_validation(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_root

    response = client.patch(
        "/api/v1/projects/current",
        content=b"{" + b"x" * (64 * 1024),
        headers={**auth_headers(), "Content-Type": "application/json"},
    )

    assert_stable_error(response, status_code=413, code="request_body_too_large")


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
        headers={**auth_headers(), "Content-Type": "application/json"},
    )

    assert "content-length" not in response.request.headers
    assert_stable_error(response, status_code=413, code="request_body_too_large")


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
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"identity-bound"
    upload_id = str(create_upload(client, content)["id"])
    allocated = root / ".uploads" / upload_id
    original = root / ".uploads" / f"{upload_id}.original"
    allocated.rename(original)
    allocated.mkdir()
    sentinel = allocated / "do-not-touch.txt"
    sentinel.write_bytes(b"replacement")

    written = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content,
        headers=auth_headers(),
    )
    cancelled = client.delete(
        f"/api/v1/uploads/{upload_id}", headers=auth_headers()
    )

    assert_stable_error(written, status_code=409, code="upload_path_changed")
    assert_stable_error(cancelled, status_code=409, code="upload_path_changed")
    assert sentinel.read_bytes() == b"replacement"
    assert not (original / "0.chunk").exists()


def test_upload_rejects_chunk_identity_swap_and_hard_link(
    client_and_root,
) -> None:  # type: ignore[no-untyped-def]
    client, root = client_and_root
    content = b"identity-bound-chunk"
    upload_id = str(create_upload(client, content)["id"])
    chunked = client.put(
        f"/api/v1/uploads/{upload_id}/chunks/0",
        content=content,
        headers=auth_headers(),
    )
    chunk = root / ".uploads" / upload_id / "0.chunk"
    original = chunk.with_suffix(".owned")
    chunk.rename(original)
    outside = root / "outside-owned.bin"
    outside.write_bytes(content)
    os.link(outside, chunk)

    completed = client.post(
        f"/api/v1/uploads/{upload_id}/complete", headers=auth_headers()
    )
    cancelled = client.delete(
        f"/api/v1/uploads/{upload_id}", headers=auth_headers()
    )

    assert chunked.status_code == 204
    assert_stable_error(completed, status_code=409, code="upload_path_changed")
    assert_stable_error(cancelled, status_code=409, code="upload_path_changed")
    assert outside.read_bytes() == content
    assert original.read_bytes() == content


def test_upload_completion_can_retry_after_project_save_failure(tmp_path: Path) -> None:
    root = tmp_path / "retry-project"
    repository = ProjectRepository(root)
    repository.save(repository.create("retry"))
    services = ApiServices(
        project_repository=repository,
        environment_doctor=StaticDoctor(),
        pipeline_runner=SucceedingRunner(),
        worker_registry=NoopWorkerRegistry(),
    )
    settings = ApiSettings(
        bind_host="127.0.0.1",
        port=0,
        session_token=TOKEN,
        allowed_origins=(ORIGIN,),
    )
    app = create_app(settings, services)
    content = b"retry-completion"
    original_save = repository.save
    save_attempts = 0

    def fail_once(project):  # type: ignore[no-untyped-def]
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 1:
            raise OSError("simulated persistence failure")
        original_save(project)

    with TestClient(app, raise_server_exceptions=False) as client:
        upload_id = str(create_upload(client, content)["id"])
        chunked = client.put(
            f"/api/v1/uploads/{upload_id}/chunks/0",
            content=content,
            headers=auth_headers(),
        )
        repository.save = fail_once  # type: ignore[method-assign]

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
    replace_started = Event()
    release_replace = Event()
    original_replace = uploads.os.replace

    def pausing_replace(source: object, destination: object) -> None:
        if Path(source).name == "assembled.tmp":
            replace_started.set()
            assert release_replace.wait(2)
        original_replace(source, destination)

    monkeypatch.setattr(uploads.os, "replace", pausing_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = executor.submit(manager.complete, created.id)
        assert replace_started.wait(1)
        cancelled = executor.submit(manager.cancel, created.id)
        assert not cancelled.done()
        release_replace.set()
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

    def disk_full(path: Path, content: bytes) -> None:
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(uploads, "_write_chunk_atomic", disk_full)
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


def test_run_api_validates_loopback_before_invoking_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(app: FastAPI, **kwargs: object) -> None:
        del app, kwargs
        nonlocal calls
        calls += 1

    monkeypatch.setattr(uvicorn, "run", fake_run)

    with pytest.raises(ValidationError):
        local_app.run_api("0.0.0.0", 8000)

    assert calls == 0


def test_run_api_generates_private_token_and_uses_bounded_uvicorn_options(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_run(app: FastAPI, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)

    exit_code = local_app.run_api("127.0.0.1", 0)

    assert exit_code == 0
    api = captured.pop("app")
    assert isinstance(api, FastAPI)
    token = api.state.settings.session_token.get_secret_value()
    assert len(token) >= 32
    assert captured == {
        "host": "127.0.0.1",
        "port": 0,
        "access_log": False,
        "log_config": None,
    }
    assert token not in repr(captured)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


def test_cli_rejects_non_loopback_serve_host() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--serve", "--host", "0.0.0.0"])

    assert error.value.code == 2
