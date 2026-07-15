import errno
import hashlib
from pathlib import Path

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
    token = api.state.settings.session_token
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
