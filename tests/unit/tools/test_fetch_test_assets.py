from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pytest
import tools.fetch_test_assets as fetch_assets

from tools.fetch_test_assets import (
    AssetFetcher,
    AssetIntegrityError,
    AssetLock,
    AssetManifest,
    AssetOfflineError,
    AssetSecurityError,
    LockedAsset,
    default_cache_root,
    extract_selected,
    fetch_asset,
    main,
    safe_member_path,
    verify_or_remove,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


class Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        final_url: str = "https://example.test/asset.zip",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(body)
        self.status = status
        self._final_url = final_url
        self.headers = headers or {}

    def geturl(self) -> str:
        return self._final_url

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RecordingOpener:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = responses
        self.requests: list[Any] = []
        self.timeouts: list[float] = []

    def __call__(self, request: Any, *, timeout: float) -> Response:
        self.requests.append(request)
        self.timeouts.append(timeout)
        return self.responses.pop(0)


class FailIfCalledOpener:
    def __call__(self, _request: Any, *, timeout: float) -> Response:
        raise AssertionError(f"network opened with timeout={timeout}")


class ObservingResponse(Response):
    def __init__(self, body: bytes, partial: Path, progress: list[str], **kwargs: Any) -> None:
        super().__init__(body, **kwargs)
        self.partial = partial
        self.progress = progress
        self.read_count = 0

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        if self.read_count == 2:
            assert self.partial.read_bytes() == b"first"
            assert self.progress[-1] == "download asset=demo bytes=5 total=11"
        return super().read(5 if self.read_count == 1 else size)


def locked_asset(
    archive_hash: str,
    *,
    url: str = "https://example.test/asset.zip",
    members: dict[str, str] | None = None,
) -> LockedAsset:
    return LockedAsset(
        id="demo",
        group="acceptance",
        url=url,
        archive_sha256=archive_hash,
        archive_size=6,
        members=members or {},
    )


@pytest.mark.parametrize(
    "member",
    [
        "../../escape.txt",
        "../escape.txt",
        "/absolute.txt",
        "C:/drive.txt",
        "C:\\drive.txt",
        "..\\..\\escape.txt",
        "folder\\..\\..\\escape.txt",
    ],
)
def test_rejects_zip_member_outside_destination(tmp_path: Path, member: str) -> None:
    with pytest.raises(AssetSecurityError, match="越界路径"):
        safe_member_path(tmp_path / "output", member)


def test_extract_rejects_unsafe_selected_member(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "bad.zip", {"../../escape.txt": b"bad"})
    with pytest.raises(AssetSecurityError, match="越界路径"):
        extract_selected(archive, tmp_path / "output", ["**/*"])


def test_hash_mismatch_removes_download(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.write_bytes(b"changed")
    with pytest.raises(AssetIntegrityError):
        verify_or_remove(target, "0" * 64)
    assert target.exists() is False


def test_valid_cached_file_does_not_open_network(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.write_bytes(b"locked")
    fetcher = AssetFetcher(opener=FailIfCalledOpener())
    assert fetcher.fetch(locked_asset(sha256_bytes(b"locked")), target) == target


@pytest.mark.parametrize("url", ["http://example.test/a.zip", "file:///tmp/a.zip"])
def test_download_rejects_non_https_initial_url(tmp_path: Path, url: str) -> None:
    fetcher = AssetFetcher(opener=FailIfCalledOpener())
    with pytest.raises(AssetSecurityError, match="HTTPS"):
        fetcher.fetch(locked_asset(sha256_bytes(b"locked"), url=url), tmp_path / "a.zip")


def test_download_rejects_non_https_redirect_and_removes_partial(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    opener = RecordingOpener([Response(b"data", final_url="http://example.test/asset.zip")])
    with pytest.raises(AssetSecurityError, match="HTTPS"):
        AssetFetcher(opener=opener).fetch(locked_asset(sha256_bytes(b"data")), target)
    assert not target.with_suffix(".zip.partial").exists()


def test_range_resume_appends_when_server_returns_206(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    partial = target.with_suffix(".zip.partial")
    partial.write_bytes(b"abc")
    opener = RecordingOpener(
        [Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})]
    )

    AssetFetcher(opener=opener).fetch(locked_asset(sha256_bytes(b"abcdef")), target)

    assert target.read_bytes() == b"abcdef"
    assert opener.requests[0].get_header("Range") == "bytes=3-"
    assert opener.timeouts == [60.0]


@pytest.mark.parametrize(
    "content_range",
    [None, "not-a-content-range", "bytes 2-5/6"],
)
def test_range_resume_rejects_missing_malformed_or_wrong_start_content_range(
    tmp_path: Path, content_range: str | None
) -> None:
    target = tmp_path / "asset.zip"
    partial = target.with_suffix(".zip.partial")
    partial.write_bytes(b"abc")
    headers = {"Content-Range": content_range} if content_range is not None else {}
    opener = RecordingOpener([Response(b"def", status=206, headers=headers)])

    with pytest.raises(AssetIntegrityError, match="Content-Range"):
        AssetFetcher(opener=opener).fetch(locked_asset(sha256_bytes(b"abcdef")), target)

    assert partial.read_bytes() == b"abc"
    assert not target.exists()


def test_range_resume_restarts_when_server_returns_200(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.with_suffix(".zip.partial").write_bytes(b"stale")
    opener = RecordingOpener([Response(b"fresh", status=200)])

    AssetFetcher(opener=opener).fetch(locked_asset(sha256_bytes(b"fresh")), target)

    assert target.read_bytes() == b"fresh"


def test_partial_grows_and_progress_emits_before_response_completes(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    partial = target.with_suffix(".zip.partial")
    progress: list[str] = []
    response = ObservingResponse(
        b"firstsecond",
        partial,
        progress,
        headers={"Content-Length": "11"},
    )

    AssetFetcher(
        opener=RecordingOpener([response]), progress=progress.append
    ).fetch(locked_asset(sha256_bytes(b"firstsecond")), target)

    assert progress == [
        "download asset=demo status=200 resumed=0 total=11",
        "download asset=demo bytes=5 total=11",
        "download asset=demo bytes=11 total=11",
    ]


def test_resumed_progress_accounts_for_existing_bytes(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.with_suffix(".zip.partial").write_bytes(b"abc")
    progress: list[str] = []
    opener = RecordingOpener(
        [Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})]
    )

    AssetFetcher(opener=opener, progress=progress.append).fetch(
        locked_asset(sha256_bytes(b"abcdef")), target
    )

    assert progress == [
        "download asset=demo status=206 resumed=3 total=6",
        "download asset=demo bytes=6 total=6",
    ]


def test_restart_progress_resets_existing_partial_accounting(tmp_path: Path) -> None:
    target = tmp_path / "asset.zip"
    target.with_suffix(".zip.partial").write_bytes(b"stale")
    progress: list[str] = []
    opener = RecordingOpener([Response(b"fresh", status=200, headers={"Content-Length": "5"})])

    AssetFetcher(opener=opener, progress=progress.append).fetch(
        locked_asset(sha256_bytes(b"fresh")), target
    )

    assert progress == [
        "download asset=demo status=200 resumed=0 total=5",
        "download asset=demo bytes=5 total=5",
    ]


def test_extracts_only_selected_members_and_returns_hashes(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "asset.zip",
        {"keep/a.txt": b"a", "keep/b.txt": b"b", "skip/c.txt": b"c"},
    )
    hashes = extract_selected(archive, tmp_path / "out", ["keep/**"])
    assert hashes == {
        "keep/a.txt": sha256_bytes(b"a"),
        "keep/b.txt": sha256_bytes(b"b"),
    }
    assert not (tmp_path / "out" / "skip" / "c.txt").exists()


def test_fetch_asset_verifies_each_selected_member(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "source.zip", {"keep/a.txt": b"tampered"})
    cache = tmp_path / "cache"
    cache.mkdir()
    cached_archive = cache / "demo" / "archive.zip"
    cached_archive.parent.mkdir()
    cached_archive.write_bytes(archive.read_bytes())
    entry = AssetManifest.from_dict(
        {
            "schema_version": 1,
            "assets": [
                {
                    "id": "demo",
                    "group": "acceptance",
                    "url": "https://example.test/a.zip",
                    "source_page": "https://example.test/source",
                    "usage": "test",
                    "include": ["keep/**"],
                }
            ],
        }
    ).assets[0]
    lock = AssetLock(
        schema_version=1,
        assets=[
            locked_asset(
                sha256_bytes(archive.read_bytes()),
                url="https://example.test/a.zip",
                members={"keep/a.txt": "0" * 64},
            )
        ],
    )
    selected = cache / "demo" / "selected" / "keep" / "a.txt"
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"trusted")
    with pytest.raises(AssetIntegrityError, match="keep/a.txt"):
        fetch_asset(entry, lock, cache, opener=FailIfCalledOpener())
    assert selected.read_bytes() == b"trusted"
    assert not (cache / "demo" / "selected.partial").exists()


def test_redirect_handler_rejects_https_to_http_hop_before_a_later_https_hop() -> None:
    handler = fetch_assets.HTTPSOnlyRedirectHandler()
    request = fetch_assets.urllib.request.Request("https://example.test/start")

    with pytest.raises(AssetSecurityError, match="HTTPS"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://example.test/intermediate",
        )


def test_replace_with_retry_recovers_from_transient_windows_sharing_violation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "asset.zip.partial"
    target = tmp_path / "asset.zip"
    source.write_bytes(b"complete")
    attempts = 0
    sleeps: list[float] = []

    def transient_replace(old: Path, new: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(32, "sharing violation")
        old.replace(new)

    fetch_assets.replace_with_retry(
        source,
        target,
        replace=transient_replace,
        sleep=sleeps.append,
        attempts=4,
        delay=0.01,
    )

    assert target.read_bytes() == b"complete"
    assert attempts == 3
    assert sleeps == [0.01, 0.01]


def test_replace_with_retry_is_bounded(tmp_path: Path) -> None:
    source = tmp_path / "asset.zip.partial"
    target = tmp_path / "asset.zip"
    source.write_bytes(b"complete")
    attempts = 0

    def always_locked(_old: Path, _new: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError(32, "sharing violation")

    with pytest.raises(PermissionError):
        fetch_assets.replace_with_retry(
            source,
            target,
            replace=always_locked,
            sleep=lambda _delay: None,
            attempts=3,
        )

    assert attempts == 3
    assert source.read_bytes() == b"complete"


def test_offline_missing_cache_fails_without_network(tmp_path: Path) -> None:
    fetcher = AssetFetcher(opener=FailIfCalledOpener(), offline=True)
    with pytest.raises(AssetOfflineError, match="离线"):
        fetcher.fetch(locked_asset("0" * 64), tmp_path / "missing.zip")


def test_default_cache_root_honors_environment_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GS_VIDEO_TEST_ASSETS", str(tmp_path))
    assert default_cache_root() == tmp_path


def test_lock_requires_explicit_source_review_acknowledgement(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["lock", "--group", "smoke"])
    assert exc.value.code == 2
    assert "--acknowledge-source-review" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["fetch", "lock"])
@pytest.mark.parametrize("group", ["smoke", "acceptance"])
def test_cli_accepts_both_asset_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    group: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    manifest.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    lock.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    monkeypatch.setattr("tools.fetch_test_assets.MANIFEST_PATH", manifest)
    monkeypatch.setattr("tools.fetch_test_assets.LOCK_PATH", lock)
    args = [command, "--group", group]
    if command == "lock":
        args.append("--acknowledge-source-review")
    assert main(args) == 0
    assert f"group={group}" in capsys.readouterr().out


def test_dry_run_reports_size_cache_and_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "manifest.json"
    lock = tmp_path / "lock.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "demo",
                        "group": "acceptance",
                        "url": "https://example.test/a.zip",
                        "source_page": "https://example.test/source",
                        "usage": "test",
                        "include": ["keep/**"],
                        "expected_size": 123,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    lock.write_text(json.dumps({"schema_version": 1, "assets": []}), encoding="utf-8")
    monkeypatch.setattr("tools.fetch_test_assets.MANIFEST_PATH", manifest)
    monkeypatch.setattr("tools.fetch_test_assets.LOCK_PATH", lock)
    monkeypatch.setenv("GS_VIDEO_TEST_ASSETS", str(tmp_path / "cache"))
    monkeypatch.setattr(
        "tools.fetch_test_assets.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(1000, 100, 900),
    )

    assert main(["fetch", "--group", "acceptance", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "123" in output
    assert str(tmp_path / "cache") in output
    assert "900" in output
