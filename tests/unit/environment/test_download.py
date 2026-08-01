from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import pytest

from gs_video.environment.download import (
    RepairIntegrityError,
    RepairSecurityError,
    download_verified,
    extract_zip,
)


class Response(io.BytesIO):
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(body)
        self.status = status
        self.headers = headers or {}

    def geturl(self) -> str:
        return "https://example.test/runtime.bin"

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class Opener:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.request = None

    def __call__(self, request: object, *, timeout: float) -> Response:
        del timeout
        self.request = request
        return self.response


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_download_verified_resumes_partial_bytes(tmp_path: Path) -> None:
    target = tmp_path / "runtime.bin"
    Path(f"{target}.partial").write_bytes(b"abc")
    opener = Opener(
        Response(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})
    )

    download_verified(
        "https://example.test/runtime.bin",
        target,
        expected_size=6,
        expected_sha256=digest(b"abcdef"),
        progress=lambda _current, _total: None,
        cancelled=lambda: False,
        opener=opener,
    )

    assert target.read_bytes() == b"abcdef"
    assert opener.request is not None
    assert opener.request.get_header("Range") == "bytes=3-"  # type: ignore[attr-defined]


def test_download_verified_removes_partial_after_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "runtime.bin"
    with pytest.raises(RepairIntegrityError):
        download_verified(
            "https://example.test/runtime.bin",
            target,
            expected_size=5,
            expected_sha256=digest(b"other"),
            progress=lambda _current, _total: None,
            cancelled=lambda: False,
            opener=Opener(Response(b"wrong")),
        )

    assert not target.exists()
    assert not Path(f"{target}.partial").exists()


def test_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../../outside.txt", b"no")

    with pytest.raises(RepairSecurityError):
        extract_zip(archive_path, tmp_path / "output")
