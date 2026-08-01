from __future__ import annotations

from collections.abc import Callable
import hashlib
import re
from pathlib import Path, PurePosixPath
import stat
import urllib.error
import urllib.request
from typing import Any, Protocol, cast
from urllib.parse import urlsplit
import zipfile

class RepairDownloadError(RuntimeError):
    """A runtime artifact could not be downloaded."""


class RepairIntegrityError(RepairDownloadError):
    """A runtime artifact did not match its manifest."""


class RepairSecurityError(RepairDownloadError):
    """A runtime URL or archive member violated the local security policy."""


class RepairCancelled(RepairDownloadError):
    """The repair manager requested cancellation."""


class ResponseLike(Protocol):
    status: int
    headers: Any

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> ResponseLike: ...

    def __exit__(self, *args: object) -> object: ...


Opener = Callable[..., ResponseLike]
Progress = Callable[[int, int | None], None]
Cancelled = Callable[[], bool]

DOWNLOAD_TIMEOUT_SECONDS = 60.0
DOWNLOAD_BLOCK_SIZE = 1024 * 1024
HASH_BLOCK_SIZE = 8 * 1024 * 1024


def _require_https(url: str, *, allowed_hosts: frozenset[str] | None = None) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme.lower() != "https"
        or parsed.username
        or parsed.password
        or (
            allowed_hosts is not None
            and (parsed.hostname is None or parsed.hostname.lower() not in allowed_hosts)
        )
    ):
        raise RepairSecurityError("runtime resources may only be downloaded over HTTPS")


class _HttpsRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _require_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = cast(Opener, urllib.request.build_opener(_HttpsRedirectHandler()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", value.strip())
    if match is None:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    if end < start or (total is not None and end >= total):
        return None
    return start, end, total


def _response_total(response: ResponseLike, resumed: int) -> int | None:
    content_range = _parse_content_range(response.headers.get("Content-Range"))
    if content_range is not None and content_range[2] is not None:
        return content_range[2]
    content_length = response.headers.get("Content-Length")
    if content_length is not None and content_length.isdigit():
        return resumed + int(content_length)
    return None


def _replace(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)


def download_verified(
    url: str,
    target: Path,
    *,
    expected_size: int | None,
    expected_sha256: str,
    progress: Progress,
    cancelled: Cancelled,
    opener: Opener = _OPENER,
    allowed_hosts: frozenset[str] | None = None,
) -> Path:
    """Download one manifest artifact with resumable partial bytes and final hash checks."""

    _require_https(url, allowed_hosts=allowed_hosts)
    if (expected_size is not None and expected_size <= 0) or len(expected_sha256) != 64:
        raise RepairIntegrityError("runtime artifact lock is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{target}.partial")
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise RepairSecurityError(f"runtime download target is not an ordinary file: {target}")
        if (expected_size is None or target.stat().st_size == expected_size) and sha256_file(target) == expected_sha256:
            progress(target.stat().st_size, expected_size or target.stat().st_size)
            return target
        target.unlink()

    if partial.exists():
        if partial.is_symlink() or not partial.is_file():
            raise RepairSecurityError(
                f"runtime partial download is not an ordinary file: {partial}"
            )
        offset = partial.stat().st_size
        if expected_size is not None and offset > expected_size:
            partial.unlink()
            offset = 0
    else:
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            _require_https(response.geturl(), allowed_hosts=allowed_hosts)
            append = offset > 0 and response.status == 206
            if append:
                content_range = _parse_content_range(response.headers.get("Content-Range"))
                if content_range is None or content_range[0] != offset:
                    raise RepairIntegrityError("resumed runtime download has an invalid range")
            resumed = offset if append else 0
            total = _response_total(response, resumed)
            downloaded = resumed
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                progress(downloaded, total or expected_size)
                while True:
                    if cancelled():
                        raise RepairCancelled("runtime repair was cancelled")
                    block = response.read(DOWNLOAD_BLOCK_SIZE)
                    if not block:
                        break
                    stream.write(block)
                    stream.flush()
                    downloaded += len(block)
                    progress(downloaded, total or expected_size)
        if expected_size is not None and (downloaded != expected_size or partial.stat().st_size != expected_size):
            raise RepairIntegrityError(
                f"runtime artifact size mismatch: expected {expected_size}, got {downloaded}"
            )
        if sha256_file(partial) != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RepairIntegrityError("runtime artifact SHA-256 mismatch")
        _replace(partial, target)
        return target
    except RepairCancelled:
        raise
    except RepairSecurityError:
        partial.unlink(missing_ok=True)
        raise
    except RepairDownloadError:
        raise
    except (OSError, urllib.error.URLError) as error:
        raise RepairDownloadError(f"runtime download failed: {error}") from error


def safe_member_path(root: Path, member: str) -> Path:
    normalized = member.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise RepairSecurityError(f"archive member escapes runtime directory: {member}")
    destination = (root / Path(*pure.parts)).resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if destination == resolved_root or resolved_root not in destination.parents:
        raise RepairSecurityError(f"archive member escapes runtime directory: {member}")
    return destination


def extract_zip(archive_path: Path, destination: Path, *, strip_prefix: str | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    prefix = None if strip_prefix is None else strip_prefix.rstrip("/") + "/"
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            normalized = info.filename.replace("\\", "/")
            if prefix is not None and not normalized.startswith(prefix):
                continue
            member = normalized[len(prefix):] if prefix is not None else normalized
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RepairSecurityError("archive contains a symbolic link")
            if info.is_dir():
                if member:
                    safe_member_path(destination, member)
                continue
            output = safe_member_path(destination, member)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, output.open("xb") as target:
                while block := source.read(HASH_BLOCK_SIZE):
                    target.write(block)


__all__ = [
    "RepairCancelled",
    "RepairDownloadError",
    "RepairIntegrityError",
    "RepairSecurityError",
    "download_verified",
    "extract_zip",
    "sha256_file",
]
