from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "assets" / "manifest.json"
LOCK_PATH = REPOSITORY_ROOT / "tests" / "assets" / "lock.json"
DOWNLOAD_TIMEOUT_SECONDS = 60.0
HASH_BLOCK_SIZE = 8 * 1024 * 1024


class AssetError(RuntimeError):
    """Base error for acceptance assets."""


class AssetSecurityError(AssetError):
    """A URL or archive member violated the security policy."""


class AssetIntegrityError(AssetError):
    """Downloaded or extracted bytes did not match the lock."""


class AssetOfflineError(AssetError):
    """An offline fetch needed unavailable cached bytes."""


class AssetNeedsContextError(AssetError):
    """The reviewed remote source no longer has the expected structure."""


@dataclass(frozen=True)
class ManifestAsset:
    id: str
    group: str
    url: str
    source_page: str
    usage: str
    include: tuple[str, ...]
    citation: str | None = None
    license_url: str | None = None
    expected_size: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ManifestAsset:
        return cls(
            id=str(data["id"]),
            group=str(data["group"]),
            url=str(data["url"]),
            source_page=str(data["source_page"]),
            usage=str(data["usage"]),
            include=tuple(str(pattern) for pattern in data["include"]),
            citation=str(data["citation"]) if data.get("citation") else None,
            license_url=str(data["license_url"]) if data.get("license_url") else None,
            expected_size=int(data["expected_size"]) if data.get("expected_size") else None,
        )


@dataclass(frozen=True)
class AssetManifest:
    schema_version: int
    assets: list[ManifestAsset]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetManifest:
        if int(data["schema_version"]) != 1:
            raise AssetError("不支持的素材清单版本")
        return cls(1, [ManifestAsset.from_dict(item) for item in data["assets"]])

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> AssetManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LockedAsset:
    id: str
    group: str
    url: str
    archive_sha256: str
    archive_size: int
    members: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LockedAsset:
        return cls(
            id=str(data["id"]),
            group=str(data["group"]),
            url=str(data["url"]),
            archive_sha256=str(data["archive_sha256"]),
            archive_size=int(data["archive_size"]),
            members={str(name): str(digest) for name, digest in data["members"].items()},
        )


@dataclass(frozen=True)
class AssetLock:
    schema_version: int
    assets: list[LockedAsset]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetLock:
        if int(data["schema_version"]) != 1:
            raise AssetError("不支持的素材锁版本")
        return cls(1, [LockedAsset.from_dict(item) for item in data["assets"]])

    @classmethod
    def load(cls, path: Path = LOCK_PATH) -> AssetLock:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def for_id(self, asset_id: str) -> LockedAsset:
        try:
            return next(asset for asset in self.assets if asset.id == asset_id)
        except StopIteration as exc:
            raise AssetIntegrityError(f"锁文件缺少素材: {asset_id}") from exc

    def write(self, path: Path = LOCK_PATH) -> None:
        payload = {"schema_version": self.schema_version, "assets": [asdict(a) for a in self.assets]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class ResponseLike(Protocol):
    status: int

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> ResponseLike: ...

    def __exit__(self, *args: object) -> object: ...


Opener = Callable[..., ResponseLike]


def _require_https(url: str) -> None:
    if urlparse(url).scheme.lower() != "https":
        raise AssetSecurityError(f"测试素材只允许 HTTPS URL: {url}")


def _normalized_member(member: str) -> str:
    return member.replace("\\", "/")


def safe_member_path(root: Path, member: str) -> Path:
    normalized = _normalized_member(member)
    pure = PurePosixPath(normalized)
    if (
        not normalized
        or pure.is_absolute()
        or any(part == ".." for part in pure.parts)
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise AssetSecurityError(f"压缩包包含越界路径: {member}")
    destination = (root / Path(*pure.parts)).resolve()
    resolved_root = root.resolve()
    if destination == resolved_root or resolved_root not in destination.parents:
        raise AssetSecurityError(f"压缩包包含越界路径: {member}")
    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_or_remove(path: Path, expected_sha256: str) -> Path:
    actual = sha256_file(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise AssetIntegrityError(
            f"素材哈希不匹配，已删除: {path} (expected={expected_sha256}, actual={actual})"
        )
    return path


def default_cache_root() -> Path:
    override = os.environ.get("GS_VIDEO_TEST_ASSETS")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "GS-Video" / "TestAssets"
    return Path.home() / ".cache" / "GS-Video" / "TestAssets"


def _partial_path(target: Path) -> Path:
    return Path(f"{target}.partial")


class AssetFetcher:
    def __init__(self, *, opener: Opener = urllib.request.urlopen, offline: bool = False) -> None:
        self._opener = opener
        self._offline = offline

    def fetch(self, lock: LockedAsset, target: Path) -> Path:
        _require_https(lock.url)
        if target.exists():
            return verify_or_remove(target, lock.archive_sha256)
        if self._offline:
            raise AssetOfflineError(f"离线模式下缓存缺失: {target}")
        self.download(lock.url, target)
        return verify_or_remove(target, lock.archive_sha256)

    def download(self, url: str, target: Path) -> Path:
        _require_https(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = _partial_path(target)
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with self._opener(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                _require_https(response.geturl())
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(HASH_BLOCK_SIZE)
                        if not block:
                            break
                        stream.write(block)
            partial.replace(target)
            return target
        except Exception:
            if partial.exists() and isinstance(sys.exc_info()[1], AssetSecurityError):
                partial.unlink(missing_ok=True)
            raise


def _matches(member: str, patterns: Sequence[str]) -> bool:
    normalized = _normalized_member(member).rstrip("/")
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def extract_selected(archive_path: Path, destination: Path, include: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            name = _normalized_member(info.filename)
            if info.is_dir() or not _matches(name, include):
                continue
            output = safe_member_path(destination, name)
            output.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with archive.open(info) as source, output.open("wb") as target:
                while True:
                    block = source.read(HASH_BLOCK_SIZE)
                    if not block:
                        break
                    digest.update(block)
                    target.write(block)
            hashes[name] = digest.hexdigest()
    return hashes


def fetch_asset(
    entry: ManifestAsset,
    lock: AssetLock,
    cache_root: Path,
    *,
    opener: Opener = urllib.request.urlopen,
    offline: bool = False,
) -> list[Path]:
    locked = lock.for_id(entry.id)
    if locked.url != entry.url or locked.group != entry.group:
        raise AssetIntegrityError(f"清单与锁文件不一致: {entry.id}")
    asset_root = cache_root / entry.id
    archive_path = AssetFetcher(opener=opener, offline=offline).fetch(
        locked, asset_root / "archive.zip"
    )
    extracted_root = asset_root / "selected"
    actual = extract_selected(archive_path, extracted_root, entry.include)
    if actual != locked.members:
        unexpected = sorted(set(actual) ^ set(locked.members))
        mismatched = sorted(
            name for name in set(actual) & set(locked.members) if actual[name] != locked.members[name]
        )
        names = unexpected + mismatched
        raise AssetIntegrityError(f"锁定成员哈希不匹配: {entry.id}: {', '.join(names)}")
    return [safe_member_path(extracted_root, name) for name in sorted(actual)]


def _selected(manifest: AssetManifest, group: str) -> list[ManifestAsset]:
    return [asset for asset in manifest.assets if asset.group == group]


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    return f"{size} bytes ({size / (1024**3):.2f} GiB)"


def _dry_run(manifest: AssetManifest, lock: AssetLock, group: str, cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_root).free
    locked_by_id = {asset.id: asset for asset in lock.assets}
    selected = _selected(manifest, group)
    total = 0
    complete = True
    print(f"dry-run group={group}")
    print(f"cache={cache_root}")
    for entry in selected:
        locked = locked_by_id.get(entry.id)
        size = locked.archive_size if locked else entry.expected_size
        print(f"asset={entry.id} estimated-download={_format_bytes(size)}")
        if size is None:
            complete = False
        else:
            total += size
    print(f"estimated-total={_format_bytes(total) if complete else 'unknown'}")
    print(f"free-space={_format_bytes(free)}")
    if complete and total > free:
        raise AssetError("缓存磁盘剩余空间不足")
    if not selected:
        print(f"no assets selected; group={group} is a no-op")


def _lock_assets(
    manifest: AssetManifest,
    current_lock: AssetLock,
    group: str,
    cache_root: Path,
) -> AssetLock:
    _dry_run(manifest, current_lock, group, cache_root)
    selected_ids = {asset.id for asset in _selected(manifest, group)}
    retained = [asset for asset in current_lock.assets if asset.id not in selected_ids]
    generated: list[LockedAsset] = []
    fetcher = AssetFetcher()
    for entry in _selected(manifest, group):
        print(f"locking asset={entry.id}")
        asset_root = cache_root / entry.id
        archive_path = asset_root / "archive.zip"
        if not archive_path.exists():
            fetcher.download(entry.url, archive_path)
        hashes = extract_selected(archive_path, asset_root / "selected", entry.include)
        if not hashes:
            raise AssetNeedsContextError(
                f"NEEDS_CONTEXT: reviewed members were not found in remote archive: {entry.id}"
            )
        generated.append(
            LockedAsset(
                id=entry.id,
                group=entry.group,
                url=entry.url,
                archive_sha256=sha256_file(archive_path),
                archive_size=archive_path.stat().st_size,
                members=hashes,
            )
        )
    return AssetLock(schema_version=1, assets=sorted(retained + generated, key=lambda a: a.id))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch locked network acceptance assets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("--group", choices=("smoke", "acceptance"), required=True)
    fetch.add_argument("--offline", action="store_true")
    fetch.add_argument("--dry-run", action="store_true")
    lock = subparsers.add_parser("lock")
    lock.add_argument("--group", choices=("smoke", "acceptance"), default="acceptance")
    lock.add_argument("--acknowledge-source-review", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = AssetManifest.load(MANIFEST_PATH)
    lock = AssetLock.load(LOCK_PATH)
    cache_root = default_cache_root()
    if args.command == "fetch":
        if args.dry_run:
            _dry_run(manifest, lock, args.group, cache_root)
            return 0
        selected = _selected(manifest, args.group)
        if not selected:
            print(f"no assets selected; group={args.group} is a no-op")
            return 0
        for entry in selected:
            paths = fetch_asset(entry, lock, cache_root, offline=args.offline)
            print(f"verified asset={entry.id} members={len(paths)}")
        return 0
    updated = _lock_assets(manifest, lock, args.group, cache_root)
    updated.write(LOCK_PATH)
    print(f"lock updated; group={args.group}; assets={len(_selected(manifest, args.group))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
