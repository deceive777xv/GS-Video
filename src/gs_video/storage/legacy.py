from __future__ import annotations

import errno
import hashlib
import os
import shutil
import stat
from pathlib import Path
from uuid import uuid4

from gs_video.domain.models import ArtifactCategory
from gs_video.pipeline.artifacts import validate_cache_key
from gs_video.segmentation.paths import has_reparse_component
from gs_video.storage.artifacts import ArtifactStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> dict[Path, tuple[int, str]]:
    inventory: dict[Path, tuple[int, str]] = {}
    for member in root.rglob("*"):
        metadata = member.lstat()
        if has_reparse_component(member):
            raise OSError("legacy artifact tree contains a reparse point")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("legacy artifact tree contains an unsafe entry")
        inventory[member.relative_to(root)] = (metadata.st_size, _sha256(member))
    return inventory


def _remove_verified_source(source: Path, destination: Path) -> None:
    if _inventory(source) != _inventory(destination):
        raise OSError("legacy artifact verification failed")
    shutil.rmtree(source)


def _copy_across_volumes(source: Path, destination: Path) -> None:
    staging = destination.parent / f".staging-{uuid4().hex}"
    try:
        shutil.copytree(source, staging)
        if _inventory(source) != _inventory(staging):
            raise OSError("legacy artifact copy verification failed")
        os.replace(staging, destination)
        _remove_verified_source(source, destination)
    finally:
        if staging.exists() and not has_reparse_component(staging):
            shutil.rmtree(staging)


def _is_staging_name(name: str) -> bool:
    suffix = name.removeprefix(".staging-")
    return (
        name.startswith(".staging-")
        and len(suffix) == 32
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _migrate_preview_file(source: Path, destination: Path) -> None:
    metadata = source.lstat()
    if (
        has_reparse_component(source)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or len(source.stem) != 32
        or any(character not in "0123456789abcdef" for character in source.stem)
        or source.suffix != ".png"
    ):
        raise OSError("legacy preview artifact is unsafe")
    source_digest = _sha256(source)
    if destination.exists():
        target = destination.lstat()
        if (
            has_reparse_component(destination)
            or not stat.S_ISREG(target.st_mode)
            or target.st_nlink != 1
            or target.st_size != metadata.st_size
            or _sha256(destination) != source_digest
        ):
            raise OSError("legacy preview destination has different data")
        source.unlink()
        return
    try:
        os.replace(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise
        temporary = destination.with_name(f".preview-{uuid4().hex}.tmp")
        try:
            shutil.copy2(source, temporary)
            copied = temporary.lstat()
            if (
                copied.st_size != metadata.st_size
                or copied.st_nlink != 1
                or _sha256(temporary) != source_digest
            ):
                raise OSError("legacy preview copy verification failed")
            os.replace(temporary, destination)
            source.unlink()
        finally:
            temporary.unlink(missing_ok=True)


def migrate_legacy_project_artifacts(
    project_root: Path,
    project_id: str,
    artifact_store: ArtifactStore,
) -> None:
    """Move legacy project-local cache trees into the machine cache store.

    Each immutable cache-key directory is moved atomically. An interrupted run is
    therefore resumable without invalidating already successful stage records.
    """

    source_project = Path(project_root).absolute()
    destination_project = artifact_store.project_root(project_id)
    for category in ArtifactCategory:
        source_category = source_project / category.value
        if not source_category.exists():
            continue
        metadata = source_category.lstat()
        if has_reparse_component(source_category) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("legacy artifact category is unsafe")

        destination_category = destination_project / category.value
        destination_category.mkdir(exist_ok=True)
        if has_reparse_component(destination_category):
            raise OSError("artifact destination category is unsafe")

        for source in tuple(source_category.iterdir()):
            if _is_staging_name(source.name):
                staging_metadata = source.lstat()
                if (
                    has_reparse_component(source)
                    or not stat.S_ISDIR(staging_metadata.st_mode)
                ):
                    raise OSError("legacy artifact staging is unsafe")
                shutil.rmtree(source)
                continue
            if category is ArtifactCategory.PREVIEWS and source.is_file():
                _migrate_preview_file(source, destination_category / source.name)
                continue
            validate_cache_key(source.name)
            source_metadata = source.lstat()
            if has_reparse_component(source) or not stat.S_ISDIR(source_metadata.st_mode):
                raise OSError("legacy artifact entry is unsafe")
            _inventory(source)
            destination = destination_category / source.name
            if destination.exists():
                try:
                    _remove_verified_source(source, destination)
                except OSError as exc:
                    raise OSError(
                        "legacy artifact destination already exists with different data"
                    ) from exc
                continue
            try:
                os.replace(source, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
                    raise
                _copy_across_volumes(source, destination)

        source_category.rmdir()
