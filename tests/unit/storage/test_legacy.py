import errno
import os
from pathlib import Path
from uuid import uuid4

import pytest

from gs_video.domain.models import ArtifactCategory
from gs_video.storage.artifacts import ArtifactStore
from gs_video.storage.legacy import migrate_legacy_project_artifacts


_KEY = "a" * 64


def test_migrate_legacy_project_artifacts_moves_cache_key_trees(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "frames" / _KEY
    source.mkdir(parents=True)
    (source / "000001.png").write_bytes(b"frame")
    store = ArtifactStore(tmp_path / "cache")
    project_id = str(uuid4())

    migrate_legacy_project_artifacts(project_root, project_id, store)

    reference = store.reference(project_id, ArtifactCategory.FRAMES, _KEY)
    migrated = store.resolve(reference, directory=True)
    assert (migrated / "000001.png").read_bytes() == b"frame"
    assert not (project_root / "frames").exists()


def test_migrate_legacy_project_artifacts_is_resumable(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    store = ArtifactStore(tmp_path / "cache")

    migrate_legacy_project_artifacts(project_root, str(uuid4()), store)


def test_migrate_legacy_project_artifacts_rejects_collision(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    source = project_root / "frames" / _KEY
    source.mkdir(parents=True)
    store = ArtifactStore(tmp_path / "cache")
    project_id = str(uuid4())
    destination = store.project_root(project_id) / "frames" / _KEY
    destination.mkdir(parents=True)
    (source / "frame.png").write_bytes(b"source")
    (destination / "frame.png").write_bytes(b"different")

    with pytest.raises(OSError, match="different data"):
        migrate_legacy_project_artifacts(project_root, project_id, store)


def test_migrate_legacy_project_artifacts_recovers_completed_move(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "frames" / _KEY
    source.mkdir(parents=True)
    (source / "frame.png").write_bytes(b"same")
    store = ArtifactStore(tmp_path / "cache")
    project_id = str(uuid4())
    destination = store.project_root(project_id) / "frames" / _KEY
    destination.mkdir(parents=True)
    (destination / "frame.png").write_bytes(b"same")

    migrate_legacy_project_artifacts(project_root, project_id, store)

    assert not (project_root / "frames").exists()
    assert (destination / "frame.png").read_bytes() == b"same"


def test_migrate_legacy_project_artifacts_copies_across_volumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    source = project_root / "frames" / _KEY
    source.mkdir(parents=True)
    (source / "frame.png").write_bytes(b"cross-volume")
    store = ArtifactStore(tmp_path / "cache")
    project_id = str(uuid4())
    destination = store.project_root(project_id) / "frames" / _KEY
    real_replace = os.replace

    def replace_with_cross_volume_error(source_path: Path, target_path: Path) -> None:
        if Path(source_path) == source and Path(target_path) == destination:
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(source_path, target_path)

    monkeypatch.setattr(os, "replace", replace_with_cross_volume_error)

    migrate_legacy_project_artifacts(project_root, project_id, store)

    assert not source.exists()
    assert (destination / "frame.png").read_bytes() == b"cross-volume"


def test_migrate_legacy_preview_files_and_abandoned_staging(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    preview_root = project_root / "previews"
    preview_root.mkdir(parents=True)
    preview_id = "b" * 32
    (preview_root / f"{preview_id}.png").write_bytes(b"preview")
    staging = preview_root / f".staging-{'c' * 32}"
    staging.mkdir()
    (staging / "partial.png").write_bytes(b"partial")
    store = ArtifactStore(tmp_path / "cache")
    project_id = str(uuid4())

    migrate_legacy_project_artifacts(project_root, project_id, store)

    destination = store.project_root(project_id) / "previews" / f"{preview_id}.png"
    assert destination.read_bytes() == b"preview"
    assert not preview_root.exists()
