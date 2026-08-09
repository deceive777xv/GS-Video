from pathlib import Path
from types import SimpleNamespace

import pytest

from gs_video.domain.models import SceneSummary, VideoSummary
from gs_video.project.assets import AssetKind, AssetLibrary
import gs_video.project.assets as assets_module


def video_summary(path: Path, size: int, sha256: str) -> VideoSummary:
    return VideoSummary(
        filename=path.name,
        size=size,
        sha256=sha256,
        width=1920,
        height=1080,
        duration_seconds=1.0,
        fps="30",
        has_audio=False,
        frame_count=30,
    )


def scene_summary(path: Path, size: int, sha256: str) -> SceneSummary:
    return SceneSummary(
        filename=path.name,
        size=size,
        sha256=sha256,
        gaussian_count=1,
        estimated_vram_mb=1,
    )


def test_asset_library_separates_kinds_and_deduplicates_content(tmp_path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"same-content")
    library = AssetLibrary(tmp_path / "assets")

    first, first_created = library.import_file(
        AssetKind.VIDEO, source, video_summary
    )
    duplicate, duplicate_created = library.import_file(
        AssetKind.VIDEO, source, video_summary
    )
    ply, ply_created = library.import_file(AssetKind.PLY, source, scene_summary)

    assert first_created is True
    assert duplicate_created is False
    assert duplicate.asset_id == first.asset_id
    assert ply_created is True
    assert ply.asset_id != first.asset_id
    assert [record.asset_id for record in library.list(AssetKind.VIDEO)] == [
        first.asset_id
    ]
    assert [record.asset_id for record in library.list(AssetKind.PLY)] == [
        ply.asset_id
    ]


def test_asset_library_resolves_only_expected_kind_and_deletes(tmp_path) -> None:
    source = tmp_path / "scene.ply"
    source.write_bytes(b"ply-content")
    library = AssetLibrary(tmp_path / "assets")
    record, _ = library.import_file(AssetKind.PLY, source, scene_summary)

    resolved = library.resolve(record.asset_id, AssetKind.PLY)
    assert resolved.read_bytes() == b"ply-content"
    with pytest.raises(ValueError, match="kind"):
        library.resolve(record.asset_id, AssetKind.VIDEO)

    library.delete(record.asset_id)

    assert not resolved.exists()
    with pytest.raises(KeyError):
        library.get(record.asset_id)


def test_asset_library_requires_twenty_percent_disk_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    library = AssetLibrary(tmp_path / "assets")
    monkeypatch.setattr(
        assets_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=5),
    )

    with pytest.raises(OSError, match="20% headroom"):
        library.import_file(AssetKind.VIDEO, source, video_summary)


def test_asset_delete_orphan_is_retried_on_next_startup(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"cleanup-content")
    root = tmp_path / "assets"
    library = AssetLibrary(root)
    record, _ = library.import_file(AssetKind.VIDEO, source, video_summary)
    stored = library.resolve(record.asset_id, AssetKind.VIDEO)
    original_unlink = Path.unlink

    def fail_stored(path: Path, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        if path == stored:
            raise PermissionError(path)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stored)
    with pytest.raises(PermissionError):
        library.delete(record.asset_id)
    assert library.list() == ()
    assert stored.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    AssetLibrary(root)
    assert not stored.exists()
