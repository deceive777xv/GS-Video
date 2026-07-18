import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from gs_video.pipeline.artifacts import ArtifactPublisher


GOOD_KEY = "a" * 64
OTHER_KEY = "b" * 64


def _write_frame(staging: Path, payload: bytes = b"frame") -> None:
    (staging / "000001.png").write_bytes(payload)


def test_artifact_publisher_publishes_owned_tree_and_reuses_valid_generation(
    tmp_path: Path,
) -> None:
    publisher = ArtifactPublisher(tmp_path)
    published = publisher.publish_tree("proxies", GOOD_KEY, _write_frame)

    assert published == tmp_path / "proxies" / GOOD_KEY
    assert (published / "000001.png").read_bytes() == b"frame"

    called = False

    def unexpected_builder(staging: Path) -> None:
        nonlocal called
        called = True
        _write_frame(staging, b"replacement")

    assert publisher.publish_tree("proxies", GOOD_KEY, unexpected_builder) == published
    assert called is False
    assert (published / "000001.png").read_bytes() == b"frame"


def test_artifact_publisher_keeps_old_generation_when_builder_fails(
    tmp_path: Path,
) -> None:
    publisher = ArtifactPublisher(tmp_path)
    published = publisher.publish_tree(
        "proxies", GOOD_KEY, lambda staging: _write_frame(staging, b"old")
    )

    def fail(staging: Path) -> None:
        _write_frame(staging, b"partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        publisher.publish_tree("proxies", OTHER_KEY, fail)

    assert (published / "000001.png").read_bytes() == b"old"
    assert not (tmp_path / "proxies" / OTHER_KEY).exists()
    assert not tuple((tmp_path / "proxies").glob(".staging-*"))


@pytest.mark.parametrize(
    ("category", "cache_key"),
    [
        ("../proxies", GOOD_KEY),
        ("source", GOOD_KEY),
        ("proxies/child", GOOD_KEY),
        ("proxies", "A" * 64),
        ("proxies", "a" * 63),
        ("proxies", "../" + "a" * 61),
    ],
)
def test_artifact_publisher_rejects_unowned_category_or_invalid_cache_key(
    tmp_path: Path,
    category: str,
    cache_key: str,
) -> None:
    with pytest.raises(ValueError):
        ArtifactPublisher(tmp_path).publish_tree(category, cache_key, _write_frame)


def test_artifact_publisher_rejects_invalid_existing_generation_without_deleting_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "proxies" / GOOD_KEY
    destination.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"shared")
    os.link(outside, destination / "000001.png")

    with pytest.raises(OSError, match="artifact"):
        ArtifactPublisher(tmp_path).publish_tree("proxies", GOOD_KEY, _write_frame)

    assert outside.read_bytes() == b"shared"
    assert (destination / "000001.png").exists()


def test_artifact_publisher_rejects_builder_hardlinks_and_removes_only_staging(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"shared")

    def hardlink(staging: Path) -> None:
        os.link(outside, staging / "000001.png")

    with pytest.raises(OSError, match="artifact"):
        ArtifactPublisher(tmp_path).publish_tree("proxies", GOOD_KEY, hardlink)

    assert outside.read_bytes() == b"shared"
    assert not (tmp_path / "proxies" / GOOD_KEY).exists()
    assert not tuple((tmp_path / "proxies").glob(".staging-*"))


def test_artifact_publisher_keeps_other_cache_keys_when_publication_succeeds(
    tmp_path: Path,
) -> None:
    publisher = ArtifactPublisher(tmp_path)
    first = publisher.publish_tree("proxies", GOOD_KEY, _write_frame)
    second = publisher.publish_tree(
        "proxies", OTHER_KEY, lambda staging: _write_frame(staging, b"new")
    )

    assert (first / "000001.png").read_bytes() == b"frame"
    assert (second / "000001.png").read_bytes() == b"new"


def test_artifact_publisher_reuses_valid_winner_of_concurrent_publication(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)

    def publish(payload: bytes) -> Path:
        def build(staging: Path) -> None:
            _write_frame(staging, payload)
            barrier.wait(timeout=5)

        return ArtifactPublisher(tmp_path).publish_tree("proxies", GOOD_KEY, build)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(publish, b"first")
        second = executor.submit(publish, b"second")

    assert first.result() == second.result() == tmp_path / "proxies" / GOOD_KEY
    assert (first.result() / "000001.png").read_bytes() in {b"first", b"second"}
    assert not tuple((tmp_path / "proxies").glob(".staging-*"))
