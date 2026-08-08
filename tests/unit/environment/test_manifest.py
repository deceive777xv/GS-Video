from __future__ import annotations

from pathlib import Path

import pytest

from gs_video.environment.manifest import RuntimeManifest, RuntimeManifestError


MANIFEST = Path(__file__).resolve().parents[3] / "tools" / "runtime-manifest.json"


def test_checked_in_manifest_contains_pinned_edge_tam_and_ffmpeg_resources() -> None:
    manifest = RuntimeManifest.load(MANIFEST)

    source = manifest.resource("edgetam-source")
    ffmpeg = manifest.resource("ffmpeg-essentials")

    assert source.url.startswith("https://codeload.github.com/")
    assert source.size == 168774855
    assert tuple(marker.as_posix() for marker in source.markers) == ("sam2/__init__.py",)
    assert tuple(marker.as_posix() for marker in ffmpeg.markers) == (
        "bin/ffmpeg.exe",
        "bin/ffprobe.exe",
    )
    assert ffmpeg.size == 109728040
    assert ffmpeg.target.as_posix() == "data/cache/ffmpeg"
    assert any(
        action.source is not None
        and action.source.as_posix() == "segmentation/EdgeTAM"
        and action.resources == ("edgetam-source",)
        for action in manifest.actions
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["resources"][0].update({"url": "http://example.test/a.zip"}),
        lambda payload: payload["resources"][0].update({"target": "../escape"}),
        lambda payload: payload["resources"][0].update({"sha256": "not-a-hash"}),
        lambda payload: payload["resources"][0].pop("markers"),
    ],
)
def test_manifest_rejects_unsafe_resource_lock(mutation: object) -> None:
    import json

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(RuntimeManifestError):
        RuntimeManifest.from_payload(payload)
