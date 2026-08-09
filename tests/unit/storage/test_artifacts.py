import os
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from gs_video.domain.models import ArtifactCategory, ArtifactRef
from gs_video.storage.artifacts import ArtifactStore


PROJECT_ID = str(uuid4())
KEY = "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", "not-a-uuid"),
        ("cache_key", "A" * 64),
        ("cache_key", "a" * 63),
        ("member", "../final.mp4"),
        ("member", "nested\\final.mp4"),
        ("member", "C:/final.mp4"),
        ("member", "/final.mp4"),
    ],
)
def test_artifact_reference_rejects_noncanonical_fields(field: str, value: str) -> None:
    payload: dict[str, object] = {
        "project_id": PROJECT_ID,
        "category": ArtifactCategory.EXPORTS,
        "cache_key": KEY,
        "member": "final.mp4",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(payload)


def test_artifact_store_publishes_and_resolves_project_scoped_tree(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache")

    reference = store.publish_tree(
        PROJECT_ID,
        ArtifactCategory.EXPORTS,
        KEY,
        lambda staging: (staging / "final.mp4").write_bytes(b"video"),
    )

    assert reference == ArtifactRef(
        project_id=PROJECT_ID,
        category=ArtifactCategory.EXPORTS,
        cache_key=KEY,
    )
    member = reference.model_copy(update={"member": "final.mp4"})
    assert store.resolve(reference, directory=True) == (
        tmp_path / "cache" / "projects" / PROJECT_ID / "exports" / KEY
    )
    assert store.resolve(member, directory=False).read_bytes() == b"video"


def test_artifact_store_rejects_hardlinked_member(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"shared")

    def build(staging: Path) -> None:
        os.link(outside, staging / "final.mp4")

    with pytest.raises(OSError, match="artifact"):
        store.publish_tree(PROJECT_ID, ArtifactCategory.EXPORTS, KEY, build)

    assert outside.read_bytes() == b"shared"


def test_artifact_store_rejects_cross_project_reference(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache")
    reference = store.publish_tree(
        PROJECT_ID,
        ArtifactCategory.PROXIES,
        KEY,
        lambda staging: (staging / "000001.jpg").write_bytes(b"frame"),
    )
    other = reference.model_copy(update={"project_id": str(uuid4())})

    with pytest.raises(OSError, match="unavailable"):
        store.resolve(other, directory=True)


def test_artifact_store_lookup_does_not_create_project_namespace(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cache")

    root = store.lookup_project_root(PROJECT_ID)

    assert root == tmp_path / "cache" / "projects" / PROJECT_ID
    assert not root.exists()
