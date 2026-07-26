from pathlib import Path

import pytest

from gs_video.project.repository import ProjectInstanceLock


def test_project_instance_lock_rejects_second_owner_and_releases(tmp_path: Path) -> None:
    root = tmp_path / "project"
    first = ProjectInstanceLock.acquire(root)
    try:
        with pytest.raises(RuntimeError, match="already open"):
            ProjectInstanceLock.acquire(root)
    finally:
        first.close()

    reopened = ProjectInstanceLock.acquire(root)
    reopened.close()
