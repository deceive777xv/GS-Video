from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path

from gs_video.domain.models import ArtifactCategory, ArtifactRef
from gs_video.pipeline.artifacts import ArtifactPublisher
from gs_video.segmentation.paths import has_reparse_component


class ArtifactStore:
    """Resolve and publish immutable cache artifacts behind logical references."""

    def __init__(self, root: Path, *, project_namespace: bool = True) -> None:
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_namespace = project_namespace
        self.projects_root = self.root / "projects" if project_namespace else self.root
        self.projects_root.mkdir(exist_ok=True)
        for directory in (self.root, self.projects_root):
            metadata = directory.lstat()
            if has_reparse_component(directory) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("artifact store root must be an ordinary directory")
        self._root_identity = self._identity(self.root)
        self._projects_identity = self._identity(self.projects_root)

    @staticmethod
    def _identity(path: Path) -> tuple[int, int, int]:
        metadata = path.lstat()
        return (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_ctime_ns))

    def _assert_roots(self) -> None:
        if (
            has_reparse_component(self.root)
            or self._identity(self.root) != self._root_identity
            or self._identity(self.projects_root) != self._projects_identity
        ):
            raise OSError("artifact store root identity changed")

    def project_root(self, project_id: str) -> Path:
        root = self.lookup_project_root(project_id)
        root.mkdir(exist_ok=True)
        if has_reparse_component(root):
            raise OSError("artifact project root is unsafe")
        return root

    def lookup_project_root(self, project_id: str) -> Path:
        reference = ArtifactRef(
            project_id=project_id,
            category=ArtifactCategory.FRAMES,
            cache_key="0" * 64,
        )
        self._assert_roots()
        root = (
            self.projects_root / reference.project_id
            if self.project_namespace
            else self.projects_root
        )
        if (
            self.project_namespace
            and root.parent != self.projects_root
            or has_reparse_component(root)
        ):
            raise OSError("artifact project root is unsafe")
        return root

    def reference(
        self,
        project_id: str,
        category: ArtifactCategory,
        cache_key: str,
        *,
        member: str | None = None,
    ) -> ArtifactRef:
        return ArtifactRef(
            project_id=project_id,
            category=category,
            cache_key=cache_key,
            member=member,
        )

    def publish_tree(
        self,
        project_id: str,
        category: ArtifactCategory,
        cache_key: str,
        build: Callable[[Path], object],
    ) -> ArtifactRef:
        project_root = self.project_root(project_id)
        ArtifactPublisher(project_root).publish_tree(category.value, cache_key, build)
        return self.reference(project_id, category, cache_key)

    def resolve(self, reference: ArtifactRef, *, directory: bool) -> Path:
        self._assert_roots()
        project_root = (
            self.projects_root / reference.project_id
            if self.project_namespace
            else self.projects_root
        )
        category_root = project_root / reference.category.value
        artifact_root = category_root / reference.cache_key
        candidate = (
            artifact_root
            if reference.member is None
            else artifact_root.joinpath(*reference.member.split("/"))
        )
        try:
            resolved_project = project_root.resolve(strict=True)
            resolved_category = category_root.resolve(strict=True)
            resolved_artifact = artifact_root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
        except OSError as exc:
            raise OSError("artifact reference is unavailable") from exc
        if (
            (self.project_namespace and resolved_project.parent != self.projects_root)
            or resolved_category.parent != resolved_project
            or resolved_artifact.parent != resolved_category
            or not resolved.is_relative_to(resolved_artifact)
            or has_reparse_component(candidate)
            or has_reparse_component(resolved)
            or (directory and not stat.S_ISDIR(metadata.st_mode))
            or (not directory and (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1))
        ):
            raise OSError("artifact reference is unsafe")
        return resolved
