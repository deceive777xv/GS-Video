from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gs_video.segmentation.paths import has_reparse_component
from gs_video.storage.settings import read_user_settings, write_user_settings


_MARKER = ".gs-video-storage.json"
_COPY_CHUNK = 4 * 1024 * 1024
_LOWER_HEX = frozenset("0123456789abcdef")
_MIGRATION_OPERATION = ".gs-video-migration.json"
_MIGRATION_STAGING = ".gs-video-migration-staging"


class StorageKind(StrEnum):
    PROJECT_LIBRARY = "project_library"
    CACHE = "cache"


class ProjectLibraryAction(StrEnum):
    MIGRATE = "migrate"
    OPEN_EXISTING = "open_existing"


class CacheAction(StrEnum):
    START_FRESH = "start_fresh"
    MIGRATE = "migrate"


class CacheCleanupMode(StrEnum):
    SAFE = "safe"
    DEEP = "deep"


class StorageMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    storage_id: str
    kind: StorageKind


class StorageLayoutPreference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    project_library_root: str
    project_library_id: str
    cache_root: str
    cache_id: str


class StorageLayoutStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    project_library_root: str
    project_library_id: str
    cache_root: str
    cache_id: str
    restart_required: bool = False
    editable: bool = True
    blocked_reason: str | None = None


class StorageLayoutSnapshot(StorageLayoutStatus):
    project_library_bytes: int = Field(default=0, ge=0)
    project_library_free_bytes: int = Field(default=0, ge=0)
    cache_bytes: int = Field(default=0, ge=0)
    cache_free_bytes: int = Field(default=0, ge=0)


class CacheCleanupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    mode: CacheCleanupMode
    removed_entries: int = Field(ge=0)
    freed_bytes: int = Field(ge=0)
    storage: StorageLayoutSnapshot


class CacheCleanupPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    plan_token: str
    mode: CacheCleanupMode
    removable_entries: int = Field(ge=0)
    reclaimable_bytes: int = Field(ge=0)
    expires_in_seconds: int = Field(ge=1)


DriveTypeProbe = Callable[[Path], int]


def _windows_drive_type(path: Path) -> int:
    if os.name != "nt":
        return 3
    drive = path.drive
    if not drive:
        return 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    probe = kernel32.GetDriveTypeW
    probe.argtypes = [ctypes.c_wchar_p]
    probe.restype = ctypes.c_uint
    return int(probe(f"{drive}\\"))


def _canonical_uuid(value: str) -> str:
    parsed = UUID(value)
    if str(parsed) != value:
        raise ValueError("storage id must be a canonical UUID")
    return value


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def _is_publisher_staging(name: str) -> bool:
    suffix = name.removeprefix(".staging-")
    return (
        name.startswith(".staging-")
        and len(suffix) == 32
        and all(character in _LOWER_HEX for character in suffix)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _tree_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        metadata = path.lstat()
        if has_reparse_component(path):
            raise OSError("storage tree contains a reparse point")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise OSError("storage tree contains a hard-linked file")
            total += int(metadata.st_size)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise OSError("storage tree contains an unsupported entry")
    return total


def _tree_inventory(root: Path, *, ignored: frozenset[str] = frozenset()) -> dict[Path, tuple[int, str]]:
    inventory: dict[Path, tuple[int, str]] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in ignored:
            continue
        metadata = path.lstat()
        if has_reparse_component(path):
            raise OSError("storage tree contains a reparse point")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("storage tree contains an unsafe entry")
        inventory[relative] = (int(metadata.st_size), _sha256(path))
    return inventory


def _metadata_inventory(
    root: Path,
) -> list[tuple[str, str, int, int, int]]:
    inventory: list[tuple[str, str, int, int, int]] = []
    for path in root.rglob("*"):
        metadata = path.lstat()
        if has_reparse_component(path):
            raise OSError("storage tree contains a reparse point")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = "file"
        else:
            raise OSError("storage tree contains an unsafe entry")
        inventory.append(
            (
                path.relative_to(root).as_posix(),
                kind,
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_ctime_ns),
            )
        )
    inventory.sort()
    return inventory


def _path_identity(path: Path) -> tuple[int, int, int]:
    metadata = path.lstat()
    return (int(metadata.st_dev), int(metadata.st_ino), int(metadata.st_ctime_ns))


def _remove_owned_path(path: Path) -> None:
    metadata = path.lstat()
    if has_reparse_component(path):
        raise OSError("migration staging contains a reparse point")
    if stat.S_ISDIR(metadata.st_mode):
        _tree_inventory(path)
        shutil.rmtree(path)
    elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
        path.unlink()
    else:
        raise OSError("migration staging contains an unsafe entry")


class StorageLayoutManager:
    """Owns machine-level project-library and cache-root authority."""

    def __init__(
        self,
        default_container: Path,
        preference_path: Path,
        *,
        protected_roots: tuple[Path, ...] = (),
        drive_type_probe: DriveTypeProbe = _windows_drive_type,
    ) -> None:
        self.default_container = Path(default_container).absolute()
        self.preference_path = Path(preference_path).absolute()
        self.protected_roots = tuple(path.absolute() for path in protected_roots)
        self._drive_type_probe = drive_type_probe
        self._restart_required = False
        self._cleanup_lock = RLock()
        self._cleanup_plans: dict[str, tuple[float, CacheCleanupMode, str]] = {}
        self._preference = self._load_or_initialize()

    @property
    def restart_required(self) -> bool:
        return self._restart_required

    def _validate_root(self, value: Path, *, create: bool) -> Path:
        root = Path(value)
        if not root.is_absolute():
            raise ValueError("storage root must be absolute")
        root = root.absolute()
        if self._drive_type_probe(root) != 3:
            raise ValueError("storage root must be on a local fixed drive")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        try:
            metadata = root.lstat()
            resolved = root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("storage root is unavailable") from exc
        if (
            resolved != root
            or has_reparse_component(root)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ValueError("storage root must be an ordinary directory")
        for protected in self.protected_roots:
            try:
                protected_resolved = protected.resolve(strict=False)
            except OSError:
                protected_resolved = protected
            if _contains(root, protected_resolved) or _contains(protected_resolved, root):
                raise ValueError("storage root overlaps a protected application path")
        return root

    def _validate_pair(self, project_root: Path, cache_root: Path) -> None:
        if _contains(project_root, cache_root) or _contains(cache_root, project_root):
            raise ValueError("project library and cache roots must not overlap")

    @staticmethod
    def _marker_path(root: Path) -> Path:
        return root / _MARKER

    def _read_marker(self, root: Path, expected: StorageKind) -> StorageMarker:
        path = self._marker_path(root)
        try:
            marker = StorageMarker.model_validate_json(path.read_bytes())
            _canonical_uuid(marker.storage_id)
        except (OSError, ValueError, ValidationError) as exc:
            raise ValueError("storage root marker is missing or invalid") from exc
        if marker.kind is not expected:
            raise ValueError("storage root kind does not match its marker")
        return marker

    def _initialize_root(self, root: Path, kind: StorageKind) -> StorageMarker:
        marker = StorageMarker(storage_id=str(uuid4()), kind=kind)
        path = self._marker_path(root)
        if path.exists():
            return self._read_marker(root, kind)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(marker.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return marker

    def _save_preference(self, preference: StorageLayoutPreference) -> None:
        settings, invalid = read_user_settings(self.preference_path)
        if invalid:
            settings = {}
        if "vram_budget" not in settings and {
            "mode",
            "selected_vram_mb",
        }.issubset(settings):
            settings["vram_budget"] = {
                "mode": settings.pop("mode"),
                "selected_vram_mb": settings.pop("selected_vram_mb"),
            }
        settings["schema_version"] = 1
        settings["storage_layout"] = preference.model_dump(mode="json")
        write_user_settings(self.preference_path, settings)

    def _preference_from_roots(
        self, project_root: Path, cache_root: Path
    ) -> StorageLayoutPreference:
        project_marker = self._read_marker(project_root, StorageKind.PROJECT_LIBRARY)
        cache_marker = self._read_marker(cache_root, StorageKind.CACHE)
        return StorageLayoutPreference(
            project_library_root=str(project_root),
            project_library_id=project_marker.storage_id,
            cache_root=str(cache_root),
            cache_id=cache_marker.storage_id,
        )

    def _validate_existing_project_library(self, root: Path) -> None:
        from gs_video.project.assets import AssetIndex, AssetKind
        from gs_video.project.catalog import ProjectCatalogState
        from gs_video.project.repository import ProjectRepository

        self._read_marker(root, StorageKind.PROJECT_LIBRARY)
        _tree_inventory(root, ignored=frozenset({_MARKER}))
        try:
            catalog = ProjectCatalogState.model_validate_json(
                (root / "catalog.json").read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise ValueError("existing project catalog is invalid") from exc
        if len(set(catalog.project_ids)) != len(catalog.project_ids) or (
            catalog.active_project_id is not None
            and catalog.active_project_id not in catalog.project_ids
        ):
            raise ValueError("existing project catalog authority is invalid")
        projects_root = root / "projects"
        projects = []
        for project_id in catalog.project_ids:
            _canonical_uuid(project_id)
            project = ProjectRepository(projects_root / project_id).load()
            if project.project_id != project_id:
                raise ValueError("existing project identity does not match its catalog")
            projects.append(project)

        asset_index_path = root / "assets" / "index.json"
        if not asset_index_path.is_file():
            raise ValueError("existing asset index is missing")
        index = AssetIndex.model_validate_json(
            asset_index_path.read_text(encoding="utf-8")
        )
        for asset_id, record in index.assets.items():
            if asset_id != record.asset_id:
                raise ValueError("existing asset identity is invalid")
            relative = Path(record.stored_relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("existing asset path is unsafe")
            asset = root / "assets" / record.kind.value / relative
            metadata = asset.lstat()
            if (
                has_reparse_component(asset)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != record.size
                or _sha256(asset) != record.sha256
            ):
                raise ValueError("existing asset content does not match its index")
        for project in projects:
            required = (
                (project.source_video_asset_id, AssetKind.VIDEO),
                (project.scene_ply_asset_id, AssetKind.PLY),
            )
            for required_asset_id, kind in required:
                if required_asset_id is None:
                    continue
                required_record = index.assets.get(required_asset_id)
                if required_record is None or required_record.kind is not kind:
                    raise ValueError("existing project asset authority is invalid")

    def _adopt_default_layout(self) -> StorageLayoutPreference:
        container = self.default_container
        container.mkdir(parents=True, exist_ok=True)
        project_root = self._validate_root(container / "project-library", create=True)
        cache_root = self._validate_root(container / "cache-library", create=True)
        self._validate_pair(project_root, cache_root)

        for name in ("catalog.json", "projects", "assets"):
            source = container / name
            destination = project_root / name
            if not source.exists():
                continue
            if destination.exists():
                raise ValueError("legacy project-library destination is not empty")
            if has_reparse_component(source):
                raise ValueError("legacy project-library entry is unsafe")
            metadata = source.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                _tree_inventory(source)
            elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("legacy project-library entry is unsafe")
            os.replace(source, destination)

        self._initialize_root(project_root, StorageKind.PROJECT_LIBRARY)
        self._initialize_root(cache_root, StorageKind.CACHE)
        preference = self._preference_from_roots(project_root, cache_root)
        self._save_preference(preference)
        return preference

    def _load_or_initialize(self) -> StorageLayoutPreference:
        settings, invalid = read_user_settings(self.preference_path)
        if invalid:
            raise ValueError("user settings are invalid")
        raw = settings.get("storage_layout")
        if raw is None:
            return self._adopt_default_layout()
        try:
            preference = StorageLayoutPreference.model_validate(raw)
            project_root = self._validate_root(
                Path(preference.project_library_root), create=False
            )
            cache_root = self._validate_root(Path(preference.cache_root), create=False)
            self._validate_pair(project_root, cache_root)
            current = self._preference_from_roots(project_root, cache_root)
        except (OSError, ValueError, ValidationError) as exc:
            raise ValueError("stored storage layout is invalid") from exc
        if (
            current.project_library_id != preference.project_library_id
            or current.cache_id != preference.cache_id
        ):
            raise ValueError("stored storage identity changed")
        return preference

    @property
    def project_library_root(self) -> Path:
        return Path(self._preference.project_library_root)

    @property
    def cache_root(self) -> Path:
        return Path(self._preference.cache_root)

    def snapshot(
        self, *, editable: bool = True, blocked_reason: str | None = None
    ) -> StorageLayoutSnapshot:
        project_root = self.project_library_root
        cache_root = self.cache_root
        return StorageLayoutSnapshot(
            **self.status(
                editable=editable,
                blocked_reason=blocked_reason,
            ).model_dump(),
            project_library_bytes=_tree_size(project_root),
            project_library_free_bytes=shutil.disk_usage(project_root).free,
            cache_bytes=_tree_size(cache_root),
            cache_free_bytes=shutil.disk_usage(cache_root).free,
        )

    def status(
        self, *, editable: bool = True, blocked_reason: str | None = None
    ) -> StorageLayoutStatus:
        return StorageLayoutStatus(
            project_library_root=self._preference.project_library_root,
            project_library_id=self._preference.project_library_id,
            cache_root=self._preference.cache_root,
            cache_id=self._preference.cache_id,
            restart_required=self._restart_required,
            editable=editable and not self._restart_required,
            blocked_reason=(
                "restart_required"
                if self._restart_required
                else None if editable else blocked_reason
            ),
        )

    @staticmethod
    def _require_empty(root: Path) -> None:
        if any(root.iterdir()):
            raise ValueError("target storage root must be empty")

    def _require_fresh_cache(self, root: Path) -> None:
        entries = tuple(root.iterdir())
        unexpected = tuple(entry for entry in entries if entry.name != _MARKER)
        if unexpected:
            raise ValueError("target storage root must be empty")
        if self._marker_path(root).exists():
            self._read_marker(root, StorageKind.CACHE)

    def _copy_verified(
        self, source: Path, destination: Path, kind: StorageKind
    ) -> None:
        source_identity = _path_identity(source)
        destination_identity = _path_identity(destination)
        ignored_source = frozenset({_MARKER})
        ignored_destination = frozenset(
            {_MARKER, _MIGRATION_OPERATION, _MIGRATION_STAGING}
        )
        operation = destination / _MIGRATION_OPERATION
        staging = destination / _MIGRATION_STAGING
        source_files = _tree_inventory(source, ignored=ignored_source)
        destination_files = _tree_inventory(
            destination, ignored=ignored_destination
        )
        if self._marker_path(destination).exists():
            if source_files != destination_files:
                raise ValueError("completed migration target differs from source")
            if staging.exists():
                _remove_owned_path(staging)
            operation.unlink(missing_ok=True)
            return

        if not operation.exists():
            if any(destination.iterdir()):
                raise ValueError("target storage root must be empty")
            required = sum(size for size, _digest in source_files.values())
            if shutil.disk_usage(destination).free < required:
                raise OSError("target storage root has insufficient free space")
            temporary = operation.with_suffix(".tmp")
            payload = {
                "schema_version": 1,
                "source": str(source),
                "source_identity": list(source_identity),
            }
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, operation)
        else:
            try:
                payload = json.loads(operation.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise OSError("storage migration operation is invalid") from exc
            if payload != {
                "schema_version": 1,
                "source": str(source),
                "source_identity": list(source_identity),
            }:
                raise OSError("storage migration source identity changed")

        staging.mkdir(exist_ok=True)
        if has_reparse_component(staging):
            raise OSError("storage migration staging is unsafe")
        for child in source.iterdir():
            if child.name == _MARKER:
                continue
            if has_reparse_component(child):
                raise ValueError("storage source contains a reparse point")
            activated = destination / child.name
            target = staging / child.name
            if activated.exists():
                source_entry = _tree_inventory(child) if child.is_dir() else {
                    Path(child.name): (child.stat().st_size, _sha256(child))
                }
                activated_entry = (
                    _tree_inventory(activated)
                    if activated.is_dir()
                    else {
                        Path(child.name): (
                            activated.stat().st_size,
                            _sha256(activated),
                        )
                    }
                )
                if source_entry != activated_entry:
                    raise OSError("activated migration entry differs from source")
                continue
            if target.exists():
                _remove_owned_path(target)
            if child.is_dir():
                shutil.copytree(child, target)
            elif child.is_file() and child.stat().st_nlink == 1:
                shutil.copy2(child, target)
            else:
                raise ValueError("storage source contains an unsafe entry")
        if _path_identity(source) != source_identity:
            raise OSError("storage source identity changed during migration")
        if _path_identity(destination) != destination_identity:
            raise OSError("storage target identity changed during migration")
        for child in tuple(staging.iterdir()):
            os.replace(child, destination / child.name)
        staging.rmdir()
        destination_files = _tree_inventory(destination, ignored=ignored_destination)
        if source_files != destination_files or source_files != _tree_inventory(
            source, ignored=ignored_source
        ):
            raise OSError("storage copy verification failed")
        self._initialize_root(destination, kind)
        operation.unlink()

    def switch(
        self,
        project_library_root: Path,
        cache_root: Path,
        *,
        project_action: ProjectLibraryAction,
        cache_action: CacheAction,
    ) -> StorageLayoutSnapshot:
        if self._restart_required:
            raise RuntimeError("storage layout is waiting for restart")
        new_project = self._validate_root(project_library_root, create=True)
        new_cache = self._validate_root(cache_root, create=True)
        self._validate_pair(new_project, new_cache)

        if new_project != self.project_library_root:
            if project_action is ProjectLibraryAction.MIGRATE:
                self._copy_verified(
                    self.project_library_root,
                    new_project,
                    StorageKind.PROJECT_LIBRARY,
                )
            else:
                self._validate_existing_project_library(new_project)
            self._initialize_root(new_project, StorageKind.PROJECT_LIBRARY)

        if new_cache != self.cache_root:
            if cache_action is CacheAction.MIGRATE:
                self._copy_verified(self.cache_root, new_cache, StorageKind.CACHE)
            else:
                self._require_fresh_cache(new_cache)
            self._initialize_root(new_cache, StorageKind.CACHE)

        preference = self._preference_from_roots(new_project, new_cache)
        self._save_preference(preference)
        self._preference = preference
        self._restart_required = True
        return self.snapshot()

    def _referenced_cache_entries(self) -> set[tuple[str, str, str]]:
        from gs_video.project.repository import ProjectRepository

        referenced: set[tuple[str, str, str]] = set()
        projects_root = self.project_library_root / "projects"
        if not projects_root.exists():
            return referenced
        for project_root in projects_root.iterdir():
            if not (project_root / "project.json").is_file():
                continue
            project = ProjectRepository(project_root).load()
            for stage in project.stages.values():
                for reference in (*stage.output_paths, *stage.artifacts.values()):
                    referenced.add(
                        (
                            reference.project_id,
                            reference.category.value,
                            reference.cache_key,
                        )
                    )
        return referenced

    def _invalidate_all_cache_stages(self) -> None:
        from gs_video.domain.models import Project, StageStatus
        from gs_video.project.repository import ProjectRepository

        projects_root = self.project_library_root / "projects"
        if not projects_root.exists():
            return
        for project_root in projects_root.iterdir():
            if not (project_root / "project.json").is_file():
                continue
            repository = ProjectRepository(project_root)

            def invalidate(project: Project) -> None:
                for stage in project.stages.values():
                    if stage.output_paths or stage.artifacts:
                        stage.status = StageStatus.STALE
                        stage.cache_key = None
                        stage.output_paths.clear()
                        stage.artifacts.clear()
                        stage.error_code = None
                project.workflow.export_result = None
                project.workflow.preview = None
                project.workflow.foot_point = None
                project.workflow.confirmed_camera_revision = None
                project.workflow.confirmed_preview_artifact_id = None

            repository.update(invalidate)

    def _cleanup_fingerprint(self, mode: CacheCleanupMode) -> str:
        referenced = sorted(self._referenced_cache_entries())
        inventory = _metadata_inventory(self.cache_root)
        payload = json.dumps(
            {
                "mode": mode.value,
                "cache_identity": list(_path_identity(self.cache_root)),
                "project_identity": list(
                    _path_identity(self.project_library_root)
                ),
                "referenced": referenced,
                "inventory": inventory,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cleanup_candidate_totals(
        self, mode: CacheCleanupMode
    ) -> tuple[int, int]:
        referenced = (
            set() if mode is CacheCleanupMode.DEEP else self._referenced_cache_entries()
        )
        entries = 0
        size = 0
        projects_root = self.cache_root / "projects"
        if not projects_root.exists():
            return entries, size
        from gs_video.domain.models import ArtifactCategory
        from gs_video.pipeline.artifacts import validate_cache_key

        for project_root in projects_root.iterdir():
            _canonical_uuid(project_root.name)
            for category_root in project_root.iterdir():
                category = ArtifactCategory(category_root.name)
                for entry in category_root.iterdir():
                    if _is_publisher_staging(entry.name):
                        entries += 1
                        size += _tree_size(entry)
                        continue
                    if (
                        category is ArtifactCategory.PREVIEWS
                        and entry.is_file()
                        and entry.suffix == ".png"
                    ):
                        if mode is CacheCleanupMode.DEEP:
                            metadata = entry.lstat()
                            if metadata.st_nlink != 1 or has_reparse_component(entry):
                                raise OSError("cache cleanup encountered unsafe preview")
                            entries += 1
                            size += metadata.st_size
                        continue
                    validate_cache_key(entry.name)
                    identity = (project_root.name, category.value, entry.name)
                    if identity not in referenced:
                        entries += 1
                        size += _tree_size(entry)
        return entries, size

    def plan_cache_cleanup(self, mode: CacheCleanupMode) -> CacheCleanupPlan:
        if self._restart_required:
            raise RuntimeError("storage layout is waiting for restart")
        if not isinstance(mode, CacheCleanupMode):
            raise ValueError("invalid cache cleanup mode")
        with self._cleanup_lock:
            removable, reclaimable = self._cleanup_candidate_totals(mode)
            token = uuid4().hex
            self._cleanup_plans[token] = (
                time.monotonic(),
                mode,
                self._cleanup_fingerprint(mode),
            )
            return CacheCleanupPlan(
                plan_token=token,
                mode=mode,
                removable_entries=removable,
                reclaimable_bytes=reclaimable,
                expires_in_seconds=300,
            )

    def cleanup_cache(
        self, mode: CacheCleanupMode, plan_token: str
    ) -> CacheCleanupResult:
        if self._restart_required:
            raise RuntimeError("storage layout is waiting for restart")
        if not isinstance(mode, CacheCleanupMode):
            raise ValueError("invalid cache cleanup mode")
        with self._cleanup_lock:
            planned = self._cleanup_plans.pop(plan_token, None)
            if (
                planned is None
                or planned[1] is not mode
                or time.monotonic() - planned[0] > 300
                or planned[2] != self._cleanup_fingerprint(mode)
            ):
                raise ValueError("cache cleanup plan is stale or invalid")
        referenced = set() if mode is CacheCleanupMode.DEEP else self._referenced_cache_entries()
        if mode is CacheCleanupMode.DEEP:
            self._invalidate_all_cache_stages()

        removed_entries = 0
        freed_bytes = 0
        projects_root = self.cache_root / "projects"
        if projects_root.exists():
            from gs_video.domain.models import ArtifactCategory
            from gs_video.pipeline.artifacts import validate_cache_key

            for project_root in tuple(projects_root.iterdir()):
                _canonical_uuid(project_root.name)
                for category_root in tuple(project_root.iterdir()):
                    ArtifactCategory(category_root.name)
                    for entry in tuple(category_root.iterdir()):
                        if _is_publisher_staging(entry.name):
                            metadata = entry.lstat()
                            if (
                                has_reparse_component(entry)
                                or not stat.S_ISDIR(metadata.st_mode)
                            ):
                                raise OSError(
                                    "cache cleanup encountered unsafe staging"
                                )
                            freed_bytes += _tree_size(entry)
                            shutil.rmtree(entry)
                            removed_entries += 1
                            continue
                        if (
                            category_root.name == ArtifactCategory.PREVIEWS.value
                            and entry.is_file()
                            and entry.suffix == ".png"
                        ):
                            if mode is CacheCleanupMode.SAFE:
                                continue
                            metadata = entry.lstat()
                            if (
                                has_reparse_component(entry)
                                or not stat.S_ISREG(metadata.st_mode)
                                or metadata.st_nlink != 1
                            ):
                                raise OSError(
                                    "cache cleanup encountered an unsafe preview"
                                )
                            freed_bytes += metadata.st_size
                            entry.unlink()
                            removed_entries += 1
                            continue
                        validate_cache_key(entry.name)
                        identity = (project_root.name, category_root.name, entry.name)
                        if identity in referenced:
                            continue
                        metadata = entry.lstat()
                        if (
                            has_reparse_component(entry)
                            or not stat.S_ISDIR(metadata.st_mode)
                        ):
                            raise OSError("cache cleanup encountered an unsafe entry")
                        freed_bytes += _tree_size(entry)
                        shutil.rmtree(entry)
                        removed_entries += 1
                    try:
                        category_root.rmdir()
                    except OSError:
                        pass
                try:
                    project_root.rmdir()
                except OSError:
                    pass
        return CacheCleanupResult(
            mode=mode,
            removed_entries=removed_entries,
            freed_bytes=freed_bytes,
            storage=self.snapshot(),
        )


__all__ = [
    "CacheAction",
    "CacheCleanupMode",
    "CacheCleanupPlan",
    "CacheCleanupResult",
    "ProjectLibraryAction",
    "StorageKind",
    "StorageLayoutManager",
    "StorageLayoutSnapshot",
    "StorageLayoutStatus",
]
