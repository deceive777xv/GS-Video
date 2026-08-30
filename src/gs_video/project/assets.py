from __future__ import annotations

import hashlib
import os
import shutil
import stat
import time
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from threading import RLock
from typing import BinaryIO
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gs_video.domain.models import LutSummary, SceneSummary, VideoSummary
from gs_video.segmentation.paths import has_reparse_component
from gs_video.resource_admission import bytes_with_disk_headroom


class AssetKind(StrEnum):
    VIDEO = "video"
    PLY = "ply"
    LUT = "lut"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AssetRecord(_StrictModel):
    asset_id: str
    kind: AssetKind
    original_filename: str
    stored_relative_path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    imported_at: datetime
    video_summary: VideoSummary | None = None
    scene_summary: SceneSummary | None = None
    lut_summary: LutSummary | None = None

    @model_validator(mode="after")
    def validate_summary_kind(self) -> AssetRecord:
        if self.kind is AssetKind.VIDEO:
            if (
                self.video_summary is None
                or self.scene_summary is not None
                or self.lut_summary is not None
            ):
                raise ValueError("video assets require only a video summary")
        elif self.kind is AssetKind.PLY:
            if (
                self.scene_summary is None
                or self.video_summary is not None
                or self.lut_summary is not None
            ):
                raise ValueError("PLY assets require only a scene summary")
        elif (
            self.lut_summary is None
            or self.video_summary is not None
            or self.scene_summary is not None
        ):
            raise ValueError("LUT assets require only a LUT summary")
        return self


class AssetIndex(_StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    assets: dict[str, AssetRecord] = Field(default_factory=dict)


AssetSummary = VideoSummary | SceneSummary | LutSummary
AssetInspector = Callable[[Path, int, str], AssetSummary]


def _ordinary_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if (
        has_reparse_component(path)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise OSError("asset must be a single-link ordinary file")
    return metadata


class AssetLibrary:
    def __init__(self, root: Path, *, max_size: int = 4 * 1024 * 1024 * 1024) -> None:
        self.root = Path(root).absolute()
        self.path = self.root / "index.json"
        self.staging_root = self.root / ".staging"
        self.max_size = max_size
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(exist_ok=True)
        for kind in AssetKind:
            (self.root / kind.value).mkdir(exist_ok=True)
        if not self.path.exists():
            self._save(AssetIndex())
        else:
            self._retry_pending_cleanup(self._load())

    def _retry_pending_cleanup(self, index: AssetIndex) -> None:
        expected = {
            (self.root / record.kind.value / record.stored_relative_path).absolute()
            for record in index.assets.values()
        }
        for entry in self.staging_root.iterdir():
            if len(entry.name) != 32 or any(
                character not in "0123456789abcdef" for character in entry.name
            ):
                continue
            try:
                _ordinary_file(entry)
                entry.unlink()
            except OSError:
                continue
        for kind in AssetKind:
            kind_root = self.root / kind.value
            for prefix in kind_root.iterdir():
                if (
                    len(prefix.name) != 2
                    or any(
                        character not in "0123456789abcdef"
                        for character in prefix.name
                    )
                ):
                    continue
                try:
                    metadata = prefix.lstat()
                    if has_reparse_component(prefix) or not stat.S_ISDIR(
                        metadata.st_mode
                    ):
                        continue
                    for entry in prefix.iterdir():
                        if entry.absolute() in expected:
                            continue
                        stem = entry.stem
                        if len(stem) != 64 or any(
                            character not in "0123456789abcdef"
                            for character in stem
                        ):
                            continue
                        try:
                            _ordinary_file(entry)
                            entry.unlink()
                        except OSError:
                            continue
                    prefix.rmdir()
                except OSError:
                    continue

    def _load(self) -> AssetIndex:
        index = AssetIndex.model_validate_json(self.path.read_text(encoding="utf-8"))
        if set(index.assets) != {record.asset_id for record in index.assets.values()}:
            raise ValueError("asset index keys do not match asset IDs")
        for record in index.assets.values():
            self._resolve_record(record)
        return index

    def _save(self, index: AssetIndex) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temporary.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        try:
            for attempt in range(4):
                try:
                    os.replace(temporary, self.path)
                    return
                except PermissionError:
                    if attempt == 3:
                        raise
                    time.sleep(0.01 * (attempt + 1))
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_record(self, record: AssetRecord) -> Path:
        relative = Path(record.stored_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("asset path is not a safe relative path")
        kind_root = (self.root / record.kind.value).absolute()
        path = (kind_root / relative).absolute()
        if not path.is_relative_to(kind_root):
            raise ValueError("asset path escaped its kind root")
        metadata = _ordinary_file(path)
        if metadata.st_size != record.size:
            raise ValueError("asset size does not match the index")
        return path

    def list(self, kind: AssetKind | None = None) -> tuple[AssetRecord, ...]:
        with self._lock:
            records = self._load().assets.values()
            selected = [record for record in records if kind is None or record.kind is kind]
            selected.sort(key=lambda item: item.imported_at, reverse=True)
            return tuple(record.model_copy(deep=True) for record in selected)

    def get(self, asset_id: str) -> AssetRecord:
        with self._lock:
            try:
                return self._load().assets[asset_id].model_copy(deep=True)
            except KeyError as error:
                raise KeyError(asset_id) from error

    def resolve(self, asset_id: str, expected_kind: AssetKind) -> Path:
        with self._lock:
            try:
                record = self._load().assets[asset_id]
            except KeyError as error:
                raise KeyError(asset_id) from error
            if record.kind is not expected_kind:
                raise ValueError("asset kind does not match the requested input")
            return self._resolve_record(record)

    def import_file(
        self,
        kind: AssetKind,
        source: Path,
        inspect: AssetInspector,
        *,
        original_filename: str | None = None,
    ) -> tuple[AssetRecord, bool]:
        source = Path(source).absolute()
        display_name = source.name if original_filename is None else original_filename
        self._validate_filename(display_name)
        source_metadata = _ordinary_file(source)
        with source.open("rb") as reader:
            opened = os.fstat(reader.fileno())
            if (
                opened.st_dev != source_metadata.st_dev
                or opened.st_ino != source_metadata.st_ino
                or opened.st_size != source_metadata.st_size
            ):
                raise OSError("asset identity changed before import")
            result = self.import_stream(kind, display_name, reader, inspect)
            final = _ordinary_file(source)
            if (
                final.st_dev != source_metadata.st_dev
                or final.st_ino != source_metadata.st_ino
                or final.st_size != source_metadata.st_size
            ):
                raise OSError("asset identity changed during import")
            return result

    @staticmethod
    def _validate_filename(display_name: str) -> None:
        if (
            not display_name
            or Path(display_name).name != display_name
            or PureWindowsPath(display_name).name != display_name
        ):
            raise ValueError("original filename must be a basename")

    def import_stream(
        self,
        kind: AssetKind,
        original_filename: str,
        reader: BinaryIO,
        inspect: AssetInspector,
    ) -> tuple[AssetRecord, bool]:
        self._validate_filename(original_filename)
        source_metadata = os.fstat(reader.fileno())
        if not stat.S_ISREG(source_metadata.st_mode) or source_metadata.st_nlink != 1:
            raise OSError("asset stream must be a single-link ordinary file")
        if source_metadata.st_size > self.max_size:
            raise ValueError("asset exceeds the configured size limit")
        required_free = bytes_with_disk_headroom(int(source_metadata.st_size))
        if shutil.disk_usage(self.root).free < required_free:
            raise OSError("asset library has insufficient space with 20% headroom")
        staging = self.staging_root / uuid4().hex
        digest = hashlib.sha256()
        size = 0
        try:
            reader.seek(0)
            with staging.open("xb") as writer:
                for block in iter(lambda: reader.read(1024 * 1024), b""):
                    size += len(block)
                    if size > self.max_size:
                        raise ValueError("asset exceeds the configured size limit")
                    digest.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            final = os.fstat(reader.fileno())
            if (
                final.st_dev != source_metadata.st_dev
                or final.st_ino != source_metadata.st_ino
                or final.st_size != source_metadata.st_size
                or final.st_nlink != source_metadata.st_nlink
            ):
                raise OSError("asset stream identity changed during import")
            sha256 = digest.hexdigest()
            summary = inspect(staging, size, sha256)
            summary = summary.model_copy(update={"filename": original_filename})
            if kind is AssetKind.VIDEO and not isinstance(summary, VideoSummary):
                raise ValueError("video inspector returned the wrong summary type")
            if kind is AssetKind.PLY and not isinstance(summary, SceneSummary):
                raise ValueError("PLY inspector returned the wrong summary type")
            if kind is AssetKind.LUT and not isinstance(summary, LutSummary):
                raise ValueError("LUT inspector returned the wrong summary type")
            with self._lock:
                index = self._load()
                existing = next(
                    (
                        record
                        for record in index.assets.values()
                        if record.kind is kind
                        and record.size == size
                        and record.sha256 == sha256
                    ),
                    None,
                )
                if existing is not None:
                    return existing.model_copy(deep=True), False
                suffix = (
                    Path(original_filename).suffix.lower()
                    if kind is AssetKind.VIDEO
                    else ".ply"
                    if kind is AssetKind.PLY
                    else ".cube"
                )
                relative = Path(sha256[:2]) / f"{sha256}{suffix}"
                destination = self.root / kind.value / relative
                destination.parent.mkdir(exist_ok=True)
                os.replace(staging, destination)
                record = AssetRecord(
                    asset_id=str(uuid4()),
                    kind=kind,
                    original_filename=original_filename,
                    stored_relative_path=relative.as_posix(),
                    size=size,
                    sha256=sha256,
                    imported_at=datetime.now(timezone.utc),
                    video_summary=summary if isinstance(summary, VideoSummary) else None,
                    scene_summary=summary if isinstance(summary, SceneSummary) else None,
                    lut_summary=summary if isinstance(summary, LutSummary) else None,
                )
                index.assets[record.asset_id] = record
                try:
                    self._save(index)
                except BaseException:
                    destination.unlink(missing_ok=True)
                    raise
                return record.model_copy(deep=True), True
        finally:
            staging.unlink(missing_ok=True)

    def delete(self, asset_id: str) -> None:
        with self._lock:
            index = self._load()
            try:
                record = index.assets.pop(asset_id)
            except KeyError as error:
                raise KeyError(asset_id) from error
            path = self._resolve_record(record)
            self._save(index)
            path.unlink()
