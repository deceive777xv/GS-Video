from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


class RuntimeManifestError(ValueError):
    """Raised when the checked-in runtime manifest is unsafe or incomplete."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
DEFAULT_ALLOWED_HOSTS = frozenset(
    {
        "codeload.github.com",
        "download.pytorch.org",
        "files.pythonhosted.org",
        "github.com",
        "huggingface.co",
        "pypi.org",
        "raw.githubusercontent.com",
        "www.gyan.dev",
    }
)


def _relative_path(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RuntimeManifestError(f"{label} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeManifestError(f"{label} must remain inside the runtime directory")
    if re.match(r"^[A-Za-z]:", normalized):
        raise RuntimeManifestError(f"{label} must not contain a Windows drive")
    return path


def _safe_url(value: Any, label: str, allowed_hosts: frozenset[str]) -> str:
    if not isinstance(value, str) or len(value) > 4096:
        raise RuntimeManifestError(f"{label} must be a bounded HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.hostname is None
        or parsed.query
        or parsed.fragment
        or parsed.hostname.lower() not in allowed_hosts
    ):
        raise RuntimeManifestError(f"{label} must use an approved HTTPS host")
    return value


@dataclass(frozen=True)
class RuntimeResource:
    id: str
    kind: str
    version: str
    url: str
    sha256: str
    size: int | None
    target: PurePosixPath
    extract_to: PurePosixPath | None
    strip_prefix: str | None
    markers: tuple[PurePosixPath, ...]
    license: str
    license_url: str | None


@dataclass(frozen=True)
class RuntimeAction:
    type: str
    target: PurePosixPath | None
    environment: str | None
    packages: tuple[str, ...]
    resources: tuple[str, ...]
    source: PurePosixPath | None
    index_url: str | None
    editable: bool


@dataclass(frozen=True)
class RuntimeManifest:
    schema_version: int
    platform: str
    resources: tuple[RuntimeResource, ...]
    actions: tuple[RuntimeAction, ...]
    fingerprint: str

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS,
    ) -> RuntimeManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeManifestError(f"runtime manifest is unreadable: {path}") from error
        return cls.from_payload(payload, allowed_hosts=allowed_hosts)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        allowed_hosts: frozenset[str] = DEFAULT_ALLOWED_HOSTS,
    ) -> RuntimeManifest:
        if not isinstance(payload, Mapping):
            raise RuntimeManifestError("runtime manifest must be an object")
        if payload.get("schema_version") != 1:
            raise RuntimeManifestError("unsupported runtime manifest version")
        if payload.get("platform") != "windows-amd64":
            raise RuntimeManifestError("runtime manifest is not for Windows x64")
        raw_resources = payload.get("resources")
        if not isinstance(raw_resources, list):
            raise RuntimeManifestError("runtime manifest resources must be a list")
        resources: list[RuntimeResource] = []
        seen: set[str] = set()
        for raw in raw_resources:
            if not isinstance(raw, Mapping):
                raise RuntimeManifestError("runtime resource must be an object")
            resource_id = raw.get("id")
            if not isinstance(resource_id, str) or _ID.fullmatch(resource_id) is None:
                raise RuntimeManifestError("runtime resource id is invalid")
            if resource_id in seen:
                raise RuntimeManifestError(f"duplicate runtime resource: {resource_id}")
            seen.add(resource_id)
            kind = raw.get("kind")
            if kind not in {"file", "archive"}:
                raise RuntimeManifestError(f"unsupported resource kind: {resource_id}")
            version = raw.get("version")
            if not isinstance(version, str) or not version or len(version) > 128:
                raise RuntimeManifestError(f"resource version is invalid: {resource_id}")
            url = _safe_url(raw.get("url"), f"resource {resource_id} URL", allowed_hosts)
            digest = raw.get("sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise RuntimeManifestError(f"resource hash is invalid: {resource_id}")
            size = raw.get("size")
            if size is not None and (
                not isinstance(size, int) or isinstance(size, bool) or size <= 0
            ):
                raise RuntimeManifestError(f"resource size is invalid: {resource_id}")
            target = _relative_path(raw.get("target"), f"resource {resource_id} target")
            extract_to = (
                _relative_path(raw.get("extract_to"), f"resource {resource_id} extract_to")
                if raw.get("extract_to") is not None
                else None
            )
            if (kind == "archive") != (extract_to is not None):
                raise RuntimeManifestError(
                    f"archive resource {resource_id} must define extract_to and files must not"
                )
            strip_prefix = raw.get("strip_prefix")
            if strip_prefix is not None and (
                not isinstance(strip_prefix, str)
                or not strip_prefix
                or "\\" in strip_prefix
                or strip_prefix.startswith("/")
                or ".." in PurePosixPath(strip_prefix).parts
            ):
                raise RuntimeManifestError(f"resource strip_prefix is invalid: {resource_id}")
            raw_markers = raw.get("markers", [])
            if not isinstance(raw_markers, list):
                raise RuntimeManifestError(f"resource markers are invalid: {resource_id}")
            markers = tuple(
                _relative_path(marker, f"resource {resource_id} marker")
                for marker in raw_markers
            )
            if kind == "archive" and not markers:
                raise RuntimeManifestError(
                    f"archive resource {resource_id} must define markers"
                )
            license_name = raw.get("license")
            if not isinstance(license_name, str) or not license_name or len(license_name) > 256:
                raise RuntimeManifestError(f"resource license is invalid: {resource_id}")
            license_url = (
                _safe_url(raw.get("license_url"), f"resource {resource_id} license URL", allowed_hosts)
                if raw.get("license_url") is not None
                else None
            )
            resources.append(
                RuntimeResource(
                    id=resource_id,
                    kind=kind,
                    version=version,
                    url=url,
                    sha256=digest,
                    size=size,
                    target=target,
                    extract_to=extract_to,
                    strip_prefix=strip_prefix,
                    markers=markers,
                    license=license_name,
                    license_url=license_url,
                )
            )

        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            raise RuntimeManifestError("runtime manifest actions must be a list")
        actions: list[RuntimeAction] = []
        for raw in raw_actions:
            if not isinstance(raw, Mapping):
                raise RuntimeManifestError("runtime action must be an object")
            action_type = raw.get("type")
            if action_type not in {"create_venv", "install_packages", "install_editable"}:
                raise RuntimeManifestError(f"unsupported runtime action: {action_type}")
            target_value = raw.get("target")
            action_target = (
                _relative_path(target_value, "runtime action target")
                if target_value is not None
                else None
            )
            environment = raw.get("environment")
            if environment is not None and (
                not isinstance(environment, str) or _ID.fullmatch(environment) is None
            ):
                raise RuntimeManifestError("runtime action environment is invalid")
            source_value = raw.get("source")
            source = (
                _relative_path(source_value, "runtime action source")
                if source_value is not None
                else None
            )
            packages_value = raw.get("packages", [])
            resources_value = raw.get("resources", [])
            if not isinstance(packages_value, list) or not all(
                isinstance(item, str) and item and len(item) <= 256 for item in packages_value
            ):
                raise RuntimeManifestError("runtime action packages are invalid")
            if not isinstance(resources_value, list) or not all(
                isinstance(item, str) and item in seen for item in resources_value
            ):
                raise RuntimeManifestError("runtime action resources are invalid")
            index_url = (
                _safe_url(raw.get("index_url"), "runtime action index URL", allowed_hosts)
                if raw.get("index_url") is not None
                else None
            )
            if action_type == "create_venv" and action_target is None:
                raise RuntimeManifestError("create_venv requires target")
            if action_type == "install_packages" and (
                action_target is None or not packages_value
            ):
                raise RuntimeManifestError("install_packages requires target and packages")
            if action_type == "install_editable" and action_target is None:
                raise RuntimeManifestError("install_editable requires target")
            actions.append(
                RuntimeAction(
                    type=action_type,
                    target=action_target,
                    environment=environment,
                    packages=tuple(packages_value),
                    resources=tuple(resources_value),
                    source=source,
                    index_url=index_url,
                    editable=bool(raw.get("editable", True)),
                )
            )

        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return cls(
            schema_version=1,
            platform="windows-amd64",
            resources=tuple(resources),
            actions=tuple(actions),
            fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def resource(self, resource_id: str) -> RuntimeResource:
        for resource in self.resources:
            if resource.id == resource_id:
                return resource
        raise RuntimeManifestError(f"runtime manifest resource is missing: {resource_id}")

    def resolve(self, runtime_root: Path, relative: PurePosixPath) -> Path:
        root = runtime_root.resolve(strict=False)
        candidate = (root / Path(*relative.parts)).resolve(strict=False)
        if candidate != root and root not in candidate.parents:
            raise RuntimeManifestError("runtime manifest path escapes the runtime directory")
        return candidate


__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "RuntimeAction",
    "RuntimeManifest",
    "RuntimeManifestError",
    "RuntimeResource",
]
