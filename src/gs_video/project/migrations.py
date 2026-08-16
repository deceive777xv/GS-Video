from copy import deepcopy
from pathlib import PurePosixPath
from typing import cast
from uuid import UUID, uuid4

from gs_video.domain.models import ArtifactCategory


CURRENT_SCHEMA_VERSION = 6


def _canonical_project_id(value: object) -> str:
    if isinstance(value, str):
        try:
            parsed = UUID(value)
        except ValueError:
            parsed = None
        if parsed is not None and str(parsed) == value:
            return value
    return str(uuid4())


def _migrate_artifact_reference(project_id: str, value: object) -> object:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ValueError("项目 artifact 引用必须是字符串或对象")
    candidate = PurePosixPath(value.replace("\\", "/"))
    if candidate.is_absolute() or len(candidate.parts) < 2:
        raise ValueError("项目 artifact 路径无效")
    category_value, cache_key, *member_parts = candidate.parts
    try:
        category = ArtifactCategory(category_value)
    except ValueError as exc:
        raise ValueError("项目 artifact 类别无效") from exc
    if len(cache_key) != 64 or any(
        character not in "0123456789abcdef" for character in cache_key
    ):
        raise ValueError("项目 artifact 缓存键无效")
    return {
        "project_id": project_id,
        "category": category.value,
        "cache_key": cache_key,
        "member": "/".join(member_parts) or None,
    }


def migrate_project_dict(raw: dict[str, object]) -> dict[str, object]:
    data = deepcopy(raw)
    raw_version = data.get("schema_version", 0)
    if not isinstance(raw_version, (int, str)):
        raise ValueError("项目版本必须是整数")
    version = int(raw_version)

    while version < CURRENT_SCHEMA_VERSION:
        if version == 0:
            data.setdefault("stages", {})
            data["schema_version"] = 1
            version = 1
        elif version == 1:
            data.setdefault("workflow", {})
            data["schema_version"] = 2
            version = 2
        elif version == 2:
            workflow_value = data.setdefault("workflow", {})
            if not isinstance(workflow_value, dict):
                raise ValueError("项目工作流状态必须是对象")
            preview = workflow_value.get("preview")
            artifact_id = (
                preview.get("artifact_id")
                if isinstance(preview, dict)
                else None
            )
            preview_camera_revision = (
                preview.get("camera_revision")
                if isinstance(preview, dict)
                else None
            )
            preview_pick_revision = (
                preview.get("pick_buffer_revision")
                if isinstance(preview, dict)
                else None
            )
            confirmation_matches = (
                isinstance(artifact_id, str)
                and bool(artifact_id)
                and workflow_value.get("confirmed_camera_revision")
                == preview_camera_revision
            )
            if confirmation_matches:
                workflow_value["confirmed_preview_artifact_id"] = artifact_id
            else:
                if "confirmed_camera_revision" in workflow_value:
                    workflow_value["confirmed_camera_revision"] = None
                    workflow_value["confirmed_preview_artifact_id"] = None

            foot_point = workflow_value.get("foot_point")
            foot_matches = (
                confirmation_matches
                and isinstance(foot_point, dict)
                and foot_point.get("camera_revision") == preview_camera_revision
                and foot_point.get("pick_buffer_revision")
                == preview_pick_revision
            )
            if foot_matches:
                assert isinstance(foot_point, dict)
                foot_point["preview_artifact_id"] = cast(str, artifact_id)
            elif foot_point is not None:
                workflow_value["foot_point"] = None
            data["schema_version"] = 3
            version = 3
        elif version == 3:
            data.setdefault("updated_at", data.get("created_at"))
            data.setdefault("source_video_asset_id", None)
            data.setdefault("scene_ply_asset_id", None)
            data["schema_version"] = 4
            version = 4
        elif version == 4:
            project_id = _canonical_project_id(data.get("project_id"))
            data["project_id"] = project_id
            stages = data.setdefault("stages", {})
            if not isinstance(stages, dict):
                raise ValueError("项目阶段状态必须是对象")
            for stage in stages.values():
                if not isinstance(stage, dict):
                    raise ValueError("项目阶段状态必须是对象")
                outputs = stage.setdefault("output_paths", [])
                artifacts = stage.setdefault("artifacts", {})
                if not isinstance(outputs, list) or not isinstance(artifacts, dict):
                    raise ValueError("项目 artifact 状态无效")
                stage["output_paths"] = [
                    _migrate_artifact_reference(project_id, output)
                    for output in outputs
                ]
                stage["artifacts"] = {
                    role: _migrate_artifact_reference(project_id, artifact)
                    for role, artifact in artifacts.items()
                }
            data["schema_version"] = 5
            version = 5
        elif version == 5:
            workflow_value = data.setdefault("workflow", {})
            if not isinstance(workflow_value, dict):
                raise ValueError("项目工作流状态必须是对象")
            # A legacy orbit camera plus one picked point cannot safely imply a
            # source calibration or a three-point local plane. Preserve all
            # upstream media/segmentation/camera-solve artifacts and fail closed
            # at target placement.
            for legacy_key in (
                "target_camera",
                "confirmed_camera_revision",
                "confirmed_preview_artifact_id",
                "foot_point",
                "preview",
                "export_result",
            ):
                if legacy_key in workflow_value:
                    workflow_value[legacy_key] = None
            stages = data.setdefault("stages", {})
            if not isinstance(stages, dict):
                raise ValueError("项目阶段状态必须是对象")
            for stage_name in ("map_trajectory", "render", "composite", "export"):
                stage = stages.get(stage_name)
                if isinstance(stage, dict):
                    stage["status"] = "stale"
                    stage["cache_key"] = None
                    stage["error_code"] = None
                    stage["run_id"] = None
                    stage["input_generation"] = int(stage.get("input_generation", 0)) + 1
            data["schema_version"] = 6
            version = 6
        else:
            raise ValueError(f"不支持的项目版本: {version}")

    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"项目版本 {version} 高于应用支持版本")

    return data
