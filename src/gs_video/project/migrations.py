from copy import deepcopy
from typing import cast


CURRENT_SCHEMA_VERSION = 4


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
        else:
            raise ValueError(f"不支持的项目版本: {version}")

    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"项目版本 {version} 高于应用支持版本")

    return data
