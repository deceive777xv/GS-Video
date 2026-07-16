CURRENT_SCHEMA_VERSION = 2


def migrate_project_dict(raw: dict[str, object]) -> dict[str, object]:
    data = dict(raw)
    version = int(data.get("schema_version", 0))

    while version < CURRENT_SCHEMA_VERSION:
        if version == 0:
            data.setdefault("stages", {})
            data["schema_version"] = 1
            version = 1
        elif version == 1:
            data.setdefault("workflow", {})
            data["schema_version"] = 2
            version = 2
        else:
            raise ValueError(f"不支持的项目版本: {version}")

    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(f"项目版本 {version} 高于应用支持版本")

    return data
