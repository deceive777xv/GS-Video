import hashlib
import json
from typing import Any


def cache_key(
    stage: str,
    inputs: dict[str, Any],
    params: dict[str, Any],
    implementation_version: str,
) -> str:
    payload = {
        "stage": stage,
        "inputs": inputs,
        "params": params,
        "version": implementation_version,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
