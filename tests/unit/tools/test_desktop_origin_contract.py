from __future__ import annotations

import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_desktop_development_origin_is_consistent_across_hosts() -> None:
    tauri_config = json.loads(
        (REPO_ROOT / "apps" / "desktop" / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    origin = tauri_config["build"]["devUrl"]
    parsed = re.fullmatch(r"http://([^:]+):(\d+)", origin)
    assert parsed is not None
    host, port = parsed.groups()

    rust_backend = (REPO_ROOT / "apps" / "desktop" / "src-tauri" / "src" / "backend.rs").read_text(
        encoding="utf-8"
    )
    vite_config = (REPO_ROOT / "apps" / "web" / "vite.config.ts").read_text(encoding="utf-8")
    browser_launcher = (REPO_ROOT / "tools" / "run_browser_api.py").read_text(encoding="utf-8")

    assert f'pub const DEV_BROWSER_ORIGIN: &str = "{origin}";' in rust_backend
    assert f"host: '{host}'" in vite_config
    assert f"port: {port}" in vite_config
    assert f'("{origin}",)' in browser_launcher


def test_random_loopback_ports_are_development_only_csp() -> None:
    tauri_config = json.loads(
        (REPO_ROOT / "apps" / "desktop" / "src-tauri" / "tauri.conf.json").read_text(
            encoding="utf-8"
        )
    )
    security = tauri_config["app"]["security"]

    assert "http://127.0.0.1:*" not in security["csp"]
    assert "ws://127.0.0.1:*" not in security["csp"]
    assert "http://127.0.0.1:*" in security["devCsp"]
    assert "ws://127.0.0.1:*" in security["devCsp"]
