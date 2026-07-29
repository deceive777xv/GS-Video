from __future__ import annotations

from pathlib import Path
import secrets
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    repo_root = REPO_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    source_root = repo_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from tools.prepare_desktop_runtime import prepare_desktop_runtime

    from gs_video.app import run_api
    from gs_video.runtime import load_runtime_config

    runtime_path = prepare_desktop_runtime(repo_root)
    token = secrets.token_urlsafe(32)
    print(
        "Browser session token (memory only): " + token,
        file=sys.stderr,
        flush=True,
    )
    print(
        "Use the port from the startup JSON below at http://127.0.0.1:1420.",
        file=sys.stderr,
        flush=True,
    )
    return run_api(
        load_runtime_config(runtime_path),
        token,
        ("http://127.0.0.1:1420",),
        startup_handshake=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
