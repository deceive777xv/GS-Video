from __future__ import annotations

import argparse
from pathlib import Path
import sys

from gs_video.environment.doctor import EnvironmentDoctor
from gs_video.runtime import load_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--session-token-stdin", action="store_true")
    return parser


def _read_private_token() -> str:
    line = sys.stdin.readline(4097)
    if len(line) > 4096 or not line:
        raise SystemExit("session token stdin line is missing or too long")
    token = line[:-1] if line.endswith("\n") else line
    if token.endswith("\r"):
        token = token[:-1]
    if not token or any(ord(character) < 32 or ord(character) == 127 for character in token):
        raise SystemExit("session token stdin line is invalid")
    return token


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        report = EnvironmentDoctor().check()
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(report)
        return 0 if report.ready else 2

    if args.serve:
        if args.runtime_config is None or not args.runtime_config.is_absolute():
            raise SystemExit("runtime configuration must be an absolute JSON path")
        if not args.session_token_stdin:
            raise SystemExit("--serve requires --session-token-stdin")
        try:
            config = load_runtime_config(args.runtime_config)
        except (OSError, ValueError) as error:
            raise SystemExit(f"runtime configuration is invalid: {error}") from error
        token = _read_private_token()
        from gs_video.app import run_api

        return run_api(config, token)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
