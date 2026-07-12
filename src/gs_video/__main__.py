from __future__ import annotations

import argparse

from gs_video.environment.doctor import EnvironmentDoctor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=0, type=int)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.doctor:
        report = EnvironmentDoctor().check()
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(report)
        return 0 if report.ready else 2

    if args.serve:
        from gs_video.app import run_api

        return run_api(args.host, args.port)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
