from __future__ import annotations

import argparse
from ipaddress import ip_address

from gs_video.environment.doctor import EnvironmentDoctor


def _loopback_host(value: str) -> str:
    try:
        address = ip_address(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("host must be an IP loopback address") from error
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("host must be an IP loopback address")
    return value


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doctor", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", type=_loopback_host)
    parser.add_argument("--port", default=0, type=_port)
    return parser


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
        from gs_video.app import run_api

        return run_api(args.host, args.port)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
