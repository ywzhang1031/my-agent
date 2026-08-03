#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from my_agent.trajectory import make_trajectory, read_jsonl_trace, write_trajectory_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export raw trace JSONL to canonical trajectory JSON.")
    parser.add_argument("--trace", required=True, help="Input trace JSONL path.")
    parser.add_argument("--output", required=True, help="Output trajectory JSON path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        events = read_jsonl_trace(args.trace)
        trajectory = make_trajectory(events, source_path=args.trace)
        write_trajectory_json(trajectory, args.output)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
