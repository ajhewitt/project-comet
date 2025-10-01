from __future__ import annotations

import argparse

from . import context, data, demo, runners, summarize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comet", description="Project Comet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # demo
    demo_p = sub.add_parser("demo", help="Run a quick demo")
    demo_p.set_defaults(fn=demo.run)

    # run
    run_p = sub.add_parser("run", help="Run the pipeline")
    run_p.add_argument(
        "--ordering",
        choices=["AtoB", "BtoA", "both"],
        default="both",
        help="Select pipeline ordering",
    )
    run_p.set_defaults(fn=runners.run)

    # summarize
    sum_p = sub.add_parser("summarize", help="Summarize outputs")
    sum_p.set_defaults(fn=summarize.run)

    # context
    ctx_p = sub.add_parser("context", help="Build analysis context")
    ctx_p.set_defaults(fn=context.build)

    # data
    data_p = sub.add_parser("data", help="Data utilities")
    data_p.add_argument("action", choices=["fetch"], help="What to do with data")
    data_p.set_defaults(fn=data.fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(bool(args.fn(args)))


if __name__ == "__main__":
    raise SystemExit(main())
