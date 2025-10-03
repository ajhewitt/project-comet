import argparse
import json
import sys
from pathlib import Path
import numpy as np

from .commutator import commutator, z_score
from .config import get_data_dir

DEFAULT_PREREG = "config/prereg.yaml"
DEFAULT_PATHS = "config/paths.example.yaml"
DEFAULT_OUT = "artifacts/summary.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comet", description="Project Comet CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run a tiny synthetic demo")
    p_demo.add_argument("--ell-min", type=int, default=30)
    p_demo.add_argument("--ell-max", type=int, default=300)
    p_demo.set_defaults(fn=_cmd_demo)

    # run
    p_run = subparsers.add_parser("run", help="Run the Comet pipeline")
    p_run.add_argument("--prereg", default=DEFAULT_PREREG, help="Path to preregistration YAML")
    p_run.add_argument("--paths", default=DEFAULT_PATHS, help="Path to paths/config YAML")
    p_run.add_argument("--out", default=DEFAULT_OUT, help="Output summary JSON")
    p_run.add_argument(
        "--ordering",
        choices=["a_then_b", "b_then_a", "both"],
        default="both",
        help="Pipeline ordering to execute",
    )
    p_run.set_defaults(fn=_cmd_run)

    # summarize
    p_sum = subparsers.add_parser("summarize", help="Summarize latest artifacts")
    p_sum.add_argument("--in", dest="inp", default=DEFAULT_OUT, help="Summary JSON to read")
    p_sum.set_defaults(fn=_cmd_summarize)

    # context
    p_ctx = subparsers.add_parser("context", help="Show configured context templates")
    p_ctx.add_argument("--paths", default=DEFAULT_PATHS, help="Path to paths/config YAML")
    p_ctx.set_defaults(fn=_cmd_context)

    # data
    p_data = subparsers.add_parser("data", help="Show resolved data directory and files")
    p_data.add_argument("--list", action="store_true", help="List files in data dir")
    p_data.set_defaults(fn=_cmd_data)

    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    ell = np.arange(args.ell_min, args.ell_max)
    x = np.ones_like(ell, dtype=float)
    y = np.ones_like(ell, dtype=float) * 2
    delta = commutator(x, y)
    cov = np.eye(delta.size)
    payload = {"z": z_score(delta, cov), "size": int(delta.size)}
    print(json.dumps(payload))
    return 0


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    _ensure_parent(out)
    payload = {
        "prereg": args.prereg,
        "paths": args.paths,
        "ordering": args.ordering,
        "nbins": 10,
        "z": 0.0,
    }
    out.write_text(json.dumps(payload))
    print(json.dumps({"wrote": str(out), "ordering": args.ordering}))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    p = Path(args.inp)
    if not p.exists():
        print(json.dumps({"error": f"missing {p}"}))
        return 2
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 2
    print(json.dumps({"summary": data}))
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    payload = {
        "paths": args.paths,
        "templates": ["ecliptic_scan", "galactic_mask", "hitcount_gradient"],
    }
    print(json.dumps(payload))
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    d = get_data_dir()
    resp = {"data_dir": str(d)}
    if args.list and d.exists():
        try:
            resp["files"] = sorted([f.name for f in d.iterdir() if f.is_file()])
        except Exception:
            resp["files"] = []
    print(json.dumps(resp))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return ns.fn(ns)


if __name__ == "__main__":
    sys.exit(main())
