# src/comet/cli.py
import argparse
from comet import demo, runners, summarize, context, data

def main():
    p = argparse.ArgumentParser("comet")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo").set_defaults(fn=demo.run)
    r = sub.add_parser("run"); r.add_argument("--ordering", choices=["AtoB","BtoA","both"], default="both"); r.set_defaults(fn=runners.run)
    s = sub.add_parser("summarize"); s.set_defaults(fn=summarize.run)
    c = sub.add_parser("context"); c.set_defaults(fn=context.build)
    f = sub.add_parser("data"); f.add_argument("action", choices=["fetch"]); f.set_defaults(fn=data.fetch)

    args = p.parse_args(); args.fn(args)

