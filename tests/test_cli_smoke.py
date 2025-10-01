import argparse
import types

from comet import cli


def _subparser_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    # find the SubParsersAction among parser._actions
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("No subparsers found on parser")


def test_cli_builds_and_has_subcommands():
    parser = cli.build_parser()
    spa = _subparser_action(parser)
    subcmds = set(spa.choices.keys())
    assert {"demo", "run", "summarize", "context", "data"} <= subcmds


def test_cli_parses_run_defaults():
    parser = cli.build_parser()
    ns = parser.parse_args(["run"])
    assert isinstance(ns, types.SimpleNamespace) or hasattr(ns, "__dict__")
    assert callable(ns.fn)
    assert ns.ordering == "both"
