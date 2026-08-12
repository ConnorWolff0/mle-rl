from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .io import load_workload, write_result
from .simulator import run_simulation


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("candidate_serving_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args: argparse.Namespace) -> int:
    module = _load_module(Path(args.policy))
    policy_type = getattr(module, "Policy", None)
    if not isinstance(policy_type, type):
        raise RuntimeError(f"{args.policy} must export a Policy class")
    policy = policy_type()
    workload = load_workload(args.scenario)
    result = run_simulation(workload, policy)
    write_result(args.output, result.to_dict())
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m serving_sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one deterministic serving scenario")
    run.add_argument("scenario", help="input scenario JSON")
    run.add_argument("output", help="output report JSON")
    run.add_argument("--policy", required=True, help="Python file exporting Policy")
    run.set_defaults(handler=_run)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        return args.handler(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
