#!/usr/bin/env python3
"""Evaluate one released paper checkpoint from a single YAML config."""

from __future__ import annotations

import argparse

from ddo.config.schema import ConfigError
from ddo.pipeline.runner import PipelineRunner
from ddo.release import load_evaluation_config


def _print_stage(plan) -> None:
    print(f"[{plan.stage}] {plan.summary}", flush=True)
    for output in plan.outputs:
        print(f"  output: {output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        config = load_evaluation_config(args.config, dry_run=args.dry_run)
        runner = PipelineRunner(config)
        plan = runner.plan_stage("eval")
        _print_stage(plan)
        if not args.dry_run:
            runner.execute_stage("eval")
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
