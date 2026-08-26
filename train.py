#!/usr/bin/env python3
"""Train one paper model from a single YAML config."""

from __future__ import annotations

import argparse

from ddo.config.schema import ConfigError
from ddo.pipeline.runner import ALL_STAGES, PipelineRunner
from ddo.release import load_training_config, training_stages, validate_training_inputs


def _print_stage(plan) -> None:
    print(f"[{plan.stage}] {plan.summary}", flush=True)
    for output in plan.outputs:
        label = "work" if plan.stage in {"sft", "train"} else "output"
        print(f"  {label}: {output}", flush=True)
    for key in ("reference_checkpoint", "output_checkpoint"):
        if key in plan.metadata:
            print(f"  checkpoint: {plan.metadata[key]}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument(
        "--stage",
        action="append",
        choices=ALL_STAGES[:-1],
        help=(
            "Run only this training stage; repeat to run several stages in order. "
            "By default, run the stages implied by start_from."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        config = load_training_config(args.config, dry_run=args.dry_run)
        if not args.dry_run:
            validate_training_inputs(config)
        runner = PipelineRunner(config)
        stages = tuple(args.stage) if args.stage else training_stages(config)
        for stage in stages:
            plan = runner.plan_stage(stage)
            _print_stage(plan)
            if not args.dry_run:
                runner.execute_stage(stage)
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
