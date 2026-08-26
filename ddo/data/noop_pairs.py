"""No-op dataset planner for base and SFT-reference runs."""

from __future__ import annotations

from .common import PairDatasetPlanner


class NoOpPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="noop_pairs", label="no-op", training_method="base")
