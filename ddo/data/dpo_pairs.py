"""DPO pair dataset planning."""

from __future__ import annotations

from .common import PairDatasetPlanner


class DPOPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="dpo_pairs", label="DPO win-lose pair", training_method="dpo")
