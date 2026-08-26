"""DivPO pair dataset planning."""

from __future__ import annotations

from .common import PairDatasetPlanner


class DivPOPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo_pairs", label="DivPO diversity-weighted pair", training_method="divpo")
