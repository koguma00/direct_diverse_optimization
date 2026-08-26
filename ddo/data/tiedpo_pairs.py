"""TieDPO pair dataset planning."""

from __future__ import annotations

from .common import PairDatasetPlanner


class TieDPOPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo_pairs", label="TieDPO tie-aware pair", training_method="tiedpo")
