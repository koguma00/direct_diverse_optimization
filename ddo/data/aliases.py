"""Paper-visible dataset-builder aliases."""

from __future__ import annotations

from .common import PairDatasetPlanner


class DivPOFreqPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo_freq_pairs", label="DivPO frequency-weighted pair", training_method="divpo_freq")


class DivPOProbPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo_prob_pairs", label="DivPO probability-weighted pair", training_method="divpo_prob")


class TieDPORKPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo_rk_pairs", label="TieDPO-RK tie-aware pair", training_method="tiedpo_rk")


class TieDPODavPairBuilder(PairDatasetPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo_dav_pairs", label="TieDPO-Dav tie-aware pair", training_method="tiedpo_dav")
