"""Paper-visible training method aliases."""

from __future__ import annotations

from .common import LoRATrainingPlanner


class BaseModelTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="base", label="base model")


class ReferenceSFTTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="reference", label="SFT reference")


class DDOTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="ddo", label="DDO")


class DivPOFreqTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo_freq", label="DivPO-freq")


class DivPOProbTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo_prob", label="DivPO-prob")


class TieDPORKTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo_rk", label="TieDPO-RK")


class TieDPODavTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo_dav", label="TieDPO-Dav")
