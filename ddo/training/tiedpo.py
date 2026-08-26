"""TieDPO LoRA training planning."""

from __future__ import annotations

from .common import LoRATrainingPlanner


class TieDPOTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="tiedpo", label="TieDPO")
