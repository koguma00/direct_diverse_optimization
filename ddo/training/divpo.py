"""DivPO LoRA training planning."""

from __future__ import annotations

from .common import LoRATrainingPlanner


class DivPOTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="divpo", label="DivPO")
