"""DPO LoRA training planning."""

from __future__ import annotations

from .common import LoRATrainingPlanner


class DPOTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="dpo", label="DPO")
