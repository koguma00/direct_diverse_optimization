"""RTO training planning."""

from __future__ import annotations

from .common import LoRATrainingPlanner


class RTOTrainer(LoRATrainingPlanner):
    def __init__(self) -> None:
        super().__init__(name="rto", label="RTO")
