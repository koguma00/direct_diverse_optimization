"""BabyAI benchmark adapter."""

from __future__ import annotations

from .base import BenchmarkAdapter


class BabyAIAdapter(BenchmarkAdapter):
    name = "babyai"
    upstream_dir_name = "BALROG"
    default_task_filter = "BabyAI-MixedTrainLocal-v0/goto"
