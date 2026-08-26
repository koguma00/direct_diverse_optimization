"""Stable in-repo artifact schemas.

These dataclasses define the boundary between benchmark adapters, collection
methods, dataset builders, trainers, evaluators, and report aggregators. The
actual serialized artifacts can be JSONL/JSON, but the fields here are the
project-owned contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrajectoryStep:
    step_index: int
    observation: str
    action: str
    raw_model_output: str | None = None
    reward: float | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrajectoryArtifact:
    benchmark: str
    task_id: str
    trajectory_id: str
    model_id: str
    seed: int
    won: bool
    score: float | None
    steps: list[TrajectoryStep]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchRecord:
    branch_id: str
    trajectory_id: str
    divergence_step: int
    action: str
    won: bool
    score: float | None
    response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BranchSetArtifact:
    benchmark: str
    task_id: str
    collection_method: str
    source_trajectory_id: str
    divergence_step: int
    prompt_text: str
    branches: list[BranchRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairRecord:
    prompt_text: str
    chosen_text: str
    rejected_text: str
    pair_type: str
    target_prob: float
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairDatasetArtifact:
    benchmark: str
    task_id: str
    collection_method: str
    dataset_method: str
    training_method: str
    records: list[PairRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
