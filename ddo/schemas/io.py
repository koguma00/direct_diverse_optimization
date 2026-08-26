"""Serialization helpers for project-owned artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    BranchRecord,
    BranchSetArtifact,
    PairDatasetArtifact,
    PairRecord,
    TrajectoryArtifact,
    TrajectoryStep,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_plain(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_plain(record), sort_keys=True))
            handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object at {path}")
    return loaded


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"expected JSON object row at {path}")
            rows.append(loaded)
    return rows


def write_trajectories(path: Path, trajectories: list[TrajectoryArtifact]) -> None:
    write_jsonl(path, trajectories)


def read_trajectories(path: Path) -> list[TrajectoryArtifact]:
    return [_trajectory_from_mapping(row) for row in read_jsonl(path)]


def write_branch_sets(path: Path, branch_sets: list[BranchSetArtifact]) -> None:
    write_jsonl(path, branch_sets)


def read_branch_sets(path: Path) -> list[BranchSetArtifact]:
    return [_branch_set_from_mapping(row) for row in read_jsonl(path)]


def write_pair_dataset(path: Path, dataset: PairDatasetArtifact) -> None:
    write_json(path, dataset)


def read_pair_dataset(path: Path) -> PairDatasetArtifact:
    raw = read_json(path)
    return PairDatasetArtifact(
        benchmark=str(raw["benchmark"]),
        task_id=str(raw["task_id"]),
        collection_method=str(raw["collection_method"]),
        dataset_method=str(raw["dataset_method"]),
        training_method=str(raw["training_method"]),
        records=[
            PairRecord(
                prompt_text=str(row["prompt_text"]),
                chosen_text=str(row["chosen_text"]),
                rejected_text=str(row["rejected_text"]),
                pair_type=str(row["pair_type"]),
                target_prob=float(row["target_prob"]),
                weight=float(row.get("weight", 1.0)),
                metadata=dict(row.get("metadata", {})),
            )
            for row in raw.get("records", [])
        ],
        metadata=dict(raw.get("metadata", {})),
    )


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _trajectory_from_mapping(raw: dict[str, Any]) -> TrajectoryArtifact:
    return TrajectoryArtifact(
        benchmark=str(raw["benchmark"]),
        task_id=str(raw["task_id"]),
        trajectory_id=str(raw["trajectory_id"]),
        model_id=str(raw["model_id"]),
        seed=int(raw["seed"]),
        won=bool(raw["won"]),
        score=None if raw.get("score") is None else float(raw["score"]),
        steps=[
            TrajectoryStep(
                step_index=int(step["step_index"]),
                observation=str(step["observation"]),
                action=str(step["action"]),
                raw_model_output=step.get("raw_model_output"),
                reward=None if step.get("reward") is None else float(step["reward"]),
                done=bool(step.get("done", False)),
                metadata=dict(step.get("metadata", {})),
            )
            for step in raw.get("steps", [])
        ],
        metadata=dict(raw.get("metadata", {})),
    )


def _branch_set_from_mapping(raw: dict[str, Any]) -> BranchSetArtifact:
    return BranchSetArtifact(
        benchmark=str(raw["benchmark"]),
        task_id=str(raw["task_id"]),
        collection_method=str(raw["collection_method"]),
        source_trajectory_id=str(raw["source_trajectory_id"]),
        divergence_step=int(raw["divergence_step"]),
        prompt_text=str(raw["prompt_text"]),
        branches=[
            BranchRecord(
                branch_id=str(branch["branch_id"]),
                trajectory_id=str(branch["trajectory_id"]),
                divergence_step=int(branch["divergence_step"]),
                action=str(branch["action"]),
                won=bool(branch["won"]),
                score=None if branch.get("score") is None else float(branch["score"]),
                response_text=str(branch["response_text"]),
                metadata=dict(branch.get("metadata", {})),
            )
            for branch in raw.get("branches", [])
        ],
        metadata=dict(raw.get("metadata", {})),
    )
