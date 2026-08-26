#!/usr/bin/env python3
"""Summarize WebShop success and strategy diversity metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ddo.evaluation.webshop.common import (
    WEBSHOP_SUCCESS_THRESHOLD,
    action_type_signature,
    read_json,
    trace_success,
    write_json,
)


DIVERSITY_PURCHASE_SIGNATURE = "purchase_signature"
DIVERSITY_ACTION_TRACE = "action_trace"
DIVERSITY_ACTION_TYPE_TRACE = "action_type_trace"
DIVERSITY_PURCHASE_ACTION_TYPE_TRACE = "purchase_action_type_trace"
SUPPORTED_DIVERSITY_KEYS = (
    DIVERSITY_PURCHASE_SIGNATURE,
    DIVERSITY_ACTION_TRACE,
    DIVERSITY_ACTION_TYPE_TRACE,
    DIVERSITY_PURCHASE_ACTION_TYPE_TRACE,
)


class MissingPurchaseSignatureError(ValueError):
    """Raised when a successful composite-class rollout lacks purchase identity."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a WebShop direct or DTC run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--success-threshold", type=float, default=WEBSHOP_SUCCESS_THRESHOLD)
    parser.add_argument("--diversity-key", choices=SUPPORTED_DIVERSITY_KEYS, default=DIVERSITY_PURCHASE_ACTION_TYPE_TRACE)
    parser.add_argument("--trajectory-kind", choices=("all", "base", "alt"), default="all")
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--csv-out", default=None)
    return parser.parse_args()


def _trace_paths(run_dir: Path) -> list[Path]:
    return sorted(Path(run_dir).glob("**/*_llm_trace.json"))


def _trajectory_kind(payload: dict[str, Any], path: Path) -> str:
    kind = str(payload.get("trajectory_kind") or "").strip()
    if kind:
        return kind
    extras = payload.get("extras") if isinstance(payload.get("extras"), dict) else {}
    dtc = extras.get("dtc") if isinstance(extras.get("dtc"), dict) else {}
    kind = str(dtc.get("trajectory_kind") or "").strip()
    if kind:
        return kind
    episode_id = str(payload.get("episode_id") or path.stem)
    return "alt" if "__dtc_" in episode_id else "base"


def _actions(payload: dict[str, Any]) -> list[str]:
    calls = payload.get("calls") or []
    if not isinstance(calls, list):
        return []
    return [str(call.get("action") or "").strip() for call in calls if isinstance(call, dict)]


def _trace_aborted(payload: dict[str, Any]) -> bool:
    calls = payload.get("calls") or []
    if not isinstance(calls, list):
        return False
    for call in calls:
        if not isinstance(call, dict):
            continue
        if str(call.get("termination_reason") or "").strip() == "aborted":
            return True
        extras = call.get("extras") if isinstance(call.get("extras"), dict) else {}
        if extras.get("abort_reason"):
            return True
    return False


def _purchase_signature(payload: dict[str, Any]) -> str:
    extras = payload.get("extras") if isinstance(payload.get("extras"), dict) else {}
    signature = str(extras.get("purchase_signature") or "").strip()
    if signature:
        return signature
    calls = payload.get("calls") or []
    if not isinstance(calls, list):
        return ""
    for call in reversed(calls):
        if not isinstance(call, dict):
            continue
        call_extras = call.get("extras") if isinstance(call.get("extras"), dict) else {}
        webshop = call_extras.get("webshop") if isinstance(call_extras.get("webshop"), dict) else {}
        signature = str(webshop.get("purchase_signature") or "").strip()
        if signature:
            return signature
    return ""


def _canonical_purchase_signature(signature: str) -> str:
    """Canonicalize structured signatures while preserving legacy opaque values."""

    signature = str(signature or "").strip()
    if not signature:
        return ""
    try:
        payload = json.loads(signature)
    except (json.JSONDecodeError, TypeError):
        return signature
    if not isinstance(payload, dict):
        return signature
    asin = str(payload.get("asin") or "").strip().upper()
    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    if not asin:
        return signature
    return json.dumps(
        {"asin": asin, "options": options},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _strategy_key(payload: dict[str, Any], diversity_key: str) -> str:
    actions = _actions(payload)
    if diversity_key == DIVERSITY_ACTION_TRACE:
        return json.dumps(actions, ensure_ascii=True)
    if diversity_key == DIVERSITY_ACTION_TYPE_TRACE:
        return action_type_signature(actions)
    if diversity_key == DIVERSITY_PURCHASE_ACTION_TYPE_TRACE:
        signature = _canonical_purchase_signature(_purchase_signature(payload))
        if not signature:
            raise MissingPurchaseSignatureError(
                "purchase_action_type_trace requires a purchase_signature"
            )
        return json.dumps(
            {
                "action_type_trace": action_type_signature(actions),
                "purchase_signature": signature,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    if diversity_key != DIVERSITY_PURCHASE_SIGNATURE:
        raise ValueError(f"Unsupported diversity_key: {diversity_key}")
    signature = _purchase_signature(payload)
    if signature:
        return signature
    return json.dumps(actions, ensure_ascii=True)


def _session_id(payload: dict[str, Any]) -> int | None:
    value = payload.get("session_id")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    try:
        episode_idx = int(payload.get("episode_idx"))
    except (TypeError, ValueError):
        return None
    return episode_idx // 100



def _provenance(run_dir: Path, trace_paths: list[Path], diversity_key: str) -> dict[str, Any]:
    implementation_path = Path(__file__).resolve()
    return {
        "class_key": diversity_key,
        "input_trace_count": len(trace_paths),
        "metric_implementation": str(implementation_path.relative_to(REPO_ROOT)),
    }


def summary_output_paths(run_dir: Path, diversity_key: str) -> tuple[Path, Path]:
    suffix = "" if diversity_key == DIVERSITY_PURCHASE_SIGNATURE else f".{diversity_key}"
    stem = f"webshop_diversity_summary{suffix}"
    return run_dir / f"{stem}.json", run_dir / f"{stem}.csv"


def _entropy_effective_count(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return math.exp(entropy)


def summarize_run(
    *,
    run_dir: Path,
    success_threshold: float,
    diversity_key: str,
    trajectory_kind: str = "all",
    session_id: int | None = None,
) -> dict[str, Any]:
    if diversity_key not in SUPPORTED_DIVERSITY_KEYS:
        raise ValueError(f"Unsupported diversity_key: {diversity_key}")

    records: list[tuple[Path, dict[str, Any], str, bool, bool, str, list[str]]] = []
    for trace_path in _trace_paths(run_dir):
        payload = read_json(trace_path)
        if str(payload.get("env_name") or "") != "webshop":
            continue
        if session_id is not None and _session_id(payload) != int(session_id):
            continue
        kind = _trajectory_kind(payload, trace_path)
        if trajectory_kind != "all" and kind != trajectory_kind:
            continue
        records.append(
            (
                trace_path,
                payload,
                kind,
                trace_success(payload, success_threshold),
                _trace_aborted(payload),
                _purchase_signature(payload),
                _actions(payload),
            )
        )

    missing_success_paths = [
        trace_path
        for trace_path, _payload, _kind, success, aborted, signature, _actions_value in records
        if success and not aborted and not signature
    ]
    if diversity_key == DIVERSITY_PURCHASE_ACTION_TYPE_TRACE and missing_success_paths:
        examples = ", ".join(str(path) for path in missing_success_paths[:5])
        raise MissingPurchaseSignatureError(
            "purchase_action_type_trace cannot classify successful rollouts: "
            f"missing_purchase_signature_count={len(missing_success_paths)}; traces={examples}"
        )

    rows: list[dict[str, Any]] = []
    selected_trace_paths: list[Path] = []
    for trace_path, payload, kind, success, aborted, signature, actions in records:
        selected_trace_paths.append(trace_path)
        key = None
        if diversity_key != DIVERSITY_PURCHASE_ACTION_TYPE_TRACE or signature:
            key = _strategy_key(payload, diversity_key)
        rows.append(
            {
                "trace_path": str(trace_path),
                "episode_id": str(payload.get("episode_id") or trace_path.stem.replace("_llm_trace", "")),
                "trajectory_kind": kind,
                "session_id": _session_id(payload),
                "success": success,
                "aborted": aborted,
                "strategy_key": key,
                "purchase_signature": signature,
                "action_type_signature": action_type_signature(actions),
                "action_count": len(actions),
            }
        )

    total_count = len(rows)
    non_aborted_rows = [row for row in rows if not row["aborted"]]
    non_aborted_count = len(non_aborted_rows)
    aborted_count = total_count - non_aborted_count
    success_rows = [row for row in non_aborted_rows if row["success"]]
    success_count = len(success_rows)
    success_counts = Counter(row["strategy_key"] for row in success_rows)
    all_counts = Counter(
        row["strategy_key"] for row in non_aborted_rows if row["strategy_key"] is not None
    )
    unique_success_count = len(success_counts)
    unique_total_count = len(all_counts)
    top_success_count = max(success_counts.values(), default=0)
    entropy_effective_success_count = _entropy_effective_count(success_counts)

    return {
        "run_dir": str(run_dir),
        "success_threshold": float(success_threshold),
        "diversity_key": diversity_key,
        "class_key_definition": {
            "name": diversity_key,
            "components": ["action_type_signature", "purchase_signature"]
            if diversity_key == DIVERSITY_PURCHASE_ACTION_TYPE_TRACE
            else [diversity_key],
            "missing_success_purchase_signature_policy": "error"
            if diversity_key == DIVERSITY_PURCHASE_ACTION_TYPE_TRACE
            else "legacy_mode",
        },
        "trajectory_kind": trajectory_kind,
        "session_id": session_id,
        "trajectory_count": total_count,
        "non_aborted_count": non_aborted_count,
        "aborted_count": aborted_count,
        "success_count": success_count,
        "success_at_k": success_count / total_count if total_count else 0.0,
        "success_at_non_aborted": success_count / non_aborted_count if non_aborted_count else 0.0,
        "unique_success_strategy_count": unique_success_count,
        "entropy_effective_success_strategy_count": entropy_effective_success_count,
        "h_esd": entropy_effective_success_count / total_count if total_count else 0.0,
        "esd": unique_success_count / total_count if total_count else 0.0,
        "esd_non_aborted": unique_success_count / non_aborted_count if non_aborted_count else 0.0,
        "wsd": unique_success_count / success_count if success_count else 0.0,
        "h_wsd": entropy_effective_success_count / success_count if success_count else 0.0,
        "top1_share": top_success_count / success_count if success_count else 0.0,
        "unique_total_strategy_count": unique_total_count,
        "total_div": unique_total_count / total_count if total_count else 0.0,
        "total_div_non_aborted": unique_total_count / non_aborted_count if non_aborted_count else 0.0,
        "missing_purchase_signature_count": sum(1 for row in rows if not row["purchase_signature"]),
        "missing_success_purchase_signature_count": len(missing_success_paths),
        "undefined_strategy_key_count": sum(1 for row in rows if row["strategy_key"] is None),
        "provenance": _provenance(run_dir, selected_trace_paths, diversity_key),
        "strategy_counts": dict(success_counts),
        "trajectories": rows,
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        key: value
        for key, value in summary.items()
        if key not in {"strategy_counts", "trajectories"}
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _parse_args()
    summary = summarize_run(
        run_dir=Path(args.run_dir),
        success_threshold=args.success_threshold,
        diversity_key=args.diversity_key,
        trajectory_kind=args.trajectory_kind,
        session_id=args.session_id,
    )
    default_json_out, default_csv_out = summary_output_paths(Path(args.run_dir), args.diversity_key)
    json_out = Path(args.json_out) if args.json_out else default_json_out
    csv_out = Path(args.csv_out) if args.csv_out else default_csv_out
    write_json(json_out, summary)
    _write_csv(csv_out, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "trajectories"}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
