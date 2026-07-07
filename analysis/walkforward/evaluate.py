from __future__ import annotations

import math

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _average_precision_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Average Precision without sklearn dependency."""
    if y_true.size == 0:
        return float("nan")
    positives = int(y_true.sum())
    if positives == 0:
        return 0.0

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    ap = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
            ap += _safe_div(tp, tp + fp)
        else:
            fp += 1
    return ap / positives


def compute_predictive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if y_true.size == 0:
        nan = float("nan")
        return {
            "n": 0,
            "positive_rate": nan,
            "pr_auc": nan,
            "brier_score": nan,
            "mcc": nan,
            "balanced_accuracy": nan,
            "precision": nan,
            "recall": nan,
            "f1": nan,
        }

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + tnr)

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div((tp * tn) - (fp * fn), mcc_den) if mcc_den > 0 else 0.0

    return {
        "n": int(y_true.size),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(_average_precision_score(y_true, y_prob)),
        "brier_score": float(np.mean((y_prob - y_true) ** 2)),
        "mcc": float(mcc),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_metric_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    by_key: dict[tuple, dict[str, object]] = {}
    for row in rows:
        key = (
            row.get("model"),
            row.get("surface_id"),
            row.get("state_id"),
            row.get("baseline"),
        )
        bucket = by_key.setdefault(
            key,
            {
                "model": row.get("model"),
                "surface_id": row.get("surface_id"),
                "state_id": row.get("state_id"),
                "baseline": row.get("baseline"),
                "folds": 0,
                "pr_auc": [],
                "brier_score": [],
                "mcc": [],
                "balanced_accuracy": [],
                "precision": [],
                "recall": [],
                "f1": [],
            },
        )
        bucket["folds"] = int(bucket["folds"]) + 1
        for metric in ["pr_auc", "brier_score", "mcc", "balanced_accuracy", "precision", "recall", "f1"]:
            value = row.get(metric)
            if value is None:
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv):
                continue
            bucket[metric].append(fv)

    out: list[dict[str, object]] = []
    for bucket in by_key.values():
        row = {
            "model": bucket["model"],
            "surface_id": bucket["surface_id"],
            "state_id": bucket["state_id"],
            "baseline": bucket["baseline"],
            "folds": bucket["folds"],
        }
        for metric in ["pr_auc", "brier_score", "mcc", "balanced_accuracy", "precision", "recall", "f1"]:
            vals = bucket[metric]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out
