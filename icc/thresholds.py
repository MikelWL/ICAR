from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional

import torch

ThresholdMetric = Literal["f1", "balanced_accuracy", "accuracy", "youden_j"]


@dataclass(frozen=True)
class ThresholdTuningResult:
    best_threshold: float
    best_metric_value: float
    metric: ThresholdMetric
    table: Dict[str, list]


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den > 0, num / den, torch.zeros_like(num))


@torch.no_grad()
def tune_binary_threshold(
    probs_pos: torch.Tensor,
    targets: torch.Tensor,
    *,
    metric: ThresholdMetric = "f1",
    thresholds: Optional[torch.Tensor] = None,
    prefer_higher_threshold_on_tie: bool = True,
) -> ThresholdTuningResult:
    """
    Tune a probability threshold for binary classification.

    Args:
        probs_pos: Positive-class probabilities, shape [N], values in [0, 1].
        targets: Binary targets, shape [N], values in {0, 1}.
        metric: Which metric to maximize.
        thresholds: Optional threshold grid, shape [T]. Defaults to 0..1 in steps of 0.001.
        prefer_higher_threshold_on_tie: If True, tie-breaker prefers higher threshold (fewer positives).
    """
    probs_pos = probs_pos.detach().float().flatten().cpu()
    targets = targets.detach().long().flatten().cpu()
    if probs_pos.numel() != targets.numel():
        raise ValueError("probs_pos and targets must have the same number of elements.")

    if thresholds is None:
        thresholds = torch.linspace(0.0, 1.0, 1001)
    thresholds = thresholds.detach().float().flatten().cpu()

    preds = probs_pos[None, :] > thresholds[:, None]  # [T, N]
    t = targets[None, :] == 1
    n = ~t

    tp = (preds & t).sum(dim=1).float()
    fp = (preds & n).sum(dim=1).float()
    fn = ((~preds) & t).sum(dim=1).float()
    tn = ((~preds) & n).sum(dim=1).float()

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    tpr = recall
    fpr = _safe_div(fp, fp + tn)
    youden_j = tpr - fpr
    balanced_accuracy = 0.5 * (tpr + _safe_div(tn, tn + fp))

    metric_values = {
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "youden_j": youden_j,
    }[metric]

    best_value = float(metric_values.max().item()) if metric_values.numel() else 0.0
    best_idxs = (metric_values == metric_values.max()).nonzero(as_tuple=False).flatten()
    if best_idxs.numel() == 0:
        best_threshold = 0.5
    else:
        if prefer_higher_threshold_on_tie:
            best_idx = int(best_idxs.max().item())
        else:
            best_idx = int(best_idxs.min().item())
        best_threshold = float(thresholds[best_idx].item())

    table = {
        "threshold": [float(x) for x in thresholds.tolist()],
        "tp": [int(x) for x in tp.tolist()],
        "fp": [int(x) for x in fp.tolist()],
        "tn": [int(x) for x in tn.tolist()],
        "fn": [int(x) for x in fn.tolist()],
        "precision": [float(x) for x in precision.tolist()],
        "recall": [float(x) for x in recall.tolist()],
        "f1": [float(x) for x in f1.tolist()],
        "accuracy": [float(x) for x in accuracy.tolist()],
        "balanced_accuracy": [float(x) for x in balanced_accuracy.tolist()],
        "youden_j": [float(x) for x in youden_j.tolist()],
    }

    return ThresholdTuningResult(
        best_threshold=best_threshold,
        best_metric_value=best_value,
        metric=metric,
        table=table,
    )

