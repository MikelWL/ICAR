from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class IC9600Metrics:
    rmse: float
    rmae: float
    pearson: float
    spearman: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "rmse": float(self.rmse),
            "rmae": float(self.rmae),
            "pearson": float(self.pearson),
            "spearman": float(self.spearman),
        }


def compute_ic9600_metrics(predictions: np.ndarray, targets: np.ndarray) -> IC9600Metrics:
    """
    Metrics intended to match the ICNet-style reporting used in this repo.

    Note: `rmae` here is defined as `sqrt(mean(abs(error)))` (legacy ICNet-style).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    errors = predictions - targets
    rmae = float(np.sqrt(np.mean(np.abs(errors))))
    rmse = float(np.sqrt(np.mean(errors**2)))
    pearson = float(pearsonr(targets, predictions)[0])
    spearman = float(spearmanr(targets, predictions)[0])
    return IC9600Metrics(rmse=rmse, rmae=rmae, pearson=pearson, spearman=spearman)
