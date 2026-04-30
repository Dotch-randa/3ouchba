"""
utils/metrics.py — Focal Loss, class weights, ECE, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # FIX: register as buffer so weight moves with .to(device) automatically
        # and is never left on CPU while the model is on GPU.
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability under AMP
        log_probs = F.log_softmax(logits.float(), dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Single-pass label-smoothed cross entropy with optional class weights
        num_classes = logits.size(1)
        smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce = -(smooth_targets * log_probs)
        if self.weight is not None:
            # FIX: move weight to logits device — guards against CPU/GPU mismatch
            # even if the loss was constructed before model.to(device) was called.
            ce = ce * self.weight.to(logits.device).unsqueeze(0)
        ce = ce.sum(dim=1)

        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def compute_ece(all_probs: np.ndarray, all_labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error. Measures gap between confidence and accuracy.
    Lower = better calibrated model.

    Args:
        all_probs:  (N, C) softmax probabilities.
        all_labels: (N,) ground-truth label indices.
        n_bins:     Number of confidence bins.

    Returns:
        ECE as a float in [0, 1].
    """
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracies  = (predictions == all_labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # FIX: use >= on lower bound so confidence=0.0 isn't silently excluded
        in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy   = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(class_counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Log-dampened inverse-frequency class weights, normalized to mean=1.

    Raw inverse frequency produces a 440x weight spread for this dataset
    (96 to 42,298 samples). Log-dampening reduces that to a manageable range
    without losing the imbalance correction signal.
    """
    counts  = torch.tensor(class_counts, dtype=torch.float32)
    weights = 1.0 / torch.log1p(counts)
    weights = weights / weights.mean()
    return weights.to(device)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    num_classes: int,
) -> dict:
    """
    Compute accuracy, macro-F1, weighted-F1, per-class F1, and confusion matrix.
    Passing labels=labels_range ensures absent classes are not silently
    dropped from the F1 average, which would inflate the score.
    """
    labels_range = list(range(num_classes))
    acc         = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average="macro",    labels=labels_range, zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", labels=labels_range, zero_division=0)
    # Added: per-class F1 helps identify which diseases the model struggles with
    per_class   = f1_score(all_labels, all_preds, average=None,       labels=labels_range, zero_division=0)
    cm          = confusion_matrix(all_labels, all_preds, labels=labels_range)

    return {
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_class_f1":     per_class,
        "confusion_matrix": cm,
    }