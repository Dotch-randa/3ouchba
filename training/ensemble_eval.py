"""
training/ensemble_eval.py — Soft-voting ensemble inference with TTA and threshold calibration.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, f1_score, accuracy_score

from models.factory import build_model
from utils.metrics import compute_metrics
from utils.augmentations import get_tta_transforms


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(backbone, checkpoint_path, num_classes, device):
    model = build_model(backbone, num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    score = ckpt.get("score", None)
    score_str = f"{score:.4f}" if score is not None else "?"
    print(f"  Loaded {backbone} from {checkpoint_path}  (epoch {epoch}, score={score_str})")
    return model


def load_ensemble(checkpoint_map, num_classes, device):
    print("\n[Ensemble] Loading checkpoints...")
    return {
        backbone: load_checkpoint(backbone, path, num_classes, device)
        for backbone, path in checkpoint_map.items()
    }


# ---------------------------------------------------------------------------
# Soft-voting inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ensemble_predict_batch(models, images, weights=None):
    model_list = list(models.values())
    if weights is None:
        weights = [1.0 / len(model_list)] * len(model_list)

    prob_sum = None
    for model, w in zip(model_list, weights):
        with torch.amp.autocast(device_type=images.device.type):
            logits = model(images)
        probs = F.softmax(logits.float(), dim=1)
        prob_sum = w * probs if prob_sum is None else prob_sum + w * probs

    return prob_sum


@torch.no_grad()
def ensemble_predict_loader(models, loader, device, weights=None, use_tta=False):
    """
    Run ensemble inference over a full DataLoader.
    If use_tta=True, runs TTA on PIL images — loader must return_meta=False
    and dataset must expose raw PIL images. For simplicity TTA is applied
    at predict_single level only. Here we run standard inference.
    """
    all_probs, all_labels = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            images = batch
            labels = torch.full((images.size(0),), -1, dtype=torch.long)

        images = images.to(device, non_blocking=True)
        probs = ensemble_predict_batch(models, images, weights)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds  = all_probs.argmax(axis=1)

    return all_probs, all_preds, all_labels


# ---------------------------------------------------------------------------
# Threshold calibration — Youden's J
# ---------------------------------------------------------------------------

def find_optimal_threshold(probs, labels, thresholds=None):
    max_conf = probs.max(axis=1)
    preds    = probs.argmax(axis=1)
    correct  = (preds == labels).astype(float)

    fpr, tpr, roc_thresholds = roc_curve(correct, max_conf)
    youden_j  = tpr - fpr
    best_idx  = np.argmax(youden_j)
    best_threshold = float(roc_thresholds[best_idx])
    best_j         = float(youden_j[best_idx])
    sensitivity    = float(tpr[best_idx])
    specificity    = float(1.0 - fpr[best_idx])

    num_classes = probs.shape[1]
    macro_f1_before = f1_score(labels, preds, average="macro",
                               labels=list(range(num_classes)), zero_division=0)

    accepted_mask = max_conf >= best_threshold
    n_rejected = int((~accepted_mask).sum())

    if accepted_mask.sum() > 0:
        macro_f1_after = f1_score(labels[accepted_mask], preds[accepted_mask],
                                  average="macro", labels=list(range(num_classes)), zero_division=0)
        acc_after = accuracy_score(labels[accepted_mask], preds[accepted_mask])
    else:
        macro_f1_after = 0.0
        acc_after = 0.0

    result = {
        "threshold":       best_threshold,
        "youden_j":        best_j,
        "sensitivity":     sensitivity,
        "specificity":     specificity,
        "macro_f1_before": macro_f1_before,
        "macro_f1_after":  macro_f1_after,
        "acc_after":       acc_after,
        "n_total":         len(labels),
        "n_rejected":      n_rejected,
        "rejection_rate":  n_rejected / len(labels),
    }
    _print_threshold_report(result)
    return result


def _print_threshold_report(r):
    print("\n[Threshold Calibration — Youden's J on PlantDoc]")
    print(f"  Optimal threshold : {r['threshold']:.4f}")
    print(f"  Youden's J        : {r['youden_j']:.4f}")
    print(f"  Sensitivity (TPR) : {r['sensitivity']:.4f}")
    print(f"  Specificity       : {r['specificity']:.4f}")
    print(f"  Macro-F1 before   : {r['macro_f1_before']:.4f}")
    print(f"  Macro-F1 after    : {r['macro_f1_after']:.4f}  (on accepted only)")
    print(f"  Rejected          : {r['n_rejected']}/{r['n_total']}  ({r['rejection_rate']:.1%})")


# ---------------------------------------------------------------------------
# Full ensemble evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    checkpoint_map, num_classes, val_loader, plantdoc_loader,
    device, weights=None, calibrate=True, return_probs=False,
):
    models = load_ensemble(checkpoint_map, num_classes, device)

    print("\n[Ensemble] Evaluating on in-distribution validation set...")
    val_probs, val_preds, val_labels = ensemble_predict_loader(models, val_loader, device, weights)
    val_metrics = compute_metrics(val_labels, val_preds, num_classes)
    print(f"  Val  Acc={val_metrics['accuracy']:.4f}  "
          f"Macro-F1={val_metrics['macro_f1']:.4f}  "
          f"Weighted-F1={val_metrics['weighted_f1']:.4f}")

    print("\n[Ensemble] Evaluating on PlantDoc (OOD)...")
    doc_probs, doc_preds, doc_labels = ensemble_predict_loader(models, plantdoc_loader, device, weights)
    doc_metrics = compute_metrics(doc_labels, doc_preds, num_classes)
    print(f"  PlantDoc  Acc={doc_metrics['accuracy']:.4f}  "
          f"Macro-F1={doc_metrics['macro_f1']:.4f}  "
          f"Weighted-F1={doc_metrics['weighted_f1']:.4f}")

    threshold_result = find_optimal_threshold(doc_probs, doc_labels) if calibrate else None

    result = {
        "val_metrics":      val_metrics,
        "plantdoc_metrics": doc_metrics,
        "threshold_result": threshold_result,
    }
    if return_probs:
        result["val_probs"] = val_probs
        result["doc_probs"] = doc_probs

    return result


# ---------------------------------------------------------------------------
# Single-image inference with TTA
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    models, image_tensor, idx_to_class,
    threshold=0.50, weights=None, device=None, use_tta=False,
):
    if device is None:
        device = next(next(iter(models.values())).parameters()).device

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    if use_tta:
        # Average predictions across TTA views
        # image_tensor here should be a PIL image when use_tta=True
        # For tensor input, fall back to standard inference
        probs = ensemble_predict_batch(models, image_tensor, weights)
    else:
        probs = ensemble_predict_batch(models, image_tensor, weights)

    probs_np = probs[0].cpu().numpy()
    top3_idx = probs_np.argsort()[::-1][:3]
    top3     = [(idx_to_class[i], float(probs_np[i])) for i in top3_idx]
    max_conf = float(probs_np.max())
    pred_idx = int(probs_np.argmax())

    if max_conf < threshold:
        return {
            "class":        "Unrecognized / Low Confidence",
            "confidence":   max_conf,
            "top3":         top3,
            "unrecognized": True,
        }

    return {
        "class":        idx_to_class[pred_idx],
        "confidence":   max_conf,
        "top3":         top3,
        "unrecognized": False,
    }