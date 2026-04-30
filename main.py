import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from training.train import set_seed, make_loaders, run_sequential_training
from training.ensemble_eval import evaluate_ensemble, load_ensemble, predict_single
from utils.dataset import PlantDiseaseDataset
from utils.augmentations import get_val_transforms


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


CONFIG = {
    "train_csv":        "/kaggle/working/splits/train.csv",
    "val_csv":          "/kaggle/working/splits/val.csv",
    "plantdoc_csv":     "/kaggle/working/splits/plantdoc_val.csv",
    "class_mapping":    "/kaggle/working/class_mapping.json",
    "image_root":       None,
    "checkpoint_dir":   "/kaggle/working/checkpoints",
    "results_dir":      "/kaggle/working/results",

    # ── Train all 3 backbones for ensemble ──────────────────────────────
    "backbones":        [ "resnet50", "mobilenet_v3_small"], #"efficientnet_b0",

    "batch_size":       128,
    "num_workers":      2,
    "freeze_epochs":    5,       # increased from 3
    "unfreeze_epochs":  40,
    "lr_backbone":      1e-5,
    "lr_head":          4e-4,
    "focal_gamma":      2.0,
    "label_smoothing":  0.05,
    "patience":         6,
    "seed":             42,
    "min_samples":      100,

    # ── Lower threshold — reduces 59% rejection rate ─────────────────────
    "ensemble_weights":  None,
    "default_threshold": 0.50,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("--- Environment Setup ---")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["results_dir"],    exist_ok=True)

    # 1. Class mapping
    print("\n--- Loading Class Mappings ---")
    class_mapping = PlantDiseaseDataset.load_class_mapping(CONFIG["class_mapping"])
    class_to_idx  = PlantDiseaseDataset.build_class_to_idx(
        csv_paths=[CONFIG["train_csv"], CONFIG["val_csv"]],
        class_mapping=class_mapping,
        min_samples=CONFIG["min_samples"],
    )
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)
    print(f"Number of classes: {num_classes}")

    with open(os.path.join(CONFIG["results_dir"], "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2, cls=NumpyEncoder)

    # 2. DataLoaders
    print("\n--- Initializing DataLoaders ---")
    train_loader, val_loader, plantdoc_loader = make_loaders(
        train_csv=CONFIG["train_csv"],
        val_csv=CONFIG["val_csv"],
        plantdoc_csv=CONFIG["plantdoc_csv"],
        class_to_idx=class_to_idx,
        class_mapping=class_mapping,
        image_root=CONFIG["image_root"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # 3. Training
    if not args.eval_only:
        print("\n--- Starting Sequential Training ---")
        train_df = pd.read_csv(CONFIG["train_csv"])
        train_df["label"] = train_df["label"].map(
            lambda x: class_mapping.get(x, x) if isinstance(x, str) else "unknown"
        )
        class_counts = [
            int((train_df["label"] == cls).sum())
            for cls in sorted(class_to_idx, key=class_to_idx.get)
        ]

        summaries = run_sequential_training(
            backbones=CONFIG["backbones"],
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            plantdoc_loader=plantdoc_loader,
            class_counts=class_counts,
            checkpoint_dir=CONFIG["checkpoint_dir"],
            device=device,
            seed=CONFIG["seed"],
            freeze_epochs=CONFIG["freeze_epochs"],
            unfreeze_epochs=CONFIG["unfreeze_epochs"],
            lr_backbone=CONFIG["lr_backbone"],
            lr_head=CONFIG["lr_head"],
            focal_gamma=CONFIG["focal_gamma"],
            label_smoothing=CONFIG["label_smoothing"],
            patience=CONFIG["patience"],
        )

        clean_summaries = copy.deepcopy(summaries)
        for s in clean_summaries:
            for row in s.get("history", []):
                row.pop("confusion_matrix", None)

        with open(os.path.join(CONFIG["results_dir"], "training_summaries.json"), "w") as f:
            json.dump(clean_summaries, f, indent=2, cls=NumpyEncoder)

    # 4. Evaluation
    print("\n--- Evaluation & Thresholding ---")
    checkpoint_map = {
        b: os.path.join(CONFIG["checkpoint_dir"], f"{b}_best.pt")
        for b in CONFIG["backbones"]
    }

    for b, path in checkpoint_map.items():
        if not os.path.exists(path):
            print(f"[ERROR] Checkpoint not found: {path}")
            return

    results = evaluate_ensemble(
        checkpoint_map=checkpoint_map,
        num_classes=num_classes,
        val_loader=val_loader,
        plantdoc_loader=plantdoc_loader,
        device=device,
        calibrate=(args.threshold is None),
    )

    if args.threshold is not None:
        final_threshold = args.threshold
    elif results.get("threshold_result") is not None:
        final_threshold = results["threshold_result"]["threshold"]
    else:
        final_threshold = CONFIG["default_threshold"]

    # Clamp threshold — never go above 0.70 to avoid excessive rejection
    final_threshold = min(final_threshold, 0.70)
    print(f"\n[Inference] Final confidence threshold: {final_threshold:.4f}")

    with open(os.path.join(CONFIG["results_dir"], "threshold.json"), "w") as f:
        json.dump({"threshold": final_threshold}, f, indent=2, cls=NumpyEncoder)

    # 5. Demo
    models = load_ensemble(checkpoint_map, num_classes, device)
    _demo_prediction(models, plantdoc_loader, idx_to_class, final_threshold, device)
    print(f"\nPipeline Complete. Saved to: {CONFIG['results_dir']}")


def _demo_prediction(models, loader, idx_to_class, threshold, device):
    try:
        batch = next(iter(loader))
        images, labels = batch[0], batch[1]
        result = predict_single(models, images[0], idx_to_class, threshold, device=device)
        print(f"\n[Demo Result] Prediction: {result['class']} | Conf: {result['confidence']:.4f}")
        print("  Top-3:")
        for cls, conf in result["top3"]:
            print(f"    {cls:<40} {conf:.4f}")
    except StopIteration:
        print("\n[Demo] Skipped: loader is empty.")
    except Exception as e:
        print(f"[Demo] Failed: {e}")


if __name__ == "__main__":
    main()