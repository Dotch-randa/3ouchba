"""
training/train.py — Sequential backbone training manager.
Single GPU. Tqdm progress bar per epoch.
"""

import gc
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.factory import build_model, freeze_backbone, get_param_groups, unfreeze_all
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.dataset import PlantDiseaseDataset
from utils.metrics import FocalLoss, compute_class_weights, compute_metrics


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = True


# ---------------------------------------------------------------------------
# DataParallel helper
# ---------------------------------------------------------------------------

def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=6, min_delta=1e-4, checkpoint_path="best_model.pt"):
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_score      = None
        self.counter         = 0
        self.should_stop     = False

    def reset(self):
        self.counter     = 0
        self.should_stop = False

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.counter    = 0
            # FIX: unwrap DataParallel before state_dict() — otherwise saves
            # with 'module.' prefixes that break load_state_dict() at inference.
            torch.save(
                {"epoch": epoch, "model_state": unwrap(model).state_dict(), "score": score},
                self.checkpoint_path,
            )
            print(f"  ✓ New best Gen-Score: {score:.4f} — saved.")
            return True
        else:
            self.counter += 1
            print(f"  Early stopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    clip_grad_norm=1.0, epoch_label=""
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    n          = 0

    bar = tqdm(loader, desc=epoch_label, leave=False,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += bs

        bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.3f}")

    return total_loss / n, correct / n


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes) -> Dict:
    model.eval()
    total_loss            = 0.0
    all_preds, all_labels = [], []
    n                     = 0

    bar = tqdm(loader, desc="  val", leave=False,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        bs          = labels.size(0)
        total_loss += loss.item() * bs
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        n          += bs

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics         = compute_metrics(all_labels, all_preds, num_classes)
    metrics["loss"] = total_loss / n
    return metrics


# ---------------------------------------------------------------------------
# Per-backbone training routine
# ---------------------------------------------------------------------------

def train_backbone(
    backbone, num_classes, train_loader, val_loader, plantdoc_loader,
    class_counts, checkpoint_dir, device,
    freeze_epochs   = 5,
    unfreeze_epochs = 30,
    lr_backbone     = 1e-5,
    lr_head         = 4e-4,
    patience        = 6,
    focal_gamma     = 2.0,    # FIX: reverted from 2.5 — was hurting OOD generalisation
    label_smoothing = 0.05,   # FIX: reverted from 0.1 — was hurting OOD generalisation
    seed            = 42,
) -> Dict:
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f"  Training backbone: {backbone.upper()}")
    print(f"{'='*60}")

    model         = build_model(backbone, num_classes).to(device)
    class_weights = compute_class_weights(class_counts, device)
    criterion     = FocalLoss(gamma=focal_gamma, weight=class_weights, label_smoothing=label_smoothing)
    scaler        = GradScaler("cuda")
    best_path     = os.path.join(checkpoint_dir, f"{backbone}_best.pt")
    stopper       = EarlyStopping(patience=patience, checkpoint_path=best_path)

    history      = []
    global_epoch = 0

    # ── Phase 1: Head only ────────────────────────────────────────────────
    print(f"\n[Phase 1] Freeze backbone — {freeze_epochs} epochs (head only)")
    freeze_backbone(model, backbone)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_head, weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    for epoch in range(freeze_epochs):
        t0           = time.time()
        global_epoch += 1

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch_label=f"  [P1|E{global_epoch:03d}] train",
        )
        val_m = validate(model, val_loader,      criterion, device, num_classes)
        doc_m = validate(model, plantdoc_loader, criterion, device, num_classes)

        gen_score = _harmonic_mean(val_m["macro_f1"], doc_m["macro_f1"])
        scheduler.step(gen_score)

        row = _make_row(1, global_epoch, train_loss, train_acc, val_m, doc_m, gen_score, t0)
        history.append(row)
        _print_epoch(row)
        stopper.step(gen_score, model, global_epoch)
        if stopper.should_stop:
            print("  Early stopping triggered in Phase 1.")
            break

    # ── Phase 2: Full fine-tune ───────────────────────────────────────────
    print(f"\n[Phase 2] Unfreeze all — {unfreeze_epochs} epochs (discriminative LRs)")
    unfreeze_all(model)
    stopper.reset()

    param_groups = get_param_groups(model, backbone, lr_backbone, lr_head)
    optimizer    = AdamW(param_groups, weight_decay=1e-4)
    # FIX: reverted to ReduceLROnPlateau — CosineAnnealingWarmRestarts caused
    # PlantDoc F1 to decline steadily instead of improving. ReduceLROnPlateau
    # is more stable for OOD generalisation with this dataset.
    scheduler    = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    for epoch in range(unfreeze_epochs):
        t0           = time.time()
        global_epoch += 1

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch_label=f"  [P2|E{global_epoch:03d}] train",
        )
        val_m = validate(model, val_loader,      criterion, device, num_classes)
        doc_m = validate(model, plantdoc_loader, criterion, device, num_classes)

        gen_score = _harmonic_mean(val_m["macro_f1"], doc_m["macro_f1"])
        scheduler.step(gen_score)

        row = _make_row(2, global_epoch, train_loss, train_acc, val_m, doc_m, gen_score, t0)
        history.append(row)
        _print_epoch(row)
        stopper.step(gen_score, model, global_epoch)
        if stopper.should_stop:
            print("  Early stopping triggered in Phase 2.")
            break

    # Reload best — checkpoint saved without 'module.' prefixes so this
    # works regardless of whether DataParallel was used.
    ckpt = torch.load(best_path, map_location=device)
    unwrap(model).load_state_dict(ckpt["model_state"])

    summary = {
        "backbone":        backbone,
        "best_gen_score":  stopper.best_score,
        "checkpoint_path": best_path,
        "total_epochs":    global_epoch,
        "history":         history,
    }
    print(f"\n  Best Gen-Score : {stopper.best_score:.4f}")
    print(f"  Checkpoint     : {best_path}")

    del model, optimizer, scheduler, scaler, criterion, class_weights
    gc.collect()
    torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# Sequential training manager
# ---------------------------------------------------------------------------

def run_sequential_training(
    backbones, num_classes, train_loader, val_loader, plantdoc_loader,
    class_counts, checkpoint_dir="checkpoints", device=None, seed=42, **train_kwargs,
) -> List[Dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    all_summaries = []

    for backbone in backbones:
        summary = train_backbone(
            backbone=backbone, num_classes=num_classes,
            train_loader=train_loader, val_loader=val_loader,
            plantdoc_loader=plantdoc_loader, class_counts=class_counts,
            checkpoint_dir=checkpoint_dir, device=device, seed=seed,
            **train_kwargs,
        )
        all_summaries.append(summary)
        print(f"\n[Memory] Clearing GPU cache after {backbone}...")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    _print_summary_table(all_summaries)
    return all_summaries


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    train_csv, val_csv, plantdoc_csv, class_to_idx, class_mapping,
    image_root=None, batch_size=128, num_workers=0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = PlantDiseaseDataset(
        train_csv, class_to_idx, class_mapping,
        transform=get_train_transforms(), image_root=image_root,
    )
    val_ds = PlantDiseaseDataset(
        val_csv, class_to_idx, class_mapping,
        transform=get_val_transforms(), image_root=image_root,
    )
    doc_ds = PlantDiseaseDataset(
        plantdoc_csv, class_to_idx, class_mapping,
        transform=get_val_transforms(), image_root=image_root,
    )

    kw = dict(
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = torch.cuda.is_available(),
        persistent_workers = (num_workers > 0),
    )

    print(f"[Loaders] train={len(train_ds):,}  val={len(val_ds):,}  plantdoc={len(doc_ds):,}")
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(doc_ds,   shuffle=False, **kw),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b + 1e-6)


def _make_row(phase, epoch, train_loss, train_acc, val_m, doc_m, gen_score, t0):
    return {
        "phase": phase, "epoch": epoch,
        "train_loss": train_loss, "train_acc": train_acc,
        "val_macro_f1": val_m["macro_f1"], "val_weighted_f1": val_m["weighted_f1"],
        "plantdoc_macro_f1": doc_m["macro_f1"], "plantdoc_weighted_f1": doc_m["weighted_f1"],
        "gen_score": gen_score, "elapsed": time.time() - t0,
    }


def _print_epoch(row: dict) -> None:
    print(
        f"  [P{row['phase']}|E{row['epoch']:03d}] "
        f"loss={row['train_loss']:.4f} acc={row['train_acc']:.3f} | "
        f"val_f1={row['val_macro_f1']:.3f} | "
        f"plantdoc_f1={row['plantdoc_macro_f1']:.3f} | "
        f"gen={row['gen_score']:.3f}  ({row['elapsed']:.1f}s)"
    )


def _print_summary_table(summaries: list) -> None:
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Backbone':<20} {'Best Gen-Score':>16} {'Epochs':>8}")
    print("  " + "-" * 46)
    for s in summaries:
        print(f"  {s['backbone']:<20} {s['best_gen_score']:>16.4f} {s['total_epochs']:>8}")
    print("=" * 60)