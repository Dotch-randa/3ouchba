================================================================================
PLANT DISEASE CLASSIFIER — BUG FIX INSTRUCTIONS
For: NABTA Hackathon | PyTorch Soft-Voting Ensemble
================================================================================

You are an expert PyTorch ML engineer. Below is a complete list of bugs and
issues found across 5 files of a plant disease classifier project. Fix each
one exactly as described. Do not change anything not listed here.

--------------------------------------------------------------------------------
FILE 1: utils/augmentations.py
--------------------------------------------------------------------------------

BUG 1 — GaussianBlur applied unconditionally (p=1.0)
SEVERITY: Medium — hurts disease texture learning
CURRENT:
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
FIX — wrap in RandomApply with p=0.5:
    T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),

BUG 2 — Redundant scale in RandomAffine (already handled by RandomResizedCrop)
SEVERITY: Minor — wasteful, no benefit
CURRENT:
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), fill=128),
FIX — remove the scale argument:
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), fill=128),

--------------------------------------------------------------------------------
FILE 2: utils/metrics.py
--------------------------------------------------------------------------------

BUG 3 — CRITICAL: compute_ece is indented inside FocalLoss class with no self parameter
SEVERITY: Critical — calling it will pass all_probs as self and crash
CURRENT: compute_ece is defined as a method inside FocalLoss (wrong indentation level)
FIX — move compute_ece completely outside the FocalLoss class as a standalone
module-level function. It should be at the same indentation level as
compute_class_weights and compute_metrics. No other changes to the function body.

BUG 4 — CRITICAL: Entire file content is duplicated
SEVERITY: Critical — second definition silently overwrites first
FIX — delete the second full copy of the file. Keep only one copy of every
class and function. The file should contain each of these exactly once:
  - imports
  - class FocalLoss
  - function compute_ece (moved outside class per BUG 3)
  - function compute_class_weights
  - function compute_metrics

BUG 5 — CRITICAL: Raw inverse-frequency class weights cause training instability
SEVERITY: Critical — 440x imbalance (96 to 42,298 samples) will make loss
dominated by the 4 bottom classes and destabilize training
CURRENT:
    def compute_class_weights(class_counts, device):
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / counts
        weights = weights / weights.mean()
        return weights.to(device)
FIX — replace raw inverse frequency with log-dampened inverse frequency:
    def compute_class_weights(class_counts: List[int], device: torch.device) -> torch.Tensor:
        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / torch.log1p(counts)   # log-dampened to reduce 440x ratio
        weights = weights / weights.mean()     # normalize to mean=1
        return weights.to(device)

BUG 6 — Medium: Double softmax in FocalLoss.forward causes numerical inconsistency under AMP
SEVERITY: Medium — (1-pt)^gamma and ce are computed from slightly different
probability estimates when using float16 AMP
CURRENT forward() logic:
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    ce = F.cross_entropy(logits, targets, weight=self.weight,
                         reduction="none", label_smoothing=self.label_smoothing)
    focal = (1.0 - pt) ** self.gamma * ce
FIX — replace the entire forward() method with a single-pass implementation:
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=1)  # cast to float32 for stability
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        num_classes = logits.size(1)
        smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        ce = -(smooth_targets * log_probs)
        if self.weight is not None:
            ce = ce * self.weight.unsqueeze(0)
        ce = ce.sum(dim=1)

        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

BUG 7 — Minor: Unused import
CURRENT:
    from typing import List, Tuple
FIX:
    from typing import List

--------------------------------------------------------------------------------
FILE 3: main.py
--------------------------------------------------------------------------------

BUG 8 — CRITICAL: Wrong import casing for Augmentations
SEVERITY: Critical — crashes on Linux/Kaggle (case-sensitive filesystem)
CURRENT:
    from utils.Augmentations import get_val_transforms  # capital A
FIX — change to lowercase to match the actual filename augmentations.py:
    from utils.augmentations import get_val_transforms

BUG 9 — Medium: class_counts computation is fragile and split from class_to_idx logic
SEVERITY: Medium — if filtering logic in build_class_to_idx ever diverges from
the inline pandas block in main.py, loss weights will silently use wrong counts
CURRENT: class_counts is computed in main.py by re-reading and re-filtering
the CSV manually after build_class_to_idx has already been called.
FIX — move class_counts computation inside make_loaders() in train.py so it
is computed from the same filtered dataset object used for training. Return it
as a fourth value: (train_loader, val_loader, doc_loader, class_counts).
Then in main.py replace the inline pandas block with:
    train_loader, val_loader, plantdoc_loader, class_counts = make_loaders(...)

--------------------------------------------------------------------------------
FILE 4: models/factory.py
--------------------------------------------------------------------------------

BUG 10 — Medium: MobileNetV3 head has no Dropout unlike the other two backbones
SEVERITY: Medium — MobileNetV3-Small will overfit faster, breaking ensemble diversity
CURRENT:
    model.classifier[-1] = nn.Linear(in_features, num_classes)
FIX — add Dropout before the final linear layer:
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
NOTE: after this change, _head_attr still returns "classifier" for mobilenet_v3_small
which is correct — no change needed there.

BUG 11 — Medium: freeze_backbone sets BN layers to eval(), but model.train() in
the training loop will recursively undo this every epoch
SEVERITY: Medium — frozen BN layers will silently resume updating running stats
during Phase 1, corrupting the pretrained statistics with small batch estimates
FIX — add this helper function to factory.py:
    def set_bn_eval(model: nn.Module) -> None:
        """Call after model.train() during frozen phase to re-lock BN layers."""
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if not m.weight.requires_grad:
                    m.eval()

Then in train.py inside train_one_epoch(), add a call at the top:
    model.train()
    set_bn_eval(model)   # re-lock frozen BN layers after model.train()

BUG 12 — Minor: get_param_groups has no guard for empty backbone_params
SEVERITY: Minor — silent failure if unfreeze_all() was not called before Phase 2
FIX — add an assertion after building the param lists:
    assert len(backbone_params) > 0, (
        f"[get_param_groups] backbone_params is empty for {backbone}. "
        "Did you call unfreeze_all() before Phase 2?"
    )

--------------------------------------------------------------------------------
FILE 5: training/train.py
--------------------------------------------------------------------------------

BUG 13 — Medium: Summary key name is misleading — reports val_macro_f1 as best_plantdoc_macro_f1
SEVERITY: Medium — early stopping monitors val_macro_f1 but the summary dict
calls it best_plantdoc_macro_f1, which will confuse anyone reading the results
CURRENT (in train_backbone return value):
    "best_plantdoc_macro_f1": early_stopper.best_score,
FIX:
    "best_val_macro_f1": early_stopper.best_score,
Also update _print_summary_table to match:
CURRENT:
    f"{s['best_plantdoc_macro_f1']:>18.4f} "
FIX:
    f"{s['best_val_macro_f1']:>18.4f} "
And update the print line at the end of train_backbone:
CURRENT:
    print(f"\n  Best PlantDoc Macro-F1: {early_stopper.best_score:.4f}")
FIX:
    print(f"\n  Best Val Macro-F1: {early_stopper.best_score:.4f}")

BUG 14 — Medium: Model not explicitly deleted in run_sequential_training before
next backbone loads, preventing GPU memory from being freed
SEVERITY: Medium — old model's VRAM allocation persists until build_model()
inside train_backbone() overwrites the variable, meaning both models briefly
coexist on GPU
FIX — in run_sequential_training, after train_backbone() returns, the model
is kept alive inside the summary dict via history. Since the model state is
already saved to disk in the checkpoint, there is no need to keep it in memory.
Add explicit cleanup inside train_backbone() right before the return statement:
    del model, optimizer, scheduler, scaler, criterion
    gc.collect()
    torch.cuda.empty_cache()
    return summary

BUG 15 — Minor: EarlyStopping.counter reset between phases is manual and brittle
SEVERITY: Minor — easy to forget if Phase 3 is ever added
FIX — add a reset() method to EarlyStopping:
    def reset(self) -> None:
        self.counter = 0
        self.should_stop = False

Then replace the manual resets in train_backbone:
CURRENT:
    early_stopper.counter = 0
    early_stopper.should_stop = False
FIX:
    early_stopper.reset()

--------------------------------------------------------------------------------
FILE 6: training/ensemble_eval.py
--------------------------------------------------------------------------------

BUG 16 — Medium: No warning when threshold is calibrated on very few classes
SEVERITY: Medium — with only 2 PlantDoc overlap classes, the Youden's J
threshold will not generalise to the other 55 classes at inference time
FIX — add this warning at the top of find_optimal_threshold(), before the
roc_curve call:
    valid_labels = labels[labels >= 0]
    unique_classes = len(np.unique(valid_labels))
    if unique_classes < 10:
        print(
            f"  ⚠️  Warning: calibrating threshold on only {unique_classes} "
            f"class(es). Threshold may not generalise to all {probs.shape[1]} classes."
        )

BUG 17 — Minor: macro_f1_after vs macro_f1_before comparison is misleading
SEVERITY: Minor — F1 after threshold is computed only on accepted samples,
so it will always look better even if the threshold is unhelpful
FIX — update _print_threshold_report to make this explicit:
CURRENT:
    print(f"  Macro-F1 after    : {r['macro_f1_after']:.4f}  (on accepted only)")
FIX:
    print(f"  Macro-F1 after    : {r['macro_f1_after']:.4f}  (accepted samples only — not comparable to before)")

================================================================================
SUMMARY TABLE
================================================================================

| # | File                    | Bug                                      | Severity |
|---|-------------------------|------------------------------------------|----------|
| 3 | metrics.py              | compute_ece inside FocalLoss class       | CRITICAL | **SOLVED**
| 4 | metrics.py              | Entire file duplicated                   | CRITICAL | **SOLVED**
| 5 | metrics.py              | Raw inverse-freq weights (440x imbalance)| CRITICAL | **SOLVED**
| 8 | main.py                 | Wrong import casing (Augmentations vs augmentations) | CRITICAL | **SOLVED**
| 6 | metrics.py              | Double softmax in FocalLoss.forward      | MEDIUM   | **SOLVED**
| 9 | main.py                 | class_counts split from class_to_idx     | MEDIUM   | **SOLVED**
|10 | factory.py              | MobileNetV3 head missing Dropout         | MEDIUM   | **SOLVED**
|11 | factory.py              | BN eval() undone by model.train()        | MEDIUM   | **SOLVED**
|13 | train.py                | Misleading best_plantdoc_macro_f1 key    | MEDIUM   | **SOLVED**
|14 | train.py                | Model not deleted before next backbone   | MEDIUM   | **SOLVED**
|16 | ensemble_eval.py        | No warning for low-class threshold calib | MEDIUM   | **SOLVED**
| 1 | augmentations.py        | GaussianBlur unconditional (p=1.0)       | MEDIUM   | **SOLVED**
| 2 | augmentations.py        | Redundant scale in RandomAffine          | MINOR    | **SOLVED**
| 7 | metrics.py              | Unused Tuple import                      | MINOR    | **SOLVED**
|12 | factory.py              | No assertion for empty backbone_params   | MINOR    | **SOLVED**
|15 | train.py                | Manual early stopping counter reset      | MINOR    | **SOLVED**
|17 | ensemble_eval.py        | Misleading F1 before/after comparison    | MINOR    | **SOLVED**

================================================================================
END OF BUG FIX INSTRUCTIONS
================================================================================