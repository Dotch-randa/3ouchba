"""
utils/dataset.py — Unified PlantDisease Dataset
Reads from stratified CSV splits, applies class_mapping.json to unify labels
across PlantVillage, Plant Disease Expert, and PlantDoc.
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PlantDiseaseDataset(Dataset):
    """
    Unified dataset for plant disease classification.

    CSV format expected:
        image_path, label, source_dataset

    class_mapping.json format:
        { "raw_label": "unified_label", ... }

    Args:
        csv_path:       Path to the stratified split CSV.
        class_to_idx:   Dict mapping unified class name → integer index.
                        Build once from training set; reuse for val/test.
        class_mapping:  Dict loaded from class_mapping.json.
        transform:      torchvision transform pipeline.
        return_meta:    If True, also return (original_label, source_dataset).
        image_root:     Optional root prefix to prepend to relative image paths.
    """

    def __init__(
        self,
        csv_path: str,
        class_to_idx: Dict[str, int],
        class_mapping: Dict[str, str],
        transform: Optional[Callable] = None,
        return_meta: bool = False,
        image_root: Optional[str] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.class_mapping = class_mapping
        self.transform = transform
        self.return_meta = return_meta
        self.image_root = Path(image_root) if image_root else None

        # Validate required columns
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        # Remap labels and drop rows whose unified label isn't in class_to_idx
        self.df = self.df.copy()
        self.df["unified_label"] = self.df["label"].map(
            lambda l: self.class_mapping.get(l, l)
        )
        before = len(self.df)
        self.df = self.df[self.df["unified_label"].isin(self.class_to_idx)].reset_index(
            drop=True
        )
        dropped = before - len(self.df)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with unmapped labels.")

        if "source_dataset" not in self.df.columns:
            self.df["source_dataset"] = "unknown"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        
        if self.image_root and not img_path.is_absolute():
            img_path = self.image_root / img_path
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
          
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[row["unified_label"]]

        if self.return_meta:
            return image, label_idx, row["label"], row["source_dataset"]
        return image, label_idx

    # ------------------------------------------------------------------
    # Class utilities
    # ------------------------------------------------------------------

    @staticmethod
    def build_class_to_idx(
        csv_paths: List[str],
        class_mapping: Dict[str, str],
        min_samples: int = 100,
    ) -> Dict[str, int]:
        """
        Build a class_to_idx mapping from one or more CSVs.
        Drops any unified class with fewer than min_samples images.
        Call this once on training CSVs, then reuse for val/test.
        """
        frames = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(frames, ignore_index=True)
        df["unified_label"] = df["label"].map(lambda l: class_mapping.get(l, l))
        counts = df["unified_label"].value_counts()
        valid = sorted(counts[counts >= min_samples].index.tolist())
        class_to_idx = {cls: i for i, cls in enumerate(valid)}
        print(
            f"[Dataset] {len(class_to_idx)} classes retained "
            f"(dropped {len(counts) - len(class_to_idx)} with <{min_samples} samples)."
        )
        return class_to_idx

    @staticmethod
    def load_class_mapping(path: str) -> Dict[str, str]:
        with open(path, "r") as f:
            return json.load(f)