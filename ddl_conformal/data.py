"""
data.py

Localization dataloaders over precomputed log-mel features (from ddl_dataloader.py).

Splits (annotation CSVs in <data_root>/annotations/), keeping rows with is_corrupt == False and
sample_9600 == True:
  train            train_dataset_with_nondrone_with_9600_flag.csv                      40,692
  val evaluation   val_dataset_...csv, conformal_role == "evaluation"  (early stopping,  2,904
                   HP selection)
  val calibration  val_dataset_...csv, conformal_role == "calibration" (conformal q-hat) 2,912
  test             test_dataset_with_nondrone_with_9600_flag.csv                        11,628

Features are read from <data_root>/<condition>/features/logmel/{ID:08d}.pt, so the same CSVs
serve every test condition (plain, awgn_*, clutter_*): only the feature folder changes.

Targets: bearing_vec = (cos, sin) of "Bearing to Target" (degrees), range = "Range to Target" / 250 m.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import MAX_RANGE, annotations_dir, logmel_dir

TRAIN_CSV = "train_dataset_with_nondrone_with_9600_flag.csv"
VAL_CSV = "val_dataset_with_nondrone_with_9600_flag.csv"
TEST_CSV = "test_dataset_with_nondrone_with_9600_flag.csv"


def build_feature_index(feature_dir: Path) -> dict:
    """{file_id: path} for every *.pt in feature_dir ('00062089.pt' or '00062089_logmel.pt')."""
    index = {}
    if not feature_dir.exists():
        return index
    for path in feature_dir.glob("*.pt"):
        index[path.stem.split("_")[0]] = path
    return index


def load_filtered_dataframe(csv_path, conformal_role: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = (df["is_corrupt"] == False) & (df["sample_9600"] == True)  # noqa: E712
    if conformal_role:
        keep &= df["conformal_role"] == conformal_role
    return df[keep].reset_index(drop=True)


class LocalizationDataset(Dataset):
    """(logmel, bearing_vec, range) per recording; with_id=True also returns the recording ID."""

    def __init__(self, data_frame: pd.DataFrame, feature_dir: Path, with_id: bool = False,
                 max_range: float = MAX_RANGE, split: str = ""):
        self.data_frame = data_frame
        self.with_id = with_id
        self.max_range = max_range
        self.feature_index = build_feature_index(Path(feature_dir))
        print(f"[data] {split}: {feature_dir} ({len(self.feature_index)} feature files, {len(data_frame)} rows)")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_id = f"{self.data_frame.iloc[idx, 0]:08d}"
        if file_id not in self.feature_index:
            raise FileNotFoundError(f"Feature for {file_id} not found.")
        audio_data = torch.load(self.feature_index[file_id], weights_only=True).float()

        radians = np.deg2rad(float(self.data_frame.iloc[idx, 3]))
        bearing_vec = torch.tensor([np.cos(radians), np.sin(radians)], dtype=torch.float32)
        range_to_target = torch.tensor(float(self.data_frame.iloc[idx, 4]) / self.max_range, dtype=torch.float32)

        if self.with_id:
            return audio_data, bearing_vec, range_to_target, torch.tensor(int(file_id))
        return audio_data, bearing_vec, range_to_target


def get_localization_loaders(data_root, condition: str = "plain", batch_size: int = 64,
                             num_workers: int = 4, pin_memory: bool = True,
                             prefetch_factor: Optional[int] = 2):
    """(train, val_evaluation, val_calibration, test) loaders reading features of `condition`.

    Only the clean condition has train/val features; noisy conditions are test-only.
    """
    ann = annotations_dir(data_root)
    feat = logmel_dir(data_root, condition)
    train_df = load_filtered_dataframe(ann / TRAIN_CSV)
    val_evaluation_df = load_filtered_dataframe(ann / VAL_CSV, conformal_role="evaluation")
    val_calibration_df = load_filtered_dataframe(ann / VAL_CSV, conformal_role="calibration")
    test_df = load_filtered_dataframe(ann / TEST_CSV)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                     persistent_workers=num_workers > 0)
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    return (
        DataLoader(LocalizationDataset(train_df, feat, split="train"), shuffle=True, **dl_kwargs),
        DataLoader(LocalizationDataset(val_evaluation_df, feat, True, split="val evaluation"), shuffle=False, **dl_kwargs),
        DataLoader(LocalizationDataset(val_calibration_df, feat, True, split="val calibration"), shuffle=False, **dl_kwargs),
        DataLoader(LocalizationDataset(test_df, feat, True, split=f"test ({condition})"), shuffle=False, **dl_kwargs),
    )
