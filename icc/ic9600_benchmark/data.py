from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, random_split

from .preprocess import get_eval_transforms, get_train_transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class IC9600Paths:
    train_txt: str
    test_txt: str
    img_dir: str


class IC9600Dataset(Dataset):
    """
    IC9600 format:
      Each line in txt: `image_name  score` (double-space separator).
    """

    def __init__(
        self,
        *,
        txt_path: Union[str, Path],
        img_dir: Union[str, Path],
        transform=None,
        return_filenames: bool = False,
    ):
        self.txt_path = Path(txt_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.return_filenames = return_filenames

        self._items: List[Tuple[str, float]] = []
        with self.txt_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("  ")
                if len(parts) != 2:
                    raise ValueError(
                        f"Bad IC9600 line (expected double-space separator): {line!r}"
                    )
                filename, score_s = parts[0], parts[1]
                self._items.append((filename, float(score_s)))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):
        filename, score = self._items[idx]
        path = self.img_dir / filename
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(score, dtype=torch.float32)
        if self.return_filenames:
            return image, target, filename
        return image, target


class IC9600RegressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        ic9600_train_txt: str,
        ic9600_test_txt: str,
        ic9600_img_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 512,
        val_split: float = 0.1,
        seed: int = 42,
        augment: bool = True,
    ):
        super().__init__()
        self.ic9600_train_txt = ic9600_train_txt
        self.ic9600_test_txt = ic9600_test_txt
        self.ic9600_img_dir = ic9600_img_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.img_size = int(img_size)
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.augment = bool(augment)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        train_tf = get_train_transforms(img_size=self.img_size, augment=self.augment)
        eval_tf = get_eval_transforms(img_size=self.img_size)

        full_train = IC9600Dataset(
            txt_path=self.ic9600_train_txt,
            img_dir=self.ic9600_img_dir,
            transform=train_tf,
            return_filenames=False,
        )

        val_size = int(round(len(full_train) * self.val_split))
        train_size = len(full_train) - val_size
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_train, [train_size, val_size], generator=generator
        )

        self.test_dataset = IC9600Dataset(
            txt_path=self.ic9600_test_txt,
            img_dir=self.ic9600_img_dir,
            transform=eval_tf,
            return_filenames=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
