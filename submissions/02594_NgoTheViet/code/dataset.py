"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện giữ nguyên để notebook, train.py và eval.py ghép được với nhau:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).
    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    """
    labels_dir = Path(labels_dir)
    train_path = labels_dir / f"train_subset{fold}.csv"
    val_path = labels_dir / f"val_subset{fold}.csv"
    test_path = labels_dir / f"test_subset{fold}.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.
      1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. Giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
      3. Hợp ba tập phải bằng đúng 17.509 ảnh
      4. Mọi Filename đều tồn tại trong `images_dir`
    """
    images_dir = Path(images_dir)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    # 1. Kiểm tra tỉ lệ
    train_pct = (n_train / total_imgs) * 100
    val_pct = (n_val / total_imgs) * 100
    test_pct = (n_test / total_imgs) * 100

    # Phân bố theo lớp
    per_class_train = train_df["Label"].value_counts().sort_index().to_dict()
    per_class_val = val_df["Label"].value_counts().sort_index().to_dict()
    per_class_test = test_df["Label"].value_counts().sort_index().to_dict()

    # 2. Giao rỗng
    s_train = set(train_df["Filename"])
    s_val = set(val_df["Filename"])
    s_test = set(test_df["Filename"])

    inter_train_val = len(s_train & s_val)
    inter_train_test = len(s_train & s_test)
    inter_val_test = len(s_val & s_test)

    assert inter_train_val == 0, f"LỖI: Giao train và val không rỗng ({inter_train_val} ảnh trùng)!"
    assert inter_train_test == 0, f"LỖI: Giao train và test không rỗng ({inter_train_test} ảnh trùng)!"
    assert inter_val_test == 0, f"LỖI: Giao val và test không rỗng ({inter_val_test} ảnh trùng)!"

    # 3. Hợp đủ 17.509
    union_set = s_train | s_val | s_test
    assert len(union_set) == 17509, f"LỖI: Hợp 3 tập không đủ 17.509 ảnh (thực tế: {len(union_set)})!"

    # 4. Kiểm tra file ảnh tồn tại
    missing_train = [f for f in train_df["Filename"] if not (images_dir / f).exists()]
    missing_val = [f for f in val_df["Filename"] if not (images_dir / f).exists()]
    missing_test = [f for f in test_df["Filename"] if not (images_dir / f).exists()]

    assert len(missing_train) == 0, f"LỖI: Thiếu {len(missing_train)} ảnh train trong {images_dir}!"
    assert len(missing_val) == 0, f"LỖI: Thiếu {len(missing_val)} ảnh val trong {images_dir}!"
    assert len(missing_test) == 0, f"LỖI: Thiếu {len(missing_test)} ảnh test trong {images_dir}!"

    info = {
        "n": {
            "train": n_train, "val": n_val, "test": n_test, "total": total_imgs,
            "percentages": {"train": train_pct, "val": val_pct, "test": test_pct}
        },
        "per_class": {
            "train": per_class_train, "val": per_class_val, "test": per_class_test
        },
        "overlap": {
            "train_val": inter_train_val,
            "train_test": inter_train_test,
            "val_test": inter_val_test,
            "union": len(union_set)
        },
        "missing_files": 0
    }
    return info


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform theo `train`, `img_size` và mức `aug` ('basic', 'color', 'trivial', 'randaug').
    Val/test: CenterCrop(img_size) và chuẩn hoá theo ImageNet. KHÔNG augmentation ngẫu nhiên khi đánh giá.
    """
    if train:
        t_list = []
        t_list.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if aug == "basic":
            pass
        elif aug == "color":
            t_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        elif aug == "trivial":
            t_list.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            t_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug == "flip_v":
            t_list.append(transforms.RandomVerticalFlip(p=0.5))
        else:
            raise ValueError(f"Không nhận diện aug={aug}. Chọn: basic, color, trivial, randaug, flip_v")

        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transforms.Compose(t_list)
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).
    __getitem__(i) trả về (image_tensor, label: int, filename: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str]:
        filename = self.filenames[i]
        label = int(self.labels[i])
        img_path = self.images_dir / filename

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)

        return img_tensor, label, filename


def _seed_worker(worker_id: int):
    """Worker init fn để đảm bảo tính tái lập (reproducibility) khi dùng nhiều worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader.
    - train=True: shuffle=True (hoặc dùng WeightedRandomSampler nếu sampler='balanced').
    - train=False: shuffle=False, giữ nguyên thứ tự df để map với tên file khi eval.
    """
    dataset = DeepWeedsDataset(df, images_dir=images_dir, transform=transform)

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts()
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights[lbl] for lbl in df["Label"]]
            sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
            data_sampler = WeightedRandomSampler(
                weights=sample_weights_tensor,
                num_samples=len(sample_weights_tensor),
                replacement=True
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=data_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=_seed_worker
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=True,
                worker_init_fn=_seed_worker
            )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )

    return loader
