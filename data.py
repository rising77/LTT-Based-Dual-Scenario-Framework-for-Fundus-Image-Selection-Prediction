"""
src/data.py
===========

APTOS-2019 数据集加载、切分与 DataLoader 构建。

数据准备约定
------------
DATA_ROOT
├── train.csv          # 含两列：id_code, diagnosis (0..4)
└── train_images/      # 形如 {id_code}.png

按 SEED 进行 6:2:2 分层切分 → (train_df, cal_df, test_df)。
论文中 cal_df 即作为 LTT 校准集使用。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs import config as C


# ============================ 切分 ============================
def load_dataframes(data_root: Path,
                    train_ratio: float = C.TRAIN_RATIO,
                    val_ratio:   float = C.VAL_RATIO,
                    test_ratio:  float = C.TEST_RATIO,
                    seed:        int   = C.SEED):
    """读取 train.csv 后做分层 6:2:2 切分。"""
    data_root = Path(data_root)
    csv_path  = data_root / "train.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"未找到 {csv_path}. 请把 APTOS-2019 的 train.csv 与 "
            f"train_images/ 放到 {data_root}"
        )
    df = pd.read_csv(csv_path)
    df["filepath"] = df["id_code"].apply(
        lambda x: str(data_root / "train_images" / f"{x}.png"))

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 先切 train vs (val + test)
    train_df, rest_df = train_test_split(
        df, train_size=train_ratio,
        stratify=df["diagnosis"], random_state=seed)
    # 再把 (val + test) 一分为二
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        rest_df, train_size=rel_val,
        stratify=rest_df["diagnosis"], random_state=seed)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ============================ 增强与归一化 ============================
_IMNET_MEAN = (0.485, 0.456, 0.406)
_IMNET_STD  = (0.229, 0.224, 0.225)


def build_train_transforms(img_size: int = C.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=0, p=0.5),
        A.Normalize(mean=_IMNET_MEAN, std=_IMNET_STD),
        ToTensorV2(),
    ])


def build_eval_transforms(img_size: int = C.IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_IMNET_MEAN, std=_IMNET_STD),
        ToTensorV2(),
    ])


# ============================ Dataset ============================
class APTOSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None,
                 img_size: int = C.IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.tf = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["filepath"]).convert("RGB"))
        if self.tf is not None:
            img = self.tf(image=img)["image"]
        label = int(row["diagnosis"])
        return img, label


# ============================ DataLoader 工厂 ============================
def make_loaders(train_df, val_df, test_df,
                 img_size: int = C.IMG_SIZE,
                 batch_size: int = C.BATCH_SIZE,
                 num_workers: int = C.NUM_WORKERS):
    train_ds = APTOSDataset(train_df, build_train_transforms(img_size), img_size)
    val_ds   = APTOSDataset(val_df,   build_eval_transforms(img_size),  img_size)
    test_ds  = APTOSDataset(test_df,  build_eval_transforms(img_size),  img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
