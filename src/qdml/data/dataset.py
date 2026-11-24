"""
PyTorch Dataset, PKSampler, and condition transforms for metric learning.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PKSampler(Sampler[List[int]]):
    """
    PKSampler ensures that every batch contains P distinct classes with K instances each.
    Total batch size = P * K.
    Guarantees every class appears in every batch, satisfying prototype construction (AQPM)
    and preventing positive-mining collapse (F8 in CQHN).
    """

    def __init__(self, labels: List[int] | np.ndarray | torch.Tensor, p: int = 5, k: int = 6, seed: int = 42):
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        self.labels = np.asarray(labels)

        # Build class-to-indices mapping
        self.classes = np.unique(self.labels)
        if len(self.classes) < self.p:
            raise ValueError(f"Dataset has {len(self.classes)} classes, but P={self.p} requested.")

        self.class_to_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        # Compute number of batches based on total samples
        self.n_batches = len(self.labels) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_batches):
            # Select P classes
            selected_classes = self.rng.choice(self.classes, size=self.p, replace=False)
            batch = []
            for c in selected_classes:
                indices = self.class_to_indices[c]
                # Sample K items (with replacement if fewer than K available)
                replace = len(indices) < self.k
                chosen = self.rng.choice(indices, size=self.k, replace=replace)
                batch.extend(chosen)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class SymmetricConditionTransform:
    """
    Applies symmetric condition perturbations for CQHN dual views:
    A ~ {brightness, contrast, blur, resolution, scale, mild geometry}.
    Both views (a) and (b) are drawn from the SAME symmetric distribution (Section 7).
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        blur_prob: float = 0.4,
        max_blur_radius: float = 2.0,
        res_down_range: Tuple[float, float] = (0.4, 1.0),
        scale_crop_range: Tuple[float, float] = (0.8, 1.0),
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.max_blur_radius = max_blur_radius
        self.res_down_range = res_down_range
        self.scale_crop_range = scale_crop_range

    def __call__(self, img: Image.Image) -> Image.Image:
        # 1. Random brightness
        bf = float(np.random.uniform(*self.brightness_range))
        img = ImageEnhance.Brightness(img).enhance(bf)

        # 2. Random contrast
        cf = float(np.random.uniform(*self.contrast_range))
        img = ImageEnhance.Contrast(img).enhance(cf)

        # 3. Random blur
        if np.random.rand() < self.blur_prob:
            radius = float(np.random.uniform(0.5, self.max_blur_radius))
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # 4. Resolution downsampling / upsampling
        rf = float(np.random.uniform(*self.res_down_range))
        if rf < 0.95:
            w, h = img.size
            down_w = max(16, int(w * rf))
            down_h = max(16, int(h * rf))
            img = img.resize((down_w, down_h), Image.Resampling.BILINEAR)
            img = img.resize((w, h), Image.Resampling.BILINEAR)

        # 5. Mild random crop / scale
        sf = float(np.random.uniform(*self.scale_crop_range))
        if sf < 0.98:
            w, h = img.size
            new_w, new_h = int(w * sf), int(h * sf)
            left = np.random.randint(0, w - new_w + 1)
            top = np.random.randint(0, h - new_h + 1)
            img = img.crop((left, top, left + new_w, top + new_h))

        return img


class DefectCropDataset(Dataset):
    """
    Loads defect instance crops from manifest.csv.
    Supports single view (standard / AQPM) or dual view (CQHN).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = "train",
        image_size: int = 224,
        is_training: bool = True,
        dual_views: bool = False,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.split = split
        self.image_size = image_size
        self.is_training = is_training
        self.dual_views = dual_views

        df = pd.read_csv(self.manifest_path)
        if split is not None:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        # Base normalization for ImageNet-pretrained ResNet-18
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.condition_transform = SymmetricConditionTransform()

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        full_path = self.root_dir / rel_path
        return Image.open(full_path).convert("RGB")

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img_resized = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        t = TF.to_tensor(img_resized)
        return self.normalize(t)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        row = self.df.iloc[idx]
        raw_img = self._load_image(row["path"])
        label = int(row["label"])
        crop_id = str(row["crop_id"])
        source_image = str(row["source_image"])

        if not self.dual_views:
            # Single view
            if self.is_training:
                # Mild training augmentation
                if np.random.rand() > 0.5:
                    raw_img = raw_img.transpose(Image.FLIP_LEFT_RIGHT)
            tensor_img = self._to_tensor(raw_img)
            return {
                "image": tensor_img,
                "label": label,
                "crop_id": crop_id,
                "source_image": source_image,
                "index": idx,
            }
        else:
            # CQHN dual views: two independent draws from symmetric condition distribution
            view_a = self.condition_transform(raw_img)
            view_b = self.condition_transform(raw_img)
            return {
                "image_a": self._to_tensor(view_a),
                "image_b": self._to_tensor(view_b),
                "label": label,
                "crop_id": crop_id,
                "source_image": source_image,
                "index": idx,
            }
