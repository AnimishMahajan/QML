"""
Dataset preparation: converts Pascal VOC object detection annotations into
instance-level defect crops with condition profiling and leak-free 7:2:1 splitting.
"""

from __future__ import annotations
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Canonical 5 classes mapped from Chinese pinyin tags
CLASS_MAPPING = {
    "hxq_gjbs": "silica_gel_discoloration",
    "bj_bpmh": "blurred_meter_reading",
    "bjdsyc": "blurred_meter_reading",
    "gbps": "damaged_cover_plate",
    "bj_wkps": "damaged_meter_housing",
    "hxq_gjtps": "fractured_silica_gel_body",
}

CANONICAL_CLASSES = [
    "silica_gel_discoloration",
    "blurred_meter_reading",
    "damaged_cover_plate",
    "damaged_meter_housing",
    "fractured_silica_gel_body",
]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}


def compute_crop_attributes(crop_img: np.ndarray, frame_shape: Tuple[int, int], box_area: float) -> Dict[str, float]:
    """
    Computes brightness, contrast, sharpness, and scale fraction for a crop.
    crop_img: BGR uint8 image
    """
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    
    # Laplacian variance as reference-free blur/sharpness proxy
    lap = cv2.Laplacian((gray * 255.0).astype(np.uint8), cv2.CV_64F)
    sharpness = float(np.var(lap))
    
    frame_h, frame_w = frame_shape
    frame_area = frame_w * frame_h
    scale_frac = float(box_area / max(1.0, frame_area))
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "scale_frac": scale_frac,
    }


def square_pad_and_resize(crop: np.ndarray, max_size: int = 384) -> np.ndarray:
    """
    Pads image to square with black borders and resizes so long side <= max_size.
    """
    h, w = crop.shape[:2]
    max_side = max(h, w)
    
    # Create black square canvas
    square = np.zeros((max_side, max_side, 3), dtype=crop.dtype)
    y_off = (max_side - h) // 2
    x_off = (max_side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = crop
    
    # Resize if larger than max_size
    if max_side > max_size:
        square = cv2.resize(square, (max_size, max_size), interpolation=cv2.INTER_AREA)
    return square


def prepare_dataset(
    raw_root: str | Path,
    out_dir: str | Path,
    context_ratio: float = 0.15,
    min_box: int = 24,
    save_size: int = 384,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Parses Pascal VOC annotations in raw_root and converts to defect crops.
    """
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    
    ann_dir = raw_root / "Annotations"
    img_dir = raw_root / "JPEGImages"
    
    if not ann_dir.exists() or not img_dir.exists():
        # Check subdirectories
        sub_ann = list(raw_root.rglob("Annotations"))
        if sub_ann:
            ann_dir = sub_ann[0]
            img_dir = ann_dir.parent / "JPEGImages"
            
    xml_files = sorted(list(ann_dir.glob("*.xml")))
    print(f"Found {len(xml_files)} annotation XML files.")
    
    parsed_instances = []
    skipped_counts = {"tiny_box": 0, "unknown_class": 0, "missing_image": 0}
    source_frame_classes: Dict[str, List[str]] = {}
    
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue
            
        filename = root.findtext("filename")
        if not filename:
            filename = xml_path.stem + ".jpg"
            
        img_path = img_dir / filename
        if not img_path.exists():
            # Try matching stem
            candidates = list(img_dir.glob(f"{xml_path.stem}.*"))
            if candidates:
                img_path = candidates[0]
            else:
                skipped_counts["missing_image"] += 1
                continue
                
        size_elem = root.find("size")
        if size_elem is not None:
            src_w = int(size_elem.findtext("width", "0"))
            src_h = int(size_elem.findtext("height", "0"))
        else:
            src_w, src_h = 0, 0
            
        frame_key = str(img_path.name)
        if frame_key not in source_frame_classes:
            source_frame_classes[frame_key] = []
            
        box_idx = 0
        for obj in root.findall("object"):
            raw_name = obj.findtext("name", "").strip()
            if raw_name not in CLASS_MAPPING:
                skipped_counts["unknown_class"] += 1
                continue
                
            canonical_name = CLASS_MAPPING[raw_name]
            source_frame_classes[frame_key].append(canonical_name)
            
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
                
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            
            bw = xmax - xmin
            bh = ymax - ymin
            if max(bw, bh) < min_box:
                skipped_counts["tiny_box"] += 1
                continue
                
            parsed_instances.append({
                "xml_path": str(xml_path),
                "img_path": str(img_path),
                "source_image": frame_key,
                "box_idx": box_idx,
                "raw_name": raw_name,
                "class_name": canonical_name,
                "label": CLASS_TO_IDX[canonical_name],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "src_w": src_w,
                "src_h": src_h,
            })
            box_idx += 1

    print(f"Extracted {len(parsed_instances)} valid defect boxes across {len(source_frame_classes)} frames.")
    
    # 7:2:1 split grouped by source frame
    rng = np.random.default_rng(seed)
    unique_frames = sorted(list(source_frame_classes.keys()))
    rng.shuffle(unique_frames)
    
    n_total = len(unique_frames)
    n_train = int(0.70 * n_total)
    n_val = int(0.20 * n_total)
    
    frame_to_split = {}
    for i, frame in enumerate(unique_frames):
        if i < n_train:
            frame_to_split[frame] = "train"
        elif i < n_train + n_val:
            frame_to_split[frame] = "val"
        else:
            frame_to_split[frame] = "test"
            
    manifest_rows = []
    
    for inst in parsed_instances:
        frame_key = inst["source_image"]
        split = frame_to_split[frame_key]
        
        # Load image lazily
        img = cv2.imread(inst["img_path"])
        if img is None:
            continue
        fh, fw = img.shape[:2]
        inst["src_w"] = fw
        inst["src_h"] = fh
        
        # Expand bounding box by context_ratio
        bw = inst["xmax"] - inst["xmin"]
        bh = inst["ymax"] - inst["ymin"]
        x1 = max(0, int(round(inst["xmin"] - bw * context_ratio)))
        y1 = max(0, int(round(inst["ymin"] - bh * context_ratio)))
        x2 = min(fw, int(round(inst["xmax"] + bw * context_ratio)))
        y2 = min(fh, int(round(inst["ymax"] + bh * context_ratio)))
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        attr = compute_crop_attributes(crop, (fh, fw), bw * bh)
        processed_crop = square_pad_and_resize(crop, max_size=save_size)
        
        # Stable unique crop ID
        raw_id_str = f"{frame_key}_{inst['box_idx']}_{inst['class_name']}"
        crop_id = hashlib.sha256(raw_id_str.encode()).hexdigest()[:12]
        
        rel_crop_path = Path("crops") / split / inst["class_name"] / f"{crop_id}.jpg"
        abs_crop_path = out_dir / rel_crop_path
        abs_crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(abs_crop_path), processed_crop)
        
        manifest_rows.append({
            "crop_id": crop_id,
            "path": str(rel_crop_path).replace("\\", "/"),
            "label": inst["label"],
            "class_name": inst["class_name"],
            "split": split,
            "source_image": frame_key,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "src_w": fw, "src_h": fh, "src_pixels": fw * fh,
            "crop_w": processed_crop.shape[1],
            "crop_h": processed_crop.shape[0],
            "scale_frac": attr["scale_frac"],
            "brightness": attr["brightness"],
            "contrast": attr["contrast"],
            "sharpness": attr["sharpness"],
        })
        
    df = pd.DataFrame(manifest_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False)
    
    # Metadata and integrity check
    leakage_check = _leakage_check(df)
    meta = {
        "class_names": CANONICAL_CLASSES,
        "n_crops": len(df),
        "n_source_frames": len(unique_frames),
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": df["class_name"].value_counts().to_dict(),
        "skipped": skipped_counts,
        "leakage_check": leakage_check,
    }
    with open(out_dir / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Dataset preparation complete. Saved {len(df)} crops to {out_dir}.")
    print(f"Split leakage check: {leakage_check}")
    return df


def _leakage_check(df: pd.DataFrame) -> Dict[str, int]:
    """Asserts that no source image appears in multiple splits."""
    source_splits = df.groupby("source_image")["split"].nunique()
    leaking_frames = int((source_splits > 1).sum())
    return {
        "source_images_in_multiple_splits": leaking_frames,
        "is_leak_free": leaking_frames == 0,
    }
