"""
Tests for Data Preparation, Bounding Box Context Expansion, and Split Integrity (F12, F13).
"""

import pandas as pd
import pytest
from qdml.data.prepare import _leakage_check, square_pad_and_resize
import numpy as np


def test_square_pad_and_resize():
    # Rectangular crop 100x200
    crop = np.zeros((100, 200, 3), dtype=np.uint8)
    padded = square_pad_and_resize(crop, max_size=384)
    # Long side was 200 <= 384, so output should be 200x200 square
    assert padded.shape == (200, 200, 3)

    # Large crop 500x300
    large_crop = np.zeros((500, 300, 3), dtype=np.uint8)
    padded_large = square_pad_and_resize(large_crop, max_size=384)
    # Capped at 384x384
    assert padded_large.shape == (384, 384, 3)


def test_leakage_check_flags_split_overlap():
    """Finding F13 verification: detects when same source frame appears in two splits."""
    clean_df = pd.DataFrame({
        "source_image": ["img1.jpg", "img1.jpg", "img2.jpg"],
        "split": ["train", "train", "val"],
    })
    chk_clean = _leakage_check(clean_df)
    assert chk_clean["is_leak_free"] is True
    assert chk_clean["source_images_in_multiple_splits"] == 0

    leaking_df = pd.DataFrame({
        "source_image": ["img1.jpg", "img1.jpg", "img2.jpg"],
        "split": ["train", "val", "test"],  # img1 spans train and val!
    })
    chk_leaking = _leakage_check(leaking_df)
    assert chk_leaking["is_leak_free"] is False
    assert chk_leaking["source_images_in_multiple_splits"] == 1
