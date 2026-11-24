"""
SubstationSim: Synthetic inspection corpus generator for pipeline testing (Finding D5).
Renders 5 defect categories with cluttered background, lighting variations, and box annotations.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np


class SubstationSim:
    """
    Renders synthetic structural surrogate images mimicking substation defects.
    """

    DEFECT_NAMES = [
        "silica_gel_discoloration",
        "blurred_meter_reading",
        "damaged_cover_plate",
        "damaged_meter_housing",
        "fractured_silica_gel_body",
    ]

    def __init__(self, out_dir: str | Path, seed: int = 42):
        self.out_dir = Path(out_dir)
        self.rng = np.random.default_rng(seed)

    def generate_frame(self, frame_idx: int) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        # Random resolution from 400x300 to 1200x900
        w = int(self.rng.integers(400, 1200))
        h = int(self.rng.integers(300, 900))

        # Base background: textured gray/metallic with gradient
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        base_color = self.rng.integers(90, 160, size=3)
        canvas[:] = base_color

        # Add noise and lighting gradient
        gradient = np.tile(np.linspace(0.7, 1.3, w), (h, 1))
        for c in range(3):
            canvas[:, :, c] = np.clip(canvas[:, :, c] * gradient, 0, 255).astype(np.uint8)

        # Draw structural lines (substation girders / pipes)
        n_lines = self.rng.integers(3, 8)
        for _ in range(n_lines):
            pt1 = (int(self.rng.integers(0, w)), int(self.rng.integers(0, h)))
            pt2 = (int(self.rng.integers(0, w)), int(self.rng.integers(0, h)))
            thickness = int(self.rng.integers(2, 6))
            cv2.line(canvas, pt1, pt2, (60, 60, 60), thickness)

        # Place 1 to 3 defects
        n_defects = self.rng.integers(1, 4)
        boxes = []

        for _ in range(n_defects):
            class_id = int(self.rng.integers(0, len(self.DEFECT_NAMES)))
            # Defect scale: small box (30 to 120 px)
            bw = int(self.rng.integers(35, min(120, w // 3)))
            bh = int(self.rng.integers(35, min(120, h // 3)))

            bx = int(self.rng.integers(10, w - bw - 10))
            by = int(self.rng.integers(10, h - bh - 10))

            # Render defect pattern according to class
            if class_id == 0:  # silica_gel_discoloration (pink/purple gradient)
                canvas[by:by + bh, bx:bx + bw] = (180, 105, 255)
            elif class_id == 1:  # blurred_meter_reading (blurred concentric dial)
                cv2.circle(canvas, (bx + bw // 2, by + bh // 2), min(bw, bh) // 3, (220, 220, 220), -1)
                cv2.circle(canvas, (bx + bw // 2, by + bh // 2), min(bw, bh) // 4, (40, 40, 40), 2)
            elif class_id == 2:  # damaged_cover_plate (irregular dark jagged shape)
                pts = np.array([[bx, by], [bx + bw, by + 10], [bx + bw - 10, by + bh], [bx + 5, by + bh - 5]])
                cv2.fillPoly(canvas, [pts], (30, 30, 30))
            elif class_id == 3:  # damaged_meter_housing (crack lines on housing)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (120, 130, 140), -1)
                cv2.line(canvas, (bx, by), (bx + bw, by + bh), (20, 20, 20), 2)
            else:  # fractured_silica_gel_body (shattered glass pattern)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 240, 255), 2)
                cv2.line(canvas, (bx + bw // 3, by), (bx + 2 * bw // 3, by + bh), (240, 255, 255), 2)

            boxes.append({
                "class_id": class_id,
                "class_name": self.DEFECT_NAMES[class_id],
                "xmin": bx,
                "ymin": by,
                "xmax": bx + bw,
                "ymax": by + bh,
            })

        return canvas, boxes
