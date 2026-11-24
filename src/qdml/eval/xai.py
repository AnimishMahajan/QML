"""
Explainability (XAI): Occlusion-based pairwise similarity attribution (Section 9 & 18).
Computes attribution maps for both the correct top match and highest incorrect match.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from ..models.backbone import QDMLNet
from ..quantum.statevector import state_fidelity


def compute_pair_occlusion_map(
    model: QDMLNet,
    query_tensor: torch.Tensor,
    gallery_tensor: torch.Tensor,
    patch_size: int = 16,
    stride: int = 8,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Computes pair occlusion attribution:
      Delta S_r = S(q, g) - S(q \\ r, g)
    query_tensor: (3, H, W)
    gallery_tensor: (3, H, W)
    Returns: attribution heatmap of shape (H, W).
    """
    dev = device or query_tensor.device
    model.eval()

    with torch.no_grad():
        # Baseline similarity without occlusion
        out_q = model(query_tensor.unsqueeze(0).to(dev))
        out_g = model(gallery_tensor.unsqueeze(0).to(dev))

        if out_q["psi"] is not None and out_g["psi"] is not None:
            base_sim = state_fidelity(out_q["psi"], out_g["psi"]).item()
        else:
            base_sim = float(torch.sum(out_q["x"] * out_g["x"]).item())

        _, H, W = query_tensor.shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        # Slide occlusion patch over query
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                occluded_q = query_tensor.clone()
                # Zero out patch (black patch or mean replacement)
                occluded_q[:, y:y + patch_size, x:x + patch_size] = 0.0

                out_occ = model(occluded_q.unsqueeze(0).to(dev))
                if out_occ["psi"] is not None and out_g["psi"] is not None:
                    occ_sim = state_fidelity(out_occ["psi"], out_g["psi"]).item()
                else:
                    occ_sim = float(torch.sum(out_occ["x"] * out_g["x"]).item())

                # Positive delta means patch contributed positively to similarity
                delta_s = base_sim - occ_sim
                heatmap[y:y + patch_size, x:x + patch_size] += delta_s
                count_map[y:y + patch_size, x:x + patch_size] += 1.0

        # Average overlaps
        count_map = np.maximum(count_map, 1.0)
        heatmap = heatmap / count_map

        # Normalize to [0, 1]
        h_min, h_max = np.min(heatmap), np.max(heatmap)
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)

        return heatmap
