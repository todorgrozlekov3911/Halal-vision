"""Mask helpers: mask space (model resolution, letterboxed) -> frame space.

Only what the detection viewer needs so far. The full mask pipeline (merge the
concealed people at low resolution, crop the padding, resize once, dilate) is T10.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.contracts import MaskGeometry


def outlines_in_frame(mask: np.ndarray, geometry: MaskGeometry) -> list[np.ndarray]:
    """Outer contours of a mask-space mask, as int32 point arrays in frame pixels.

    Traced at mask resolution and scaled up, which is far cheaper than resizing
    the mask to full resolution first.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    offset = np.array([geometry.pad_x, geometry.pad_y], dtype=np.float32)
    return [np.round((c.astype(np.float32) - offset) / geometry.scale).astype(np.int32) for c in contours]
