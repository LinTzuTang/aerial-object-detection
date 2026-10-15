"""Deterministic mild CLAHE; RGB/BGR order must be declared by the caller.

Operate on the original image before resizing. Geometry and annotations are
unchanged. Keep the same policy at inference when evaluating a processed view.
"""
import cv2
import numpy as np


POLICY = {"name": "lab_l_clahe_blend050", "clip_limit": 2.0,
          "tile_grid_size": [8, 8], "blend": 0.5,
          "scope": "LAB L channel only, at original resolution before resize"}


def preprocess(image, *, color_order):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H x W x 3 uint8 image")
    if color_order not in ("RGB", "BGR"):
        raise ValueError("Declare RGB or BGR explicitly")
    forward = cv2.COLOR_RGB2LAB if color_order == "RGB" else cv2.COLOR_BGR2LAB
    backward = cv2.COLOR_LAB2RGB if color_order == "RGB" else cv2.COLOR_LAB2BGR
    lab = cv2.cvtColor(np.ascontiguousarray(image), forward)
    luminance = lab[:, :, 0]
    enhanced = cv2.createCLAHE(clipLimit=POLICY["clip_limit"],
        tileGridSize=tuple(POLICY["tile_grid_size"])).apply(luminance)
    lab[:, :, 0] = cv2.addWeighted(luminance, 1.0 - POLICY["blend"],
                                  enhanced, POLICY["blend"], 0)
    return np.ascontiguousarray(cv2.cvtColor(lab, backward))


class YOLOCLAHE:
    """Insert first in the training transforms, before spatial augmentation."""
    def __call__(self, labels):
        labels["img"] = preprocess(labels["img"], color_order="BGR")
        return labels


def preprocess_pil(image):
    from PIL import Image
    return Image.fromarray(preprocess(np.asarray(image.convert("RGB")), color_order="RGB"))
