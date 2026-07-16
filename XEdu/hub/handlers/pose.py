# -*- coding: utf-8 -*-
"""Pose task handlers."""

from typing import Any, Optional

import cv2
import numpy as np

from .base import PoseHandler
from ..BaseDT.utils import bbox_xyxy2cs, mmpose_postprocess, top_down_affine


def _mmpose_preprocess_with_bbox(img: np.ndarray, input_size=(192, 256), bbox=None):
    """Match Workflow.mmpose_preprocess, including its optional bbox argument."""
    img_shape = img.shape[:2]
    if bbox is None:
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    resized_img, scale = top_down_affine(input_size, scale, center, img)

    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    resized_img = (resized_img - mean) / std

    return resized_img, center, scale


class PoseBody17Handler(PoseHandler):
    """Human body pose handler for 17 keypoints."""

    task_name = "pose_body17"
    model_id = "pose_body17-default"

    def _preprocess(self, data: Any, bbox: Optional[list] = None) -> np.ndarray:
        raise NotImplementedError

    def _postprocess(self, outputs: list) -> tuple:
        raise NotImplementedError

    def inference(self, data: Any, bbox: Optional[list] = None, **kwargs) -> tuple:
        h, w = self.model.get_inputs()[0].shape[2:]
        model_input_size = (w, h)
        img = cv2.imread(data) if isinstance(data, str) else data
        resized_img, center, scale = _mmpose_preprocess_with_bbox(img, model_input_size, bbox)
        input_tensor = [resized_img.transpose(2, 0, 1)]
        input_name = self.model.get_inputs()[0].name
        output_names = [o.name for o in self.model.get_outputs()]
        outputs = self.model.run(output_names, {input_name: input_tensor})
        return mmpose_postprocess(outputs, model_input_size, center, scale)
