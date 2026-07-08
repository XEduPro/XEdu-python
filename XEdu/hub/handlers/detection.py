# -*- coding: utf-8 -*-
"""
检测任务 Handler 实现
"""

import cv2
import numpy as np
from typing import List, Any, Optional

from .base import DetectionHandler


class DetBodyHandler(DetectionHandler):
    """人体检测"""

    task_name = "det_body"
    model_id = "det_body-default"

    def _preprocess(self, data: Any) -> np.ndarray:
        """图像预处理"""
        if isinstance(data, str):
            img = cv2.imread(data)
        else:
            img = data

        if img is None:
            raise ValueError("Failed to load image")

        # 缩放到模型输入大小（示例：640x640）
        h, w = img.shape[:2]
        size = 640
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        # 填充到 640x640
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = img

        # 归一化
        padded = padded.astype(np.float32) / 255.0
        return np.expand_dims(padded.transpose(2, 0, 1), 0)

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """后处理：提取 boxes, scores, class_ids"""
        output = outputs[0]
        boxes, scores, class_ids = [], [], []

        for det in output:
            x1, y1, x2, y2, score = det[:5]
            if score > 0.5:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(0)  # 人体检测只有 1 类

        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            np.array(class_ids) if class_ids else np.zeros(0, dtype=int),
        )


class DetCocoHandler(DetectionHandler):
    """COCO 目标检测"""

    task_name = "det_coco"
    model_id = "det_coco-default"

    def _preprocess(self, data: Any) -> np.ndarray:
        """图像预处理"""
        if isinstance(data, str):
            img = cv2.imread(data)
        else:
            img = data

        if img is None:
            raise ValueError("Failed to load image")

        h, w = img.shape[:2]
        size = 640
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = img

        padded = padded.astype(np.float32) / 255.0
        return np.expand_dims(padded.transpose(2, 0, 1), 0)

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """后处理：提取 boxes, scores, class_ids"""
        output = outputs[0]
        boxes, scores, class_ids = [], [], []

        for det in output:
            x1, y1, x2, y2, score, cls_id = det[:6]
            if score > 0.5:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(int(cls_id))

        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            np.array(class_ids) if class_ids else np.zeros(0, dtype=int),
        )


class DetFaceHandler(DetectionHandler):
    """人脸检测（YuNet）"""

    task_name = "det_face"
    model_id = "det_face-yunet"

    def __init__(self, checkpoint: Optional[str] = None, **kwargs):
        """初始化人脸检测"""
        # 调用父类初始化
        super().__init__(checkpoint, **kwargs)
        # YuNet 特殊处理
        self.score_threshold = kwargs.get("score_threshold", 0.6)
        self.nms_threshold = kwargs.get("nms_threshold", 0.3)
        self.top_k = kwargs.get("top_k", 5000)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载 YuNet 模型"""
        self.model = cv2.FaceDetectorYN_create(
            checkpoint_path, "", (320, 320), self.score_threshold, self.nms_threshold, self.top_k
        )

    def _preprocess(self, data: Any) -> np.ndarray:
        """YuNet 不需要常规预处理"""
        if isinstance(data, str):
            img = cv2.imread(data)
        else:
            img = data
        if img is None:
            raise ValueError("Failed to load image")
        return img

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """YuNet 输出处理"""
        # YuNet 的 detect() 返回 (_, detections)
        # 这里简化处理，实际在 inference 中会不同
        return outputs[0] if outputs else np.zeros((0, 15))

    def inference(self, data: Any, **kwargs) -> tuple:
        """YuNet 特殊推理逻辑"""
        img = self._preprocess(data)
        h, w = img.shape[:2]

        # 设置输入大小
        self.model.setInputSize((w, h))
        self.model.setScoreThreshold(kwargs.get("score_threshold", self.score_threshold))
        self.model.setNMSThreshold(kwargs.get("nms_threshold", self.nms_threshold))
        self.model.setTopK(kwargs.get("top_k", self.top_k))

        # 检测
        _, faces = self.model.detect(img)

        boxes, scores = [], []
        if faces is not None:
            for det in faces:
                x, y, bw, bh = det[:4]
                score = float(det[14]) if len(det) > 14 else 0.0
                boxes.append([x, y, x + bw, y + bh])
                scores.append(score)

        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            np.zeros(0, dtype=int),  # 人脸检测无 class_id
        )
