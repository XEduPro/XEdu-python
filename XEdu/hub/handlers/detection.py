# -*- coding: utf-8 -*-
"""
检测任务 Handler 实现

重要：这里的前处理/后处理数值逻辑必须与 XEdu/hub/workflow.py 中
Workflow._det_infer() 的实现逐位对齐（224x224 输入、ImageNet 归一化、
model.run([], {'input': ...})、coco_class 按 idx+1 查表、严格大于阈值）。
早期版本的这个文件用的是编造的 640x640/无归一化实现，数值上和真实模型
不匹配，产出的检测结果是错的。为避免两份逻辑再次分叉，det_body/det_coco
的核心计算复用同一个模块级纯函数 _run_legacy_detection()。
"""

import copy
import math
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from .base import DetectionHandler
from ..datatset_class import coco_class

_MEAN_VEC = np.array([0.485, 0.456, 0.406])
_STDDEV_VEC = np.array([0.229, 0.224, 0.225])
FACE_DET_LEGACY_PARAMS = {"scaleFactor", "minNeighbors", "minSize", "maxSize"}


def _legacy_preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Bit-for-bit port of the closure in Workflow._det_infer."""
    resized = cv2.resize(image_bgr, (224, 224))
    chw = np.array(resized).transpose(2, 0, 1)

    img_data = chw.astype("float32")
    norm_img_data = np.zeros(img_data.shape).astype("float32")
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - _MEAN_VEC[i]) / _STDDEV_VEC[i]

    return norm_img_data.reshape(1, 3, 224, 224).astype("float32")


def _run_legacy_detection(
    model: Any,
    data: Union[str, np.ndarray],
    threshold: float = 0.5,
    target_class: Optional[Union[str, List[str]]] = None,
):
    """Runs the same detection path as Workflow._det_infer and returns
    (boxes, scores, classes) as plain Python lists, with classes being
    the coco_class string labels (matches upstream behavior: det_body
    also populates self.classes internally even though it isn't surfaced
    in format_output).
    """
    if isinstance(data, str):
        image = cv2.imread(data)
        if image is None:
            raise ValueError(f"Failed to read image from path: {data}")
    else:
        image = copy.copy(data)

    input_data = _legacy_preprocess(image)
    raw_result = model.run([], {"input": input_data})

    h_ratio = image.shape[0] / 224
    w_ratio = image.shape[1] / 224

    boxes, scores, classes = [], [], []
    for idx, (a, b, c, d, e) in enumerate(raw_result[0][0]):
        class_id = raw_result[1][0][idx]
        label = coco_class[class_id + 1]

        if target_class is not None:
            if isinstance(target_class, str):
                if label != target_class:
                    continue
            elif isinstance(target_class, list):
                if label not in target_class:
                    continue

        if e > threshold:
            boxes.append([a * w_ratio, b * h_ratio, c * w_ratio, d * h_ratio])
            scores.append(e)
            classes.append(label)

    return boxes, scores, classes


class DetBodyHandler(DetectionHandler):
    """人体检测（复用 det_coco 的底层模型结构，仅 1 类）"""

    task_name = "det_body"
    model_id = "det_body-default"

    def _preprocess(self, data: Any) -> np.ndarray:
        # 数值计算集中在 _run_legacy_detection 里，这里不需要单独实现。
        raise NotImplementedError

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        raise NotImplementedError

    def inference(self, data: Any, threshold: float = 0.5, target_class=None, **kwargs) -> tuple:
        boxes, scores, classes = _run_legacy_detection(self.model, data, threshold, target_class)
        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            np.array(classes, dtype=object) if classes else np.zeros(0, dtype=object),
        )

    def format_output(self, raw_output: tuple, lang: str = "en") -> dict:
        boxes, scores, _classes = raw_output
        return {
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        }


class DetCocoHandler(DetectionHandler):
    """COCO 目标检测（80 类）"""

    task_name = "det_coco"
    model_id = "det_coco-default"

    def _preprocess(self, data: Any) -> np.ndarray:
        raise NotImplementedError

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        raise NotImplementedError

    def inference(self, data: Any, threshold: float = 0.5, target_class=None, **kwargs) -> tuple:
        boxes, scores, classes = _run_legacy_detection(self.model, data, threshold, target_class)
        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            np.array(classes, dtype=object) if classes else np.zeros(0, dtype=object),
        )

    def format_output(self, raw_output: tuple, lang: str = "en") -> dict:
        boxes, scores, classes = raw_output
        return {
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "classes": classes.tolist() if isinstance(classes, np.ndarray) else classes,
        }


class DetFaceHandler(DetectionHandler):
    """人脸检测（YuNet）。逐位对齐 Workflow._face_det_infer 的 YuNet 分支。"""

    task_name = "det_face"
    model_id = "det_face-yunet"

    def __init__(self, checkpoint: Optional[str] = None, **kwargs):
        self._score_threshold = kwargs.get("score_threshold", 0.6)
        self._nms_threshold = kwargs.get("nms_threshold", 0.3)
        self._top_k = kwargs.get("top_k", 5000)
        super().__init__(checkpoint, **kwargs)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if checkpoint_path.lower().endswith(".xml"):
            self.model = cv2.CascadeClassifier(checkpoint_path)
        else:
            self.model = cv2.FaceDetectorYN_create(
                checkpoint_path, "", (320, 320), self._score_threshold, self._nms_threshold, self._top_k
            )

    def _preprocess(self, data: Any) -> np.ndarray:
        raise NotImplementedError

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        raise NotImplementedError

    def inference(self, data: Any, **kwargs) -> tuple:
        if isinstance(data, str):
            img = cv2.imread(data)
            if img is None:
                raise ValueError(f"Failed to read image from path: {data}")
        else:
            img = copy.copy(data)

        use_legacy_params = any(key in kwargs for key in FACE_DET_LEGACY_PARAMS)
        is_cascade_model = hasattr(self.model, "detectMultiScale")

        if use_legacy_params or is_cascade_model:
            legacy_model = self.model if is_cascade_model else cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            scale_factor = 1.1 if "scaleFactor" not in kwargs else float(kwargs["scaleFactor"])
            min_neighbors = 5 if "minNeighbors" not in kwargs else int(kwargs["minNeighbors"])
            min_size = (50, 50) if "minSize" not in kwargs else kwargs["minSize"]
            max_size = img.shape[:2] if "maxSize" not in kwargs else kwargs["maxSize"]
            faces = legacy_model.detectMultiScale(
                img,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                maxSize=max_size,
            )
            boxes = [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in faces]
            return (
                np.array(boxes) if boxes else np.zeros((0, 4)),
                np.zeros(0),
                None,
            )

        h, w = img.shape[:2]
        self.model.setInputSize((w, h))
        self.model.setScoreThreshold(float(kwargs.get("score_threshold", kwargs.get("thr", self._score_threshold))))
        self.model.setNMSThreshold(float(kwargs.get("nms_threshold", self._nms_threshold)))
        self.model.setTopK(int(kwargs.get("top_k", self._top_k)))

        _, faces = self.model.detect(img)

        boxes, scores = [], []
        if faces is not None:
            for det in faces:
                x, y, bw, bh = det[:4]
                score = float(det[14]) if len(det) > 14 else None
                boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
                scores.append(score)

        return (
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(scores) if scores else np.zeros(0),
            None,
        )

    def format_output(self, raw_output: tuple, lang: str = "en") -> dict:
        boxes, scores, _classes = raw_output
        result = {
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
        }
        if isinstance(scores, np.ndarray) and scores.size > 0:
            result["scores"] = scores.tolist()
        return result


class PalmHandDetectorHandler(DetectionHandler):
    """Multi-hand palm detector using a pure ONNX palm model.

    The model emits normalized palm center/size values. This handler keeps
    the public det_hand return shape (boxes, scores, classes) while avoiding
    the legacy 224x224 full-image detector path.
    """

    task_name = "det_hand"
    model_id = "det_hand-palm-onnx"
    hand_box_scale = 2.2

    def _preprocess(self, data: Any) -> np.ndarray:
        raise NotImplementedError("PalmHandDetectorHandler preprocesses inside inference")

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        raise NotImplementedError("PalmHandDetectorHandler postprocesses inside inference")

    @staticmethod
    def decode_palm_boxes(raw_output: np.ndarray, image_shape, threshold: float = 0.6):
        image_h, image_w = image_shape[:2]
        square_size = max(image_h, image_w)
        padding_half = abs(image_h - image_w) // 2
        boxes = []
        scores = []
        for row in np.asarray(raw_output):
            score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = map(float, row[:8])
            if score <= threshold or box_size <= 0:
                continue
            rotation = 0.5 * math.pi - math.atan2(-(kp2_y - kp0_y), kp2_x - kp0_x)
            center_x = box_x + 0.5 * box_size * math.sin(rotation)
            center_y = box_y - 0.5 * box_size * math.cos(rotation)
            center_y = (center_y * square_size - padding_half) / image_h
            side = PalmHandDetectorHandler.hand_box_scale * box_size * square_size
            cx, cy = center_x * image_w, center_y * image_h
            boxes.append([
                max(0.0, cx - side / 2),
                max(0.0, cy - side / 2),
                min(float(image_w), cx + side / 2),
                min(float(image_h), cy + side / 2),
            ])
            scores.append(score)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(scores, dtype=np.float32)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.model = ort.InferenceSession(checkpoint_path, None)
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]

    def inference(self, data: Any, threshold: float = 0.6, thr: Optional[float] = None, **kwargs) -> tuple:
        image = cv2.imread(data) if isinstance(data, str) else np.asarray(data)
        if image is None or image.ndim != 3:
            raise ValueError("det_hand expects an image path or HWC image array")
        input_h, input_w = self.model.get_inputs()[0].shape[2:]
        image_h, image_w = image.shape[:2]
        scale = min(input_w / image_w, input_h / image_h)
        resized_w = max(1, int(round(image_w * scale)))
        resized_h = max(1, int(round(image_h * scale)))
        resized = cv2.resize(image, (resized_w, resized_h))
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        padded[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None]
        raw = self.model.run(self.output_names, {self.input_name: tensor})[0]
        boxes, scores = self.decode_palm_boxes(raw, image.shape, threshold if thr is None else thr)
        classes = np.asarray(["hand"] * len(boxes), dtype=object)
        return boxes, scores, classes

    def format_output(self, raw_output: tuple, lang: str = "en") -> dict:
        boxes, scores, classes = raw_output
        return {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
            "classes": classes.tolist(),
        }
