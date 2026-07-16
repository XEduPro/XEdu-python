# -*- coding: utf-8 -*-
"""Face landmark task handlers."""

from __future__ import annotations

import copy
import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from ._pipnet_meanface import get_wflw98_meanface_info
from .base import PoseHandler
from ..model_store import ModelStore, get_model_store


class PoseFaceLandmarkHandler(PoseHandler):
    """Face landmark handler with YuNet face detection + ONNX landmark models."""

    task_name = "pose_face"
    model_id = "pose_face_landmark-mobilenet106"

    _MOBILENET_MEAN = np.array([0.4076, 0.4580, 0.4850], dtype=np.float32)
    _PIPNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    _PIPNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    _PIPNET_NUM_NB = 10

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        download_path: Optional[str] = None,
        **kwargs,
    ):
        self.detector_model_id = kwargs.get("detector_model_id", "det_face-yunet")
        self.score_threshold = float(kwargs.get("score_threshold", 0.6))
        self.nms_threshold = float(kwargs.get("nms_threshold", 0.3))
        self.top_k = int(kwargs.get("top_k", 5000))
        self.max_faces = int(kwargs.get("max_faces", 1))
        self.expand_ratio = float(kwargs.get("expand_ratio", 0.1))
        self.face_boxes = np.zeros((0, 4), dtype=np.float32)
        self.poses = np.zeros((0, 3), dtype=np.float32)
        self._model_kind = "mobilenet"
        self._input_w = 96
        self._input_h = 96
        self._feat_w = 0
        self._feat_h = 0
        self._net_stride = 0
        self._reverse_index1 = None
        self._reverse_index2 = None
        self._reverse_max_len = 0
        self.detector = None
        super().__init__(checkpoint=checkpoint, download_path=download_path, **kwargs)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.model = ort.InferenceSession(checkpoint_path, None)
        input_cfg = self.model.get_inputs()[0]
        self.input_name = input_cfg.name
        self.output_names = [o.name for o in self.model.get_outputs()]

        shape = input_cfg.shape
        if len(shape) >= 4:
            self._input_h = int(shape[2]) if isinstance(shape[2], int) else 96
            self._input_w = int(shape[3]) if isinstance(shape[3], int) else 96

        output_count = len(self.output_names)
        output_shapes = [o.shape for o in self.model.get_outputs()]
        is_pipnet = output_count >= 5 or "pipnet" in os.path.basename(checkpoint_path)
        self._model_kind = "pipnet" if is_pipnet else "mobilenet"

        if self._model_kind == "pipnet":
            cls_shape = output_shapes[0]
            self._feat_h = int(cls_shape[2]) if isinstance(cls_shape[2], int) else max(1, self._input_h // 32)
            self._feat_w = int(cls_shape[3]) if isinstance(cls_shape[3], int) else max(1, self._input_w // 32)
            self._net_stride = max(1, self._input_h // self._feat_h)
            self._reverse_index1, self._reverse_index2, self._reverse_max_len = get_wflw98_meanface_info(
                self._PIPNET_NUM_NB
            )

        store = ModelStore(self.download_path) if self.download_path else get_model_store()
        detector_path = store.get_model_path(self.detector_model_id, auto_download=True)
        self.detector = cv2.FaceDetectorYN_create(
            detector_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    def _preprocess(self, data: Any, bbox: Optional[List] = None) -> np.ndarray:
        raise NotImplementedError

    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        raise NotImplementedError

    def inference(self, data: Any, bbox: Optional[List] = None, **kwargs) -> tuple:
        image = self._read_image(data)
        boxes, face_scores = self._resolve_faces(image, bbox)

        self.face_boxes = boxes
        if boxes.size == 0:
            num_points = 98 if self._model_kind == "pipnet" else 106
            self.poses = np.zeros((0, 3), dtype=np.float32)
            return np.zeros((0, num_points, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

        keypoints = []
        poses = []
        valid_scores = []
        valid_boxes = []
        for box, face_score in zip(boxes, face_scores):
            result = self._run_pipnet(image, box) if self._model_kind == "pipnet" else self._run_mobilenet(image, box)
            if result is None:
                continue
            points, pose = result
            keypoints.append(points)
            poses.append(pose)
            valid_scores.append(float(face_score))
            valid_boxes.append(box)

        if not keypoints:
            num_points = 98 if self._model_kind == "pipnet" else 106
            self.face_boxes = np.zeros((0, 4), dtype=np.float32)
            self.poses = np.zeros((0, 3), dtype=np.float32)
            return np.zeros((0, num_points, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

        self.face_boxes = np.asarray(valid_boxes, dtype=np.float32)
        self.poses = np.asarray(poses, dtype=np.float32)
        return np.asarray(keypoints, dtype=np.float32), np.asarray(valid_scores, dtype=np.float32)

    def format_output(self, raw_output: tuple, lang: str = "en") -> dict:
        keypoints, scores = raw_output
        return {
            "keypoints": keypoints.tolist() if isinstance(keypoints, np.ndarray) else keypoints,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "boxes": self.face_boxes.tolist() if isinstance(self.face_boxes, np.ndarray) else self.face_boxes,
            "poses": self.poses.tolist() if isinstance(self.poses, np.ndarray) else self.poses,
            "point_count": int(keypoints.shape[1]) if isinstance(keypoints, np.ndarray) and keypoints.ndim == 3 else 0,
            "model": self._model_kind,
        }

    def _read_image(self, data: Any) -> np.ndarray:
        if isinstance(data, str):
            image = cv2.imread(data)
            if image is None:
                raise ValueError(f"Failed to read image from path: {data}")
            return image
        image = copy.copy(data)
        if not isinstance(image, np.ndarray):
            raise TypeError("Face landmark input must be an image path or numpy.ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Face landmark input must be a BGR image with shape (H, W, 3).")
        return image

    def _resolve_faces(self, image: np.ndarray, bbox: Optional[List]) -> Tuple[np.ndarray, np.ndarray]:
        if bbox is not None:
            boxes = self._normalize_bbox_list(bbox)
            return boxes, np.ones(boxes.shape[0], dtype=np.float32)

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(self.score_threshold)
        self.detector.setNMSThreshold(self.nms_threshold)
        self.detector.setTopK(self.top_k)
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)

        boxes = []
        scores = []
        for det in faces:
            x, y, bw, bh = det[:4]
            score = float(det[14]) if len(det) > 14 else 1.0
            boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
            scores.append(score)

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-scores_arr)
        if self.max_faces > 0:
            order = order[: self.max_faces]
        return boxes_arr[order], scores_arr[order]

    @staticmethod
    def _normalize_bbox_list(bbox: List) -> np.ndarray:
        boxes = np.asarray(bbox, dtype=np.float32)
        if boxes.ndim == 1:
            if boxes.size != 4:
                raise ValueError("bbox must be [x1, y1, x2, y2] or an array of such boxes.")
            boxes = boxes.reshape(1, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("bbox must be [x1, y1, x2, y2] or an array with shape (N, 4).")
        return boxes

    @staticmethod
    def _clip_box(box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        x1 = max(0, min(width - 1, int(np.floor(x1))))
        y1 = max(0, min(height - 1, int(np.floor(y1))))
        x2 = max(0, min(width, int(np.ceil(x2))))
        y2 = max(0, min(height, int(np.ceil(y2))))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _expand_box(self, box: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        x1 -= bw * self.expand_ratio
        y1 -= bh * self.expand_ratio
        x2 += bw * self.expand_ratio
        y2 += bh * self.expand_ratio
        return self._clip_box(np.asarray([x1, y1, x2, y2], dtype=np.float32), width, height)

    def _run_mobilenet(self, image: np.ndarray, box: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        h, w = image.shape[:2]
        crop_box = self._expand_box(box, w, h)
        if crop_box is None:
            return None
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, (self._input_w, self._input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb - self._MOBILENET_MEAN
        blob = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.float32)
        outputs = self.model.run(self.output_names, {self.input_name: np.ascontiguousarray(blob)})

        points = outputs[0][0].reshape(-1, 2).astype(np.float32)
        points[:, 0] = points[:, 0] / float(self._input_w) * float(x2 - x1) + x1
        points[:, 1] = points[:, 1] / float(self._input_h) * float(y2 - y1) + y1
        pose = outputs[1][0].astype(np.float32) if len(outputs) > 1 else np.zeros(3, dtype=np.float32)
        return points, pose

    def _run_pipnet(self, image: np.ndarray, box: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        crop_info = self._crop_pipnet(image, box)
        if crop_info is None:
            return None
        crop, x1, y1, crop_w, crop_h = crop_info
        blob = self._preprocess_pipnet(crop)
        outputs = self.model.run(self.output_names, {self.input_name: blob})
        points = self._decode_pipnet(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
        points[:, 0] = points[:, 0] * float(crop_w) + x1
        points[:, 1] = points[:, 1] * float(crop_h) + y1
        return points.astype(np.float32), np.zeros(3, dtype=np.float32)

    @staticmethod
    def _crop_pipnet(image: np.ndarray, box: np.ndarray) -> Optional[Tuple[np.ndarray, int, int, int, int]]:
        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        det_w = x2 - x1 + 1
        det_h = y2 - y1 + 1

        pad = 0.1
        x1 -= int(det_w * pad)
        y1 += int(det_h * pad)
        x2 += int(det_w * pad)
        y2 += int(det_h * pad)

        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), img_w - 1)
        y2 = min(int(y2), img_h - 1)
        crop_w = x2 - x1 + 1
        crop_h = y2 - y1 + 1
        if crop_w <= 0 or crop_h <= 0:
            return None

        return image[y1 : y2 + 1, x1 : x2 + 1, :], x1, y1, crop_w, crop_h

    def _preprocess_pipnet(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (self._input_w, self._input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - self._PIPNET_MEAN) / self._PIPNET_STD
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def _decode_pipnet(
        self,
        cls_map: np.ndarray,
        offset_x: np.ndarray,
        offset_y: np.ndarray,
        nb_x: np.ndarray,
        nb_y: np.ndarray,
    ) -> np.ndarray:
        num_lms = int(cls_map.shape[1])
        feat_h = int(cls_map.shape[2])
        feat_w = int(cls_map.shape[3])

        cls_flat = cls_map.reshape(num_lms, feat_h * feat_w)
        max_ids = np.argmax(cls_flat, axis=1)
        cols = (max_ids % feat_w).astype(np.float32)
        rows = (max_ids // feat_w).astype(np.float32)

        off_x_flat = offset_x.reshape(num_lms, feat_h * feat_w)
        off_y_flat = offset_y.reshape(num_lms, feat_h * feat_w)
        own_x = np.take_along_axis(off_x_flat, max_ids[:, None], axis=1).squeeze(1)
        own_y = np.take_along_axis(off_y_flat, max_ids[:, None], axis=1).squeeze(1)

        nb = self._PIPNET_NUM_NB
        nb_x_flat = nb_x.reshape(num_lms, nb, feat_h * feat_w)
        nb_y_flat = nb_y.reshape(num_lms, nb, feat_h * feat_w)
        nb_ids = np.broadcast_to(max_ids[:, None, None], (num_lms, nb, 1))
        nb_own_x = np.take_along_axis(nb_x_flat, nb_ids, axis=2).squeeze(2)
        nb_own_y = np.take_along_axis(nb_y_flat, nb_ids, axis=2).squeeze(2)

        scale_x = float(self._input_w) / float(self._net_stride)
        scale_y = float(self._input_h) / float(self._net_stride)
        pred_x = (cols + own_x) / scale_x
        pred_y = (rows + own_y) / scale_y

        if num_lms != 98 or self._reverse_index1 is None or self._reverse_index2 is None:
            return np.stack([pred_x, pred_y], axis=1)

        nb_pred_x = (cols[:, None] + nb_own_x) / scale_x
        nb_pred_y = (rows[:, None] + nb_own_y) / scale_y
        rev_x = nb_pred_x.reshape(-1)[self._reverse_index1 * nb + self._reverse_index2].reshape(
            num_lms, self._reverse_max_len
        )
        rev_y = nb_pred_y.reshape(-1)[self._reverse_index1 * nb + self._reverse_index2].reshape(
            num_lms, self._reverse_max_len
        )

        merged_x = np.mean(np.concatenate([pred_x[:, None], rev_x], axis=1), axis=1)
        merged_y = np.mean(np.concatenate([pred_y[:, None], rev_y], axis=1), axis=1)
        return np.stack([merged_x, merged_y], axis=1)
