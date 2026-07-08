# -*- coding: utf-8 -*-
"""
XEdu Handler 基类定义

每个任务对应一个 Handler，负责：
- 模型初始化
- 前处理
- 推理调用
- 后处理
- 输出格式化
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os

import numpy as np
import onnxruntime as ort

from .exceptions import XEduError, XEduDependencyError
from .model_store import get_model_store, check_dependencies
from .model_registry import get_default_model


class BaseHandler(ABC):
    """所有 Handler 的基类"""

    # 子类应覆盖这些属性
    task_name: str = None
    model_id: str = None  # 可为 None，表示使用 registry 中的默认
    input_type: str = "image"
    output_type: str = "detection"

    def __init__(self, checkpoint: Optional[str] = None, download_path: Optional[str] = None, **kwargs):
        """初始化 handler

        Args:
            checkpoint: 本地模型路径（可选，不提供则使用默认）
            download_path: 模型缓存目录（可选）
            **kwargs: 任务特定的参数
        """
        self.checkpoint = checkpoint
        self.download_path = download_path
        self.model = None
        self.kwargs = kwargs

        # 检查依赖
        if self.model_id:
            check_dependencies(self.model_id)

        # 初始化模型
        self._load_model()

    def _load_model(self) -> None:
        """加载模型"""
        # 确定使用哪个模型 ID
        model_id = self.model_id
        if not model_id:
            # 从 registry 中找默认模型
            metadata = get_default_model(self.task_name)
            if metadata:
                model_id = metadata.model_id

        # 获取 checkpoint 路径
        if self.checkpoint:
            checkpoint_path = os.path.abspath(os.path.expanduser(self.checkpoint))
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            store = get_model_store()
            checkpoint_path = store.get_model_path(model_id, auto_download=True)

        # 子类具体加载逻辑
        self._load_checkpoint(checkpoint_path)

    @abstractmethod
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载 checkpoint，由子类实现"""
        pass

    @abstractmethod
    def inference(self, data: Any, **kwargs) -> Any:
        """执行推理，由子类实现"""
        pass

    @abstractmethod
    def format_output(self, raw_output: Any, lang: str = "en") -> Dict[str, Any]:
        """格式化输出，由子类实现"""
        pass


class ONNXHandler(BaseHandler):
    """ONNX 模型 Handler 基类"""

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """加载 ONNX 模型"""
        self.model = ort.InferenceSession(checkpoint_path, None)
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [o.name for o in self.model.get_outputs()]

    def _run_inference(self, ort_inputs: Dict[str, Any]) -> List[np.ndarray]:
        """执行 ONNX 推理"""
        return self.model.run(self.output_names, ort_inputs)


class DetectionHandler(ONNXHandler):
    """检测任务 Handler 基类"""

    output_type = "detection"

    @abstractmethod
    def _preprocess(self, data: Any) -> np.ndarray:
        """前处理"""
        pass

    @abstractmethod
    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """后处理，返回 (boxes, scores, class_ids)"""
        pass

    def inference(self, data: Any, **kwargs) -> tuple:
        """推理：返回 (boxes, scores, class_ids)"""
        img = self._preprocess(data)
        ort_inputs = {self.input_name: img}
        outputs = self._run_inference(ort_inputs)
        return self._postprocess(outputs)

    def format_output(self, raw_output: tuple, lang: str = "en") -> Dict[str, Any]:
        """格式化检测输出"""
        boxes, scores, class_ids = raw_output
        return {
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "class_ids": class_ids.tolist() if isinstance(class_ids, np.ndarray) else class_ids,
        }


class PoseHandler(ONNXHandler):
    """姿态估计任务 Handler 基类"""

    output_type = "pose"

    @abstractmethod
    def _preprocess(self, data: Any, bbox: Optional[List] = None) -> np.ndarray:
        """前处理"""
        pass

    @abstractmethod
    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """后处理，返回 (keypoints, scores)"""
        pass

    def inference(self, data: Any, bbox: Optional[List] = None, **kwargs) -> tuple:
        """推理：返回 (keypoints, scores)"""
        img = self._preprocess(data, bbox)
        ort_inputs = {self.input_name: img}
        outputs = self._run_inference(ort_inputs)
        return self._postprocess(outputs)

    def format_output(self, raw_output: tuple, lang: str = "en") -> Dict[str, Any]:
        """格式化姿态输出"""
        keypoints, scores = raw_output
        return {
            "keypoints": keypoints.tolist() if isinstance(keypoints, np.ndarray) else keypoints,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
        }


class EmbeddingHandler(ONNXHandler):
    """Embedding 任务 Handler 基类"""

    output_type = "embedding"

    @abstractmethod
    def _preprocess(self, data: Any) -> np.ndarray:
        """前处理"""
        pass

    @abstractmethod
    def _postprocess(self, outputs: List[np.ndarray]) -> np.ndarray:
        """后处理，返回 embedding"""
        pass

    def inference(self, data: Any, **kwargs) -> np.ndarray:
        """推理：返回 embedding"""
        processed = self._preprocess(data)
        ort_inputs = {self.input_name: processed}
        outputs = self._run_inference(ort_inputs)
        return self._postprocess(outputs)

    def format_output(self, raw_output: np.ndarray, lang: str = "en") -> Dict[str, Any]:
        """格式化 embedding 输出"""
        return {
            "embedding": raw_output.tolist() if isinstance(raw_output, np.ndarray) else raw_output,
        }


class ClassificationHandler(ONNXHandler):
    """分类任务 Handler 基类"""

    output_type = "classification"

    @abstractmethod
    def _preprocess(self, data: Any) -> np.ndarray:
        """前处理"""
        pass

    @abstractmethod
    def _postprocess(self, outputs: List[np.ndarray]) -> tuple:
        """后处理，返回 (label, score)"""
        pass

    def inference(self, data: Any, **kwargs) -> tuple:
        """推理：返回 (label, score)"""
        img = self._preprocess(data)
        ort_inputs = {self.input_name: img}
        outputs = self._run_inference(ort_inputs)
        return self._postprocess(outputs)

    def format_output(self, raw_output: tuple, lang: str = "en") -> Dict[str, Any]:
        """格式化分类输出"""
        label, score = raw_output
        return {
            "label": label,
            "score": float(score) if isinstance(score, np.ndarray) else score,
        }
