# -*- coding: utf-8 -*-
"""
handlers 包初始化
"""

from .base import (
    BaseHandler,
    ONNXHandler,
    DetectionHandler,
    PoseHandler,
    EmbeddingHandler,
    ClassificationHandler,
)
from .audio import (
    AudioEmbeddingHandler,
    AudioKeywordDetectionHandler,
    AudioPrototypeClassificationHandler,
)
from .multimodal import ImageTextMatchingHandler
from .nlp import TextEmbeddingHandler, TextPrototypeClassificationHandler
from .face_landmark import PoseFaceLandmarkHandler
from .detection import PalmHandDetectorHandler
from .pose import PoseBody17Handler

__all__ = [
    "BaseHandler",
    "ONNXHandler",
    "DetectionHandler",
    "PoseHandler",
    "EmbeddingHandler",
    "ClassificationHandler",
    "AudioEmbeddingHandler",
    "AudioKeywordDetectionHandler",
    "AudioPrototypeClassificationHandler",
    "ImageTextMatchingHandler",
    "TextEmbeddingHandler",
    "TextPrototypeClassificationHandler",
    "PoseFaceLandmarkHandler",
    "PalmHandDetectorHandler",
    "PoseBody17Handler",
]
