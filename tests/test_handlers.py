# -*- coding: utf-8 -*-
"""Handler 架构的导入与结构测试

审计发现 handlers/base.py 曾有错误的相对导入（from .exceptions 而不是
from ..exceptions），导致整个 handler 包无法导入。这里补上回归测试，
防止同类问题再次悄悄发生。
"""

import pytest


def test_handlers_base_imports():
    """handlers.base 必须能被正常导入，不应该因为相对导入层级错误而失败"""
    from XEdu.hub.handlers.base import (
        BaseHandler,
        ONNXHandler,
        DetectionHandler,
        PoseHandler,
        EmbeddingHandler,
        ClassificationHandler,
    )

    assert issubclass(ONNXHandler, BaseHandler)
    assert issubclass(DetectionHandler, ONNXHandler)


def test_handlers_detection_imports():
    """具体的检测 handler 也必须能导入"""
    from XEdu.hub.handlers.detection import (
        DetBodyHandler,
        DetCocoHandler,
        DetFaceHandler,
        PalmHandDetectorHandler,
    )
    from XEdu.hub.handlers.base import DetectionHandler

    assert issubclass(DetBodyHandler, DetectionHandler)
    assert issubclass(DetCocoHandler, DetectionHandler)
    assert issubclass(DetFaceHandler, DetectionHandler)
    assert issubclass(PalmHandDetectorHandler, DetectionHandler)


def test_handlers_pose_imports():
    """姿态 handler 必须能导入并继承 PoseHandler"""
    from XEdu.hub.handlers.pose import PoseBody17Handler
    from XEdu.hub.handlers.base import PoseHandler

    assert issubclass(PoseBody17Handler, PoseHandler)


def test_handlers_audio_and_nlp_imports():
    """音频和离线 NLP handler 必须能导入"""
    from XEdu.hub.handlers.audio import (
        AudioEmbeddingHandler,
        AudioKeywordDetectionHandler,
        AudioPrototypeClassificationHandler,
    )
    from XEdu.hub.handlers.multimodal import ImageTextMatchingHandler
    from XEdu.hub.handlers.nlp import TextEmbeddingHandler, TextPrototypeClassificationHandler
    from XEdu.hub.handlers.base import BaseHandler

    assert issubclass(AudioEmbeddingHandler, BaseHandler)
    assert issubclass(AudioKeywordDetectionHandler, AudioPrototypeClassificationHandler)
    assert issubclass(AudioPrototypeClassificationHandler, AudioEmbeddingHandler)
    assert issubclass(ImageTextMatchingHandler, BaseHandler)
    assert issubclass(TextEmbeddingHandler, BaseHandler)
    assert issubclass(TextPrototypeClassificationHandler, TextEmbeddingHandler)


def test_handler_task_and_model_ids_are_registered():
    """每个具体 handler 声明的 task_name / model_id 必须在 registry 中真实存在

    这能捕获"handler 写了但对应模型没注册"或"model_id 拼写错误"这类问题。
    """
    from XEdu.hub.handlers.audio import (
        AudioEmbeddingHandler,
        AudioKeywordDetectionHandler,
        AudioPrototypeClassificationHandler,
    )
    from XEdu.hub.handlers.detection import (
        DetBodyHandler,
        DetCocoHandler,
        DetFaceHandler,
        PalmHandDetectorHandler,
    )
    from XEdu.hub.handlers.multimodal import ImageTextMatchingHandler
    from XEdu.hub.handlers.nlp import TextEmbeddingHandler, TextPrototypeClassificationHandler
    from XEdu.hub.handlers.pose import PoseBody17Handler
    from XEdu.hub.model_registry import get_model, list_tasks

    tasks = list_tasks()
    for handler_cls in (
        AudioEmbeddingHandler,
        AudioKeywordDetectionHandler,
        AudioPrototypeClassificationHandler,
        DetBodyHandler,
        DetCocoHandler,
        DetFaceHandler,
        PalmHandDetectorHandler,
        ImageTextMatchingHandler,
        PoseBody17Handler,
        TextEmbeddingHandler,
        TextPrototypeClassificationHandler,
    ):
        assert handler_cls.task_name in tasks, (
            f"{handler_cls.__name__}.task_name={handler_cls.task_name!r} "
            f"未在 model_registry 中注册"
        )
        assert get_model(handler_cls.model_id) is not None, (
            f"{handler_cls.__name__}.model_id={handler_cls.model_id!r} "
            f"在 model_registry 中找不到对应模型"
        )


def test_palm_hand_detector_decodes_normalized_palm_boxes():
    import numpy as np

    from XEdu.hub.handlers.detection import PalmHandDetectorHandler

    raw = np.array(
        [[0.9, 0.5, 0.25, 0.05, 0.5, 0.25, 0.5, 0.2]],
        dtype=np.float32,
    )
    boxes, scores = PalmHandDetectorHandler.decode_palm_boxes(
        raw, image_shape=(1024, 1536), threshold=0.5
    )

    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(scores, [0.9], atol=1e-6)
    np.testing.assert_allclose(boxes[0], [683.52, 5.12, 852.48, 174.08], atol=1e-3)


def test_base_handler_cannot_be_instantiated_directly():
    """BaseHandler 是抽象类，不应该能被直接实例化"""
    from XEdu.hub.handlers.base import BaseHandler

    with pytest.raises(TypeError):
        BaseHandler()
