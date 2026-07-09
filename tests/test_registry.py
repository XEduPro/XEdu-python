# -*- coding: utf-8 -*-
"""Test model registry"""

import pytest
from XEdu.hub.model_registry import (
    get_model,
    get_models_by_task,
    get_default_model,
    list_tasks,
    list_all_models,
)


def test_registry_has_models():
    """Verify registry has expected models"""
    tasks = list_tasks()
    assert len(tasks) > 0
    assert "det_body" in tasks
    assert "pose_body17" in tasks
    assert "cls_imagenet" in tasks


def test_get_default_model():
    """Test getting default model for a task"""
    model = get_default_model("det_body")
    assert model is not None
    assert model.task_name == "det_body"


def test_get_model_by_id():
    """Test getting model by ID"""
    model = get_model("det_body-default")
    assert model is not None
    assert model.filename == "bodydetect.onnx"


def test_list_all_models():
    """Test listing all models"""
    models = list_all_models()
    assert len(models) > 20  # Should have many built-in models


def test_model_metadata():
    """Test model metadata structure"""
    model = get_default_model("pose_body17")
    assert model.input_type == "image"
    assert model.output_type == "pose"
