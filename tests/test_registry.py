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
    assert "pose_face" in tasks
    assert "pose_face_landmark" not in tasks
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


def test_remote_models_use_modelscope_primary_with_mirrors_and_checksums():
    """All auto-download remote models should use ModelScope as primary source.

    OpenXLab/GitHub legacy URLs are retained as mirrors so existing hosting can
    still act as a fallback.
    """
    bad = []
    for model in list_all_models():
        if model.source_type != "remote" or not model.auto_download:
            continue
        if not model.source_url:
            bad.append((model.model_id, "missing source_url"))
            continue
        if "modelscope.cn/models/wht0926/xedu-hub-models/resolve/" not in model.source_url:
            bad.append((model.model_id, "primary is not modelscope"))
        if not model.checksum:
            bad.append((model.model_id, "missing checksum"))
        if model.filename.endswith(".onnx") and not model.mirror_urls and "derived" not in model.tags:
            bad.append((model.model_id, "missing mirror_urls"))
    assert bad == []


def test_list_all_models():
    """Test listing all models"""
    models = list_all_models()
    assert len(models) > 20  # Should have many built-in models


def test_model_metadata():
    """Test model metadata structure"""
    model = get_default_model("pose_body17")
    assert model.input_type == "image"
    assert model.output_type == "pose"


def test_no_placeholder_or_malformed_source_urls():
    """Regression test: every remote model must have a real, well-formed
    download URL. This catches accidentally committed placeholders like
    '...&name=foo.onnx' that look valid but 404 at runtime.
    """
    bad = []
    for model in list_all_models():
        if model.source_type != "remote":
            continue
        if not model.auto_download:
            # Explicitly disabled auto-download models (e.g. pose_face106)
            # are allowed to have no source_url.
            continue
        url = model.source_url
        if not url or "..." in url or not url.startswith("http"):
            bad.append(model.model_id)
    assert bad == [], f"Found placeholder/malformed source_url for: {bad}"


def test_pose_face106_is_registered_but_disabled():
    """pose_face106 must be discoverable, but must not silently offer a
    download URL, since the legacy upstream URL incorrectly pointed at
    pose_wholebody133's model file.
    """
    model = get_default_model("pose_face106")
    assert model is not None
    assert model.auto_download is False


def test_pose_face_default_and_high_accuracy_models():
    """pose_face should default to the lightweight MobileNet 106
    model while also exposing the PIPNet/WFLW-98 high-accuracy option.
    """
    default_model = get_default_model("pose_face")
    assert default_model is not None
    assert default_model.model_id == "pose_face_landmark-mobilenet106"
    assert default_model.filename == "face_landmark106_mobilenet.onnx"
    assert "lightweight" in default_model.tags
    assert "derived" in default_model.tags

    models = get_models_by_task("pose_face")
    model_ids = {model.model_id for model in models}
    assert "pose_face_landmark-mobilenet106" in model_ids
    assert "pose_face_landmark-pipnet98-wflw" in model_ids

    high_accuracy = get_model("pose_face_landmark-pipnet98-wflw")
    assert high_accuracy.quality_tier == "high"
    assert high_accuracy.filename == "face_landmark98_pipnet_wflw.onnx"
    assert high_accuracy.mirror_urls


def test_segment_anything_has_encoder_and_decoder_models():
    """segment_anything needs two files (encoder + decoder); the registry
    represents this as two model entries under the same task rather than
    one entry with a single, meaningless filename.
    """
    models = get_models_by_task("segment_anything")
    model_ids = {m.model_id for m in models}
    assert "segment_anything-encoder" in model_ids
    assert "segment_anything-decoder" in model_ids
    for m in models:
        assert m.source_url.startswith("http")
        assert "..." not in m.source_url
