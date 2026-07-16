# -*- coding: utf-8 -*-
"""Model discovery API tests."""

from XEdu.hub.model_discovery import describe_task, support_models


def test_detection_tasks_expose_multiple_tiers():
    det_body = describe_task("det_body")
    assert det_body["available_models"] == ["det_body-default", "det_body_l-default"]
    assert [m["latency_tier"] for m in det_body["models"]] == ["base", "large"]
    assert det_body["models"][1]["quality_tier"] == "high"

    det_coco = describe_task("det_coco")
    assert det_coco["available_models"] == ["det_coco-default", "det_coco_l-default"]
    assert [m["latency_tier"] for m in det_coco["models"]] == ["base", "large"]


def test_legacy_large_detection_task_names_remain_discoverable():
    assert support_models("det_body_l") == ["det_body_l-compat"]
    assert support_models("det_coco_l") == ["det_coco_l-compat"]


def test_face_landmark_task_exposes_default_and_high_accuracy_tiers():
    info = describe_task("pose_face")
    assert info["available_models"] == [
        "pose_face_landmark-mobilenet106",
        "pose_face_landmark-pipnet98-wflw",
    ]
    assert [m["latency_tier"] for m in info["models"]] == ["tiny", "large"]
    assert info["models"][0]["quality_tier"] == "medium"
    assert info["models"][1]["quality_tier"] == "high"
    assert "high_accuracy" in info["recommended_for"]
