# -*- coding: utf-8 -*-
"""Pose handler equivalence tests."""

import numpy as np

from XEdu.hub.workflow import Workflow


class _Value:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class MockPoseSession:
    def __init__(self):
        self.calls = []

    def get_modelmeta(self):
        class Meta:
            custom_metadata_map = {}

        return Meta()

    def get_inputs(self):
        return [_Value("input", [1, 3, 64, 48])]

    def get_outputs(self):
        return [_Value("simcc_x"), _Value("simcc_y")]

    def run(self, output_names, ort_inputs):
        self.calls.append((output_names, ort_inputs))
        simcc_x = np.zeros((1, 17, 96), dtype=np.float32)
        simcc_y = np.zeros((1, 17, 128), dtype=np.float32)
        for idx in range(17):
            simcc_x[0, idx, idx + 10] = 0.6 + idx * 0.01
            simcc_y[0, idx, idx + 20] = 0.7 + idx * 0.01
        return [simcc_x, simcc_y]


def _workflow_with_model(model):
    wf = object.__new__(Workflow)
    wf.task = "pose_body17"
    wf.model = model
    wf.handler = None
    wf.repo = None
    return wf


def test_pose_body17_handler_matches_legacy_pose_infer(tmp_path, monkeypatch, sample_image):
    model = MockPoseSession()
    bbox = np.array([20, 30, 200, 280])
    legacy = _workflow_with_model(model)
    expected = legacy._pose_infer(sample_image, bbox=bbox)

    monkeypatch.setattr(
        "XEdu.hub.handlers.base.ONNXHandler._load_checkpoint",
        lambda self, checkpoint_path: setattr(self, "model", model),
    )
    checkpoint = tmp_path / "body17.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="pose_body17", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "PoseBody17Handler"

    actual = wf.inference(sample_image, bbox=bbox)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(wf.scores, legacy.scores)
    assert wf.format_output(lang="en") == {
        "keypoints": wf.keypoints[0].tolist(),
        "scores": wf.scores[0].tolist(),
    }
