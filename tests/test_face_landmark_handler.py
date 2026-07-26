# -*- coding: utf-8 -*-
"""Face landmark handler tests."""

import numpy as np

from XEdu.hub.workflow import Workflow


class _Value:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class FakeMobileNet106Session:
    def get_inputs(self):
        return [_Value("image", [1, 3, 96, 96])]

    def get_outputs(self):
        return [_Value("keypoints", [1, 212]), _Value("pose", [1, 3])]

    def run(self, output_names, ort_inputs):
        points = np.tile(np.array([[48.0, 48.0]], dtype=np.float32), (106, 1)).reshape(1, 212)
        pose = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        return [points, pose]


class FakePIPNet98Session:
    def get_inputs(self):
        return [_Value("input", [1, 3, 256, 256])]

    def get_outputs(self):
        return [
            _Value("cls_map", [1, 98, 8, 8]),
            _Value("offset_x", [1, 98, 8, 8]),
            _Value("offset_y", [1, 98, 8, 8]),
            _Value("nb_x", [1, 980, 8, 8]),
            _Value("nb_y", [1, 980, 8, 8]),
        ]

    def run(self, output_names, ort_inputs):
        cls_map = np.zeros((1, 98, 8, 8), dtype=np.float32)
        cls_map[:, :, 0, 0] = 1.0
        offset_x = np.full((1, 98, 8, 8), 0.5, dtype=np.float32)
        offset_y = np.full((1, 98, 8, 8), 0.5, dtype=np.float32)
        nb_x = np.full((1, 980, 8, 8), 0.5, dtype=np.float32)
        nb_y = np.full((1, 980, 8, 8), 0.5, dtype=np.float32)
        return [cls_map, offset_x, offset_y, nb_x, nb_y]


class FakeYuNet:
    def __init__(self, faces=None):
        self.faces = faces
        self.input_size = None

    def setInputSize(self, size):
        self.input_size = size

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def detect(self, image):
        return 1, self.faces


def _face_detection():
    faces = np.zeros((1, 15), dtype=np.float32)
    faces[0, :4] = [100.0, 120.0, 80.0, 100.0]
    faces[0, 14] = 0.88
    return faces


def _patch_common(monkeypatch, tmp_path, session):
    monkeypatch.setattr("XEdu.hub.handlers.face_landmark.ort.InferenceSession", lambda *args, **kwargs: session)
    monkeypatch.setattr(
        "XEdu.hub.handlers.face_landmark.ModelStore.get_model_path",
        lambda self, model_id, auto_download=True: str(tmp_path / "face_detection_yunet_2023mar.onnx"),
    )
    monkeypatch.setattr(
        "XEdu.hub.handlers.face_landmark.cv2.FaceDetectorYN_create",
        lambda *args, **kwargs: FakeYuNet(_face_detection()),
    )


def test_pose_face_defaults_to_mobilenet106(tmp_path, monkeypatch, sample_image):
    _patch_common(monkeypatch, tmp_path, FakeMobileNet106Session())
    checkpoint = tmp_path / "face_landmark106_mobilenet.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="pose_face", checkpoint=str(checkpoint))
    result = wf.inference(sample_image)

    assert wf.handler.model_id == "pose_face_landmark-mobilenet106"
    assert result.shape == (106, 2)
    assert wf.keypoints.shape == (1, 106, 2)
    np.testing.assert_allclose(result[0], [140.0, 170.0], atol=1e-5)
    np.testing.assert_allclose(wf.scores, [0.88], rtol=1e-6)

    formatted = wf.format_output(lang="en")
    assert formatted["point_count"] == 106
    assert formatted["model"] == "mobilenet"
    assert len(formatted["keypoints"][0]) == 106
    assert formatted["poses"] == [[1.0, 2.0, 3.0]]


def test_pose_face_can_select_pipnet98(tmp_path, monkeypatch, sample_image):
    _patch_common(monkeypatch, tmp_path, FakePIPNet98Session())
    checkpoint = tmp_path / "face_landmark98_pipnet_wflw.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(
        task="face_landmark",
        checkpoint=str(checkpoint),
        model_id="pose_face_landmark-pipnet98-wflw",
    )
    result = wf.inference(sample_image)

    assert wf.handler.model_id == "pose_face_landmark-pipnet98-wflw"
    assert result.shape == (98, 2)
    assert wf.keypoints.shape == (1, 98, 2)
    formatted = wf.format_output(lang="en")
    assert formatted["point_count"] == 98
    assert formatted["model"] == "pipnet"


def test_pose_face_returns_empty_array_when_no_face(tmp_path, monkeypatch, sample_image):
    monkeypatch.setattr(
        "XEdu.hub.handlers.face_landmark.ort.InferenceSession",
        lambda *args, **kwargs: FakeMobileNet106Session(),
    )
    monkeypatch.setattr(
        "XEdu.hub.handlers.face_landmark.ModelStore.get_model_path",
        lambda self, model_id, auto_download=True: str(tmp_path / "face_detection_yunet_2023mar.onnx"),
    )
    monkeypatch.setattr(
        "XEdu.hub.handlers.face_landmark.cv2.FaceDetectorYN_create",
        lambda *args, **kwargs: FakeYuNet(None),
    )
    checkpoint = tmp_path / "face_landmark106_mobilenet.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="pose_face", checkpoint=str(checkpoint))
    result = wf.inference(sample_image)

    assert result.shape == (0, 106, 2)
    assert wf.keypoints.shape == (0, 106, 2)
    assert wf.format_output(lang="en")["point_count"] == 106
