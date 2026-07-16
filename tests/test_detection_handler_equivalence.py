# -*- coding: utf-8 -*-
"""Detection handler equivalence tests."""

import numpy as np

from XEdu.hub.workflow import Workflow


class MockONNXSession:
    def __init__(self):
        self.calls = []

    def get_modelmeta(self):
        class Meta:
            custom_metadata_map = {}

        return Meta()

    def run(self, output_names, ort_inputs):
        self.calls.append((output_names, ort_inputs))
        boxes = np.array(
            [
                [
                    [10.0, 20.0, 30.0, 40.0, 0.90],
                    [50.0, 60.0, 70.0, 80.0, 0.40],
                    [5.0, 6.0, 9.0, 12.0, 0.95],
                ]
            ],
            dtype=np.float32,
        )
        class_ids = np.array([[0, 1, 2]], dtype=np.int64)
        return [boxes, class_ids]


class MockYuNet:
    def __init__(self):
        self.input_size = None
        self.score_threshold = None
        self.nms_threshold = None
        self.top_k = None

    def setInputSize(self, size):
        self.input_size = size

    def setScoreThreshold(self, threshold):
        self.score_threshold = threshold

    def setNMSThreshold(self, threshold):
        self.nms_threshold = threshold

    def setTopK(self, top_k):
        self.top_k = top_k

    def detect(self, image):
        face = np.zeros(15, dtype=np.float32)
        face[:4] = [3.0, 4.0, 20.0, 30.0]
        face[14] = 0.77
        return 1, np.array([face])


class MockCascade:
    def detectMultiScale(self, image, scaleFactor, minNeighbors, minSize, maxSize):
        assert scaleFactor == 1.2
        assert minNeighbors == 3
        assert minSize == (10, 10)
        return np.array([[1, 2, 30, 40]])


def _workflow_with_model(task, model):
    wf = object.__new__(Workflow)
    wf.task = task
    wf.model = model
    wf.handler = None
    wf.repo = None
    return wf


def test_det_body_handler_matches_legacy_det_infer(tmp_path, monkeypatch, sample_image):
    model = MockONNXSession()
    legacy = _workflow_with_model("det_body", model)
    expected = legacy._det_infer(sample_image, threshold=0.5)

    monkeypatch.setattr(
        "XEdu.hub.handlers.base.ONNXHandler._load_checkpoint",
        lambda self, checkpoint_path: setattr(self, "model", model),
    )
    checkpoint = tmp_path / "unused.onnx"
    checkpoint.write_bytes(b"mock")
    wf = Workflow(task="det_body", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "DetBodyHandler"

    actual = wf.inference(sample_image, thr=0.5)
    np.testing.assert_allclose(actual, expected)
    assert wf.scores == legacy.scores
    assert wf.classes == legacy.classes


def test_det_coco_handler_matches_legacy_target_class_filter(tmp_path, monkeypatch, sample_image):
    model = MockONNXSession()
    legacy = _workflow_with_model("det_coco", model)
    expected = legacy._det_infer(sample_image, threshold=0.5, target_class=["person", "car"])

    monkeypatch.setattr(
        "XEdu.hub.handlers.base.ONNXHandler._load_checkpoint",
        lambda self, checkpoint_path: setattr(self, "model", model),
    )
    checkpoint = tmp_path / "unused.onnx"
    checkpoint.write_bytes(b"mock")
    wf = Workflow(task="det_coco", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "DetCocoHandler"

    actual = wf.inference(sample_image, thr=0.5, target_class=["person", "car"])
    np.testing.assert_allclose(actual, expected)
    assert wf.classes == ["person", "car"]
    assert wf.format_output(lang="en") == {
        "bounding boxes": wf.bboxs,
        "scores": wf.scores,
        "class": wf.classes,
    }


def test_detection_model_id_selects_registered_tier(tmp_path, monkeypatch):
    requested = []

    def _fake_get_model_path(self, model_id, auto_download=True):
        requested.append((model_id, self.cache_dir))
        path = tmp_path / f"{model_id}.onnx"
        path.write_bytes(b"mock")
        return str(path)

    monkeypatch.setattr("XEdu.hub.model_store.ModelStore.get_model_path", _fake_get_model_path)
    monkeypatch.setattr(
        "XEdu.hub.handlers.base.ONNXHandler._load_checkpoint",
        lambda self, checkpoint_path: setattr(self, "model", MockONNXSession()),
    )

    wf = Workflow(task="det_body", download_path=str(tmp_path), model_id="det_body_l-default")

    assert wf.handler.model_id == "det_body_l-default"
    assert requested == [("det_body_l-default", str(tmp_path))]


def test_det_face_handler_yunet_path_is_used(tmp_path, monkeypatch, sample_image):
    yunet = MockYuNet()
    monkeypatch.setattr(
        "XEdu.hub.handlers.detection.cv2.FaceDetectorYN_create",
        lambda *args, **kwargs: yunet,
    )

    checkpoint = tmp_path / "unused.onnx"
    checkpoint.write_bytes(b"mock")
    wf = Workflow(task="det_face", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "DetFaceHandler"

    result = wf.inference(sample_image, thr=0.66, nms_threshold=0.2, top_k=10)
    np.testing.assert_allclose(result, np.array([[3.0, 4.0, 23.0, 34.0]]))
    assert wf.scores == [0.7699999809265137]
    assert wf.classes is None
    assert yunet.input_size == (sample_image.shape[1], sample_image.shape[0])
    assert yunet.score_threshold == 0.66
    assert yunet.nms_threshold == 0.2
    assert yunet.top_k == 10


def test_det_face_handler_legacy_cascade_path_is_preserved(tmp_path, monkeypatch, sample_image):
    monkeypatch.setattr(
        "XEdu.hub.handlers.detection.cv2.CascadeClassifier",
        lambda *args, **kwargs: MockCascade(),
    )

    checkpoint = tmp_path / "unused.xml"
    checkpoint.write_bytes(b"<xml />")
    wf = Workflow(task="det_face", checkpoint=str(checkpoint))
    result = wf.inference(sample_image, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
    np.testing.assert_array_equal(result, np.array([[1, 2, 31, 42]]))
    assert wf.scores == []
    assert wf.classes is None
