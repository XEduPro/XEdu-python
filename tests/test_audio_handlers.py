# -*- coding: utf-8 -*-
"""Audio handler tests."""

import numpy as np
import pytest

from XEdu.hub.workflow import Workflow


class FakeCLAP:
    def __init__(self):
        self.embeddings = {
            "query.wav": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "cat.wav": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "bell.wav": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def get_audio_embedding(self, audio_files):
        embeddings = []
        for audio_input in audio_files:
            if isinstance(audio_input, np.ndarray):
                embeddings.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            else:
                embeddings.append(self.embeddings[audio_input])
        return np.stack(embeddings)


@pytest.fixture
def fake_audio_loader(monkeypatch):
    def _load(self, checkpoint_path):
        self.model = FakeCLAP()

    monkeypatch.setattr("XEdu.hub.handlers.audio.AudioEmbeddingHandler._load_checkpoint", _load)


def test_embedding_audio_uses_handler_and_formats_output(tmp_path, fake_audio_loader):
    checkpoint = tmp_path / "embedding_audio.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="embedding_audio", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "AudioEmbeddingHandler"

    embedding = wf.inference("query.wav")
    np.testing.assert_array_equal(embedding, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert wf.format_output(lang="en") == {"embedding": [[1.0, 0.0, 0.0]]}


def test_cls_audio_predicts_nearest_reference_prototype(tmp_path, fake_audio_loader):
    checkpoint = tmp_path / "embedding_audio.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="cls_audio", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "AudioPrototypeClassificationHandler"

    result = wf.inference(
        "query.wav",
        prototypes={
            "cat": "cat.wav",
            "bell": "bell.wav",
        },
    )

    assert result["predictions"][0]["label"] == "cat"
    assert result["predictions"][0]["score"] > 0.99
    formatted = wf.format_output(lang="en")
    assert formatted["predictions"][0]["label"] == "cat"
    assert formatted["labels"] == ["cat", "bell"]


def test_cls_audio_accepts_precomputed_prototype_embeddings(tmp_path, fake_audio_loader):
    checkpoint = tmp_path / "embedding_audio.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="cls_audio", checkpoint=str(checkpoint))
    result = wf.inference(
        "query.wav",
        labels=["cat", "bell"],
        prototype_embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert result["predictions"][0] == {"label": "cat", "score": 1.0}


def test_det_audio_keyword_returns_thresholded_keyword_hits(tmp_path, fake_audio_loader):
    checkpoint = tmp_path / "embedding_audio.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="det_audio_keyword", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "AudioKeywordDetectionHandler"

    result = wf.inference(
        "query.wav",
        labels=["cat", "bell"],
        prototype_embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        threshold=0.5,
    )

    assert result["detections"] == [[{"keyword": "cat", "score": 1.0}]]
    assert wf.format_output(lang="en")["detections"][0][0]["keyword"] == "cat"


def test_embedding_audio_accepts_numpy_waveform_input(tmp_path, fake_audio_loader):
    checkpoint = tmp_path / "embedding_audio.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="embedding_audio", checkpoint=str(checkpoint))
    embedding = wf.inference(np.ones(44100, dtype=np.float32))

    np.testing.assert_array_equal(embedding, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))
