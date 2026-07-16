# -*- coding: utf-8 -*-
"""Offline NLP handler tests."""

import numpy as np
import pytest

from XEdu.hub.workflow import Workflow


class FakeTextEmbeddingModel:
    def __init__(self):
        self.embeddings = {
            "query": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "science": np.array([0.9, 0.1, 0.0], dtype=np.float32),
            "sports": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def embed(self, texts):
        return np.stack([self.embeddings[text] for text in texts])


@pytest.fixture
def fake_text_loader(monkeypatch):
    def _load(self, checkpoint_path):
        self.model = FakeTextEmbeddingModel()
        self.batch_size = 16

    def _embed_batch(self, texts):
        return self.model.embed(list(texts))

    monkeypatch.setattr("XEdu.hub.handlers.nlp.TextEmbeddingHandler._load_checkpoint", _load)
    monkeypatch.setattr("XEdu.hub.handlers.nlp.TextEmbeddingHandler._embed_batch", _embed_batch)


def test_embedding_text_uses_handler_and_formats_output(tmp_path, fake_text_loader):
    checkpoint = tmp_path / "embedding_text.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="embedding_text", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "TextEmbeddingHandler"

    embedding = wf.inference("query")
    np.testing.assert_array_equal(embedding, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert wf.format_output(lang="en") == {"embedding": [[1.0, 0.0, 0.0]]}


def test_cls_text_predicts_nearest_reference_prototype(tmp_path, fake_text_loader):
    checkpoint = tmp_path / "embedding_text.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="cls_text", checkpoint=str(checkpoint))
    assert wf.handler.__class__.__name__ == "TextPrototypeClassificationHandler"

    result = wf.inference(
        "query",
        prototypes={
            "science": "science",
            "sports": "sports",
        },
    )

    assert result["predictions"][0]["label"] == "science"
    assert result["predictions"][0]["score"] > 0.99
    formatted = wf.format_output(lang="en")
    assert formatted["predictions"][0]["label"] == "science"
    assert formatted["labels"] == ["science", "sports"]


def test_cls_text_accepts_precomputed_prototype_embeddings(tmp_path, fake_text_loader):
    checkpoint = tmp_path / "embedding_text.onnx"
    checkpoint.write_bytes(b"mock")

    wf = Workflow(task="cls_text", checkpoint=str(checkpoint))
    result = wf.inference(
        "query",
        labels=["science", "sports"],
        prototype_embeddings=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert result["predictions"][0] == {"label": "science", "score": 1.0}
