# -*- coding: utf-8 -*-
"""Multimodal handler tests."""

import numpy as np

from XEdu.hub.model_discovery import describe_task
from XEdu.hub.workflow import Workflow


def test_match_image_text_with_precomputed_embeddings(tmp_path, monkeypatch):
    def _load(self):
        self.loaded_for_test = True

    monkeypatch.setattr("XEdu.hub.handlers.multimodal.ImageTextMatchingHandler._load_model", _load)

    wf = Workflow(task="match_image_text")
    assert wf.handler.__class__.__name__ == "ImageTextMatchingHandler"

    result = wf.inference(
        image_embeddings=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        texts=["a science classroom", "a football match"],
        text_embeddings=np.array(
            [
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert result["matches"][0]["text"] == "a science classroom"
    assert result["matches"][0]["score"] > 0.99
    formatted = wf.format_output(lang="en")
    assert formatted["matches"][0]["text"] == "a science classroom"
    assert formatted["texts"] == ["a science classroom", "a football match"]


def test_match_image_text_is_discoverable_as_composite_task():
    info = describe_task("match_image_text")

    assert info["input_type"] == "image+text"
    assert info["output_type"] == "multimodal"
    assert info["available_models"] == ["match_image_text-clip"]
    assert "multimodal" in info["tags"]
