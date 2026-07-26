# -*- coding: utf-8 -*-
"""Multimodal task handlers."""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from .base import BaseHandler
from ..model_store import ModelStore, get_model_store
from ..tokenizer.clip_tokenizer import Preprocessor, Tokenizer
from ...utils.utils import get_similarity


def _to_batches(items: List[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ImageTextMatchingHandler(BaseHandler):
    """Match images against candidate text prompts using CLIP embeddings."""

    task_name = "match_image_text"
    model_id = "match_image_text-clip"
    input_type = "image+text"
    output_type = "multimodal"

    def __init__(self, checkpoint: Optional[Any] = None, download_path: Optional[str] = None, **kwargs):
        self.image_model_id = kwargs.get("image_model_id", "embedding_image-clip")
        self.text_model_id = kwargs.get("text_model_id", "embedding_text-clip")
        self.batch_size = kwargs.get("batch_size", 16)
        super().__init__(checkpoint=checkpoint, download_path=download_path, **kwargs)

    def _load_model(self) -> None:
        if isinstance(self.checkpoint, dict):
            image_path = self.checkpoint["image"]
            text_path = self.checkpoint["text"]
        elif isinstance(self.checkpoint, (list, tuple)):
            if len(self.checkpoint) != 2:
                raise ValueError("match_image_text checkpoint must contain image and text model paths.")
            image_path, text_path = self.checkpoint
        else:
            store = ModelStore(self.download_path) if self.download_path else get_model_store()
            image_path = store.get_model_path(self.image_model_id, auto_download=True)
            text_path = store.get_model_path(self.text_model_id, auto_download=True)

        self._load_checkpoint({"image": image_path, "text": text_path})

    def _load_checkpoint(self, checkpoint_path: Dict[str, str]) -> None:
        self.image_model = ort.InferenceSession(checkpoint_path["image"], None)
        self.text_model = ort.InferenceSession(checkpoint_path["text"], None)
        self.preprocessor = Preprocessor()
        self.tokenizer = Tokenizer()

    def _empty_embedding(self) -> np.ndarray:
        return np.empty((0, 512), dtype=np.float32)

    def _as_images(self, data: Any) -> List[Any]:
        if isinstance(data, str):
            return [Image.open(data).convert("RGB")]
        if isinstance(data, Image.Image):
            return [data.convert("RGB")]
        if isinstance(data, np.ndarray):
            if len(data.shape) == 3:
                data = np.expand_dims(data, axis=0)
            if data.shape[-1] != 3 and data.shape[-3] == 3:
                data = np.transpose(data, (0, 2, 3, 1))
            return list(data)
        if isinstance(data, Iterable) and all(isinstance(item, str) for item in data):
            return [Image.open(item).convert("RGB") for item in data]
        return list(data)

    def _embed_images(self, images: Any) -> np.ndarray:
        image_list = self._as_images(images)
        embeddings = []
        for batch in _to_batches(image_list, self.batch_size):
            encoded = [self.preprocessor.encode_image(image) for image in batch]
            if encoded:
                embeddings.append(self.image_model.run(None, {"IMAGE": np.concatenate(encoded)})[0])
        if not embeddings:
            return self._empty_embedding()
        return np.concatenate(embeddings)

    def _embed_texts(self, texts: Any) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        embeddings = []
        for batch in _to_batches(texts, self.batch_size):
            tokens = self.tokenizer.encode_text(batch)
            if len(tokens) > 0:
                embeddings.append(self.text_model.run(None, {"TEXT": tokens})[0])
        if not embeddings:
            return self._empty_embedding()
        return np.concatenate(embeddings)

    def inference(
        self,
        data: Any = None,
        texts: Optional[Iterable[str]] = None,
        image_embeddings: Optional[Any] = None,
        text_embeddings: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if image_embeddings is None:
            image_embeddings = self._embed_images(data)
        else:
            image_embeddings = np.asarray(image_embeddings, dtype=np.float32)

        if text_embeddings is None:
            if texts is None:
                raise ValueError("match_image_text requires texts or text_embeddings.")
            text_labels = list(texts) if not isinstance(texts, str) else [texts]
            text_embeddings = self._embed_texts(text_labels)
        else:
            text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
            if texts is None:
                text_labels = [str(i) for i in range(text_embeddings.shape[0])]
            else:
                text_labels = list(texts) if not isinstance(texts, str) else [texts]

        if len(text_labels) != text_embeddings.shape[0]:
            raise ValueError("texts length must match text_embeddings rows.")

        similarities = np.asarray(
            get_similarity(image_embeddings, text_embeddings, method="cosine", use_softmax=False),
            dtype=np.float32,
        )
        best_indices = np.argmax(similarities, axis=1)
        matches = [
            {
                "text": text_labels[int(idx)],
                "score": float(similarities[row, idx]),
            }
            for row, idx in enumerate(best_indices)
        ]
        return {
            "matches": matches,
            "similarities": similarities,
            "texts": text_labels,
        }

    def format_output(self, raw_output: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        return {
            "matches": raw_output["matches"],
            "texts": list(raw_output["texts"]),
            "similarities": raw_output["similarities"].tolist()
            if isinstance(raw_output["similarities"], np.ndarray)
            else raw_output["similarities"],
        }
