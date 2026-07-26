# -*- coding: utf-8 -*-
"""Offline NLP task handlers."""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import onnxruntime as ort

from .base import BaseHandler
from ..tokenizer.clip_tokenizer import Tokenizer


def _as_text_list(data: Any) -> List[str]:
    if isinstance(data, str):
        return [data]
    return list(data)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return values / norms


class TextEmbeddingHandler(BaseHandler):
    """CLIP text embedding handler."""

    task_name = "embedding_text"
    model_id = "embedding_text-clip"
    input_type = "text"
    output_type = "embedding"

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        self.model = ort.InferenceSession(checkpoint_path, None)
        self.tokenizer = Tokenizer()
        self.batch_size = self.kwargs.get("batch_size", 16)

    def _empty_embedding(self) -> np.ndarray:
        return np.empty((0, 512), dtype=np.float32)

    def _embed_batch(self, texts: Iterable[str]) -> np.ndarray:
        tokens = self.tokenizer.encode_text(texts)
        if len(tokens) == 0:
            return self._empty_embedding()
        return self.model.run(None, {"TEXT": tokens})[0]

    def inference(self, data: Any, **kwargs) -> np.ndarray:
        texts = _as_text_list(data)
        batch_size = kwargs.get("batch_size", self.batch_size)
        if not batch_size:
            return self._embed_batch(texts)

        embeddings = []
        for start in range(0, len(texts), batch_size):
            embeddings.append(self._embed_batch(texts[start : start + batch_size]))
        if not embeddings:
            return self._empty_embedding()
        return np.concatenate(embeddings)

    def format_output(self, raw_output: np.ndarray, lang: str = "en") -> Dict[str, Any]:
        return {
            "embedding": raw_output.tolist() if isinstance(raw_output, np.ndarray) else raw_output,
        }


class TextPrototypeClassificationHandler(TextEmbeddingHandler):
    """Classify text by cosine similarity to reference text prototypes."""

    task_name = "cls_text"
    model_id = "cls_text-clip-prototype"
    output_type = "classification"

    def inference(
        self,
        data: Any,
        prototypes: Optional[Dict[str, Any]] = None,
        prototype_embeddings: Optional[Any] = None,
        labels: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query_embeddings = super().inference(data, **kwargs)

        if prototype_embeddings is None:
            if not prototypes:
                raise ValueError(
                    "cls_text requires either prototype_embeddings or "
                    "prototypes={label: text_or_texts}."
                )
            labels = list(prototypes.keys())
            prototype_vectors = []
            for label in labels:
                embeddings = super().inference(prototypes[label], **kwargs)
                prototype_vectors.append(np.mean(embeddings, axis=0))
            prototype_embeddings = np.stack(prototype_vectors)
        else:
            prototype_embeddings = np.asarray(prototype_embeddings, dtype=np.float32)
            if labels is None:
                labels = [str(i) for i in range(prototype_embeddings.shape[0])]
            else:
                labels = list(labels)

        if len(labels) != prototype_embeddings.shape[0]:
            raise ValueError("labels length must match prototype_embeddings rows.")

        query_norm = _normalize_rows(np.asarray(query_embeddings, dtype=np.float32))
        proto_norm = _normalize_rows(np.asarray(prototype_embeddings, dtype=np.float32))
        similarities = query_norm @ proto_norm.T
        best_indices = np.argmax(similarities, axis=1)

        predictions = [
            {
                "label": labels[int(idx)],
                "score": float(similarities[row, idx]),
            }
            for row, idx in enumerate(best_indices)
        ]
        return {
            "predictions": predictions,
            "similarities": similarities,
            "labels": labels,
        }

    def format_output(self, raw_output: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        return {
            "predictions": raw_output["predictions"],
            "labels": list(raw_output["labels"]),
            "similarities": raw_output["similarities"].tolist()
            if isinstance(raw_output["similarities"], np.ndarray)
            else raw_output["similarities"],
        }
