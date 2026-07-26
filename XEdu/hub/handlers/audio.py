# -*- coding: utf-8 -*-
"""Audio task handlers."""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .base import BaseHandler


def _as_audio_list(data: Any) -> List[str]:
    if isinstance(data, str):
        return [data]
    if isinstance(data, np.ndarray):
        return [data]
    if (
        isinstance(data, tuple)
        and len(data) == 2
        and isinstance(data[1], (int, np.integer))
    ):
        return [data]
    if isinstance(data, dict):
        return [data]
    return list(data)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return values / norms


class AudioEmbeddingHandler(BaseHandler):
    """CLAP audio embedding handler."""

    task_name = "embedding_audio"
    model_id = "embedding_audio-clap"
    input_type = "audio"
    output_type = "embedding"

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        from ..models.clap import CLAP

        self.model = CLAP(model_path=checkpoint_path)

    def inference(self, data: Any, **kwargs) -> np.ndarray:
        return self.model.get_audio_embedding(_as_audio_list(data))

    def format_output(self, raw_output: np.ndarray, lang: str = "en") -> Dict[str, Any]:
        return {
            "embedding": raw_output.tolist() if isinstance(raw_output, np.ndarray) else raw_output,
        }


class AudioPrototypeClassificationHandler(AudioEmbeddingHandler):
    """Classify audio by cosine similarity to reference audio prototypes."""

    task_name = "cls_audio"
    model_id = "cls_audio-clap-prototype"
    output_type = "classification"

    def inference(
        self,
        data: Any,
        prototypes: Optional[Dict[str, Any]] = None,
        prototype_embeddings: Optional[Any] = None,
        labels: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query_embeddings = self.model.get_audio_embedding(_as_audio_list(data))

        if prototype_embeddings is None:
            if not prototypes:
                raise ValueError(
                    "cls_audio requires either prototype_embeddings or "
                    "prototypes={label: audio_path_or_paths}."
                )
            labels = list(prototypes.keys())
            prototype_vectors = []
            for label in labels:
                embeddings = self.model.get_audio_embedding(_as_audio_list(prototypes[label]))
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


class AudioKeywordDetectionHandler(AudioPrototypeClassificationHandler):
    """Detect keyword-like audio events with CLAP prototype similarities."""

    task_name = "det_audio_keyword"
    model_id = "det_audio_keyword-clap-prototype"
    output_type = "detection"

    def inference(
        self,
        data: Any,
        prototypes: Optional[Dict[str, Any]] = None,
        prototype_embeddings: Optional[Any] = None,
        labels: Optional[Iterable[str]] = None,
        threshold: float = 0.3,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        result = super().inference(
            data,
            prototypes=prototypes,
            prototype_embeddings=prototype_embeddings,
            labels=labels,
            **kwargs,
        )
        similarities = np.asarray(result["similarities"], dtype=np.float32)
        labels = list(result["labels"])

        detections = []
        for row, row_scores in enumerate(similarities):
            ranked_indices = np.argsort(row_scores)[::-1]
            if top_k is not None:
                ranked_indices = ranked_indices[:top_k]

            row_detections = [
                {
                    "keyword": labels[int(idx)],
                    "score": float(row_scores[idx]),
                }
                for idx in ranked_indices
                if float(row_scores[idx]) >= threshold
            ]
            detections.append(row_detections)

        return {
            **result,
            "detections": detections,
            "threshold": float(threshold),
        }

    def format_output(self, raw_output: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        return {
            "detections": raw_output["detections"],
            "predictions": raw_output["predictions"],
            "labels": list(raw_output["labels"]),
            "threshold": raw_output["threshold"],
            "similarities": raw_output["similarities"].tolist()
            if isinstance(raw_output["similarities"], np.ndarray)
            else raw_output["similarities"],
        }
