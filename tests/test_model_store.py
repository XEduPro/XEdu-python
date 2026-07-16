# -*- coding: utf-8 -*-
"""Test model store"""

import os
import tempfile
from types import SimpleNamespace
import pytest
from XEdu.hub.model_store import ModelStore, get_model_store


def test_cache_dir_resolution():
    """Test cache directory resolution"""
    store = ModelStore()
    assert os.path.isabs(store.cache_dir)
    assert "xedu" in store.cache_dir.lower() or "cache" in store.cache_dir.lower()


def test_get_model_store_singleton():
    """Test singleton pattern"""
    store1 = get_model_store()
    store2 = get_model_store()
    assert store1 is store2


def test_is_package_available():
    """Test package availability check"""
    assert ModelStore._is_package_available("os")  # stdlib
    assert not ModelStore._is_package_available("nonexistent_package_xyz")


def test_get_model_path_falls_back_to_mirror(monkeypatch, tmp_path):
    """Primary source failures should fall back to configured mirror URLs."""
    metadata = SimpleNamespace(
        filename="mock.onnx",
        auto_download=True,
        source_url="https://primary.example/mock.onnx",
        mirror_urls=["https://mirror.example/mock.onnx"],
        checksum=None,
    )
    monkeypatch.setattr("XEdu.hub.model_store.get_model", lambda model_id: metadata)

    calls = []

    def fake_download(self, url, local_path, checksum=None, chunk_size=8192):
        calls.append(url)
        if "primary" in url:
            from XEdu.hub.exceptions import XEduModelDownloadError

            raise XEduModelDownloadError("primary failed")
        with open(local_path, "wb") as f:
            f.write(b"mock")

    monkeypatch.setattr("XEdu.hub.model_store.ModelStore.download", fake_download)

    store = ModelStore(str(tmp_path))
    path = store.get_model_path("mock-model")

    assert os.path.exists(path)
    assert calls == [
        "https://primary.example/mock.onnx",
        "https://mirror.example/mock.onnx",
    ]
