# -*- coding: utf-8 -*-
"""Test model store"""

import os
import tempfile
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
