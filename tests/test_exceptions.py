# -*- coding: utf-8 -*-
"""Test exception system"""

import pytest
from XEdu.hub.exceptions import (
    XEduError,
    XEduTaskNotFoundError,
    XEduModelNotFoundError,
    XEduDependencyError,
    make_dependency_error,
    make_model_not_found_error,
)


def test_exception_hierarchy():
    """Test exception class hierarchy"""
    assert issubclass(XEduTaskNotFoundError, XEduError)
    assert issubclass(XEduModelNotFoundError, XEduError)
    assert issubclass(XEduDependencyError, XEduError)


def test_make_dependency_error():
    """Test dependency error generation"""
    err = make_dependency_error("ocr", "rapidocr_onnxruntime")
    assert isinstance(err, XEduDependencyError)
    assert "rapidocr_onnxruntime" in str(err)
    assert "pip install" in str(err)


def test_make_model_not_found_error():
    """Test model not found error generation"""
    err = make_model_not_found_error("pose_body17", "/path/to/model.onnx")
    assert isinstance(err, XEduModelNotFoundError)
    assert "/path/to/model.onnx" in str(err)
