# -*- coding: utf-8 -*-
"""conftest for pytest"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_image():
    """Provide a simple test image"""
    import numpy as np
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
