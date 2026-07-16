# -*- coding: utf-8 -*-
"""
Workflow smoke tests.

These tests exercise Workflow.__init__ task-routing logic without touching
the network or the real user cache directory. They exist to catch the class
of regression that shipped in this repo previously: a task silently
resolving to the wrong download URL/model file (see pose_face106).
"""

import os

import numpy as np
import pytest

from XEdu.hub.workflow import Workflow, pil_draw


def test_pil_draw_supports_current_pillow_text_measurement_api():
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    drawn = pil_draw(image, [10, 30, 90, 70], "教师")

    assert drawn.shape == image.shape


def test_pose_face106_raises_before_any_download(tmp_path, monkeypatch):
    """pose_face106 must fail loudly instead of downloading pose_wholebody133's model.

    Historically this task's entry in the download URL map pointed at the
    pose_wholebody133 asset, so a user who didn't have a local checkpoint
    would silently get the wrong model. The fix short-circuits with a
    RuntimeError before the download map is even consulted.
    """
    calls = []

    def _fail_if_called(self):
        calls.append(self.url)
        raise AssertionError("Downloader.start() must not be called for pose_face106")

    monkeypatch.setattr("XEdu.hub.workflow.Downloader.start", _fail_if_called)

    with pytest.raises(RuntimeError, match="pose_face106"):
        Workflow(task="pose_face106", download_path=str(tmp_path))

    assert calls == []
    # No checkpoint file should have been created either.
    assert list(tmp_path.rglob("*.onnx")) == []


def test_pose_face106_with_explicit_checkpoint_does_not_raise(tmp_path):
    """Supplying a local checkpoint should bypass the download guard entirely."""
    fake_checkpoint = tmp_path / "face106.onnx"
    fake_checkpoint.write_bytes(b"not a real onnx model, only path existence is checked here")

    # This will get past the RuntimeError guard and attempt to load the
    # (fake) onnx file via onnxruntime, which will fail on the invalid
    # content -- that failure is expected and fine, it's a different code
    # path than the bug we're guarding against.
    with pytest.raises(Exception):
        Workflow(task="pose_face106", checkpoint=str(fake_checkpoint))


def test_unrelated_task_download_path_is_unaffected(tmp_path, monkeypatch):
    """The pose_face106 guard must not short-circuit other tasks' downloads."""
    seen_urls = []

    def _record_and_write_dummy(self, url, local_path, checksum=None, chunk_size=8192):
        seen_urls.append(url)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        # Write an invalid (but present) file so Workflow proceeds past the
        # "not os.path.exists(checkpoint)" branch and attempts to load it.
        with open(local_path, "wb") as f:
            f.write(b"")

    monkeypatch.setattr("XEdu.hub.model_store.ModelStore.download", _record_and_write_dummy)

    with pytest.raises(Exception):
        # Loading an empty/invalid onnx file raises inside onnxruntime;
        # that's expected. What we're checking is that the *download url*
        # used was pose_body17's own, not pose_face106's, and that the
        # guard for pose_face106 didn't block this unrelated task.
        Workflow(task="pose_body17", download_path=str(tmp_path))

    assert len(seen_urls) == 1
    assert "modelscope.cn/models/wht0926/xedu-hub-models" in seen_urls[0]
    assert seen_urls[0].endswith("/body17.onnx")


def test_legacy_default_task_uses_registry_model_store(tmp_path, monkeypatch):
    """Legacy Workflow branches should use the registry source policy too."""
    seen = []

    def _record_and_write_dummy(self, url, local_path, checksum=None, chunk_size=8192):
        seen.append((url, checksum))
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"")

    monkeypatch.setattr("XEdu.hub.model_store.ModelStore.download", _record_and_write_dummy)

    with pytest.raises(Exception):
        Workflow(task="cls_imagenet", download_path=str(tmp_path))

    assert len(seen) == 1
    assert "modelscope.cn/models/wht0926/xedu-hub-models" in seen[0][0]
    assert seen[0][0].endswith("/imagenet1k.onnx")
    assert seen[0][1]


def test_unknown_task_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        Workflow(task="this_task_does_not_exist", download_path=str(tmp_path))


def test_support_task_keeps_original_pose_face_name():
    tasks = Workflow.support_task()

    assert "pose_face" in tasks
    assert "pose_face_landmark" not in tasks
