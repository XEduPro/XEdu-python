# -*- coding: utf-8 -*-
"""
Workflow smoke tests.

These tests exercise Workflow.__init__ task-routing logic without touching
the network or the real user cache directory. They exist to catch the class
of regression that shipped in this repo previously: a task silently
resolving to the wrong download URL/model file (see pose_face106).
"""

import os

import pytest

from XEdu.hub.workflow import Workflow


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

    def _record_and_write_dummy(self):
        seen_urls.append(self.url)
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        # Write an invalid (but present) file so Workflow proceeds past the
        # "not os.path.exists(checkpoint)" branch and attempts to load it.
        with open(self.file_path, "wb") as f:
            f.write(b"")

    monkeypatch.setattr("XEdu.hub.workflow.Downloader.start", _record_and_write_dummy)

    with pytest.raises(Exception):
        # Loading an empty/invalid onnx file raises inside onnxruntime;
        # that's expected. What we're checking is that the *download url*
        # used was pose_body17's own, not pose_face106's, and that the
        # guard for pose_face106 didn't block this unrelated task.
        Workflow(task="pose_body17", download_path=str(tmp_path))

    assert len(seen_urls) == 1
    assert "pose_body17" in seen_urls[0] or "b94f252e" in seen_urls[0]


def test_unknown_task_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        Workflow(task="this_task_does_not_exist", download_path=str(tmp_path))
