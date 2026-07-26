# Changelog

## 3.0.0

### Compatibility and installation

- Restored Python 3.8 support with version-specific NumPy and Matplotlib ranges.
- Added `gradio>=4,<5` to the optional `llm` and `all` extras while keeping the default install lightweight.
- Added a packaged getting-started Notebook and three local teaching assets. The Notebook can install from a local wheel, a local source checkout, or the maintained GitHub `2.1` branch.

### Models and tasks

- Replaced the default `det_hand` model with the tested pure-ONNX palm detector while retaining the previous model as an explicit compatibility option.
- Restored `pose_face` as the public face-landmark task name. `pose_face_landmark` remains accepted as a compatibility alias.
- Updated label drawing to use Pillow's supported `textbbox()` API.

### Breaking Changes

- Removed runtime auto-install for optional dependencies.
  Previously, `XEdu/LLM/client.py`, `XEdu/LLM/llms/gradioapi.py`, and the holiday greeting path in `XEdu/__init__.py` could run `pip install` automatically when `gradio`, `gradio-client`, or `holidays` was missing. Now missing optional dependencies raise `ImportError` with an explicit manual install instruction. Affected users are environments that relied on first-use auto-install instead of preinstalling optional packages.

- Stopped silent incorrect auto-download for `pose_face106`.
  Previously, `Workflow(task="pose_face106")` without a local checkpoint could download the `pose_wholebody133` weights and use them as if they were face106 weights, producing wrong results without a clear error. Now the default path raises `RuntimeError` and requires `checkpoint="/path/to/face106.onnx"` until a verified face106 asset URL is available. Affected users are anyone relying on the old default download path; their previous results were already based on the wrong model.

- Preserved argument types in `RepoModel._custom_infer`.
  Previously, custom inference kwargs were interpolated into an `eval()` string, so values such as `threshold=0.5` reached user `data_process.py` code as the string `"0.5"`. Now kwargs are passed directly with `**kwargs`, preserving original Python types. Affected users are custom repo models whose `data_process.py` depended on the old all-strings behavior.

### Packaging

- Fixed recursive package discovery.
  Previously, packaging used `packages=["XEdu"]`, which omitted subpackages such as `XEdu.LLM`, `XEdu.utils`, `XEdu.hub.models`, and `XEdu.hub.BaseDT` from installed distributions. Now setuptools discovers `XEdu*` packages recursively. This is a compatibility fix for normal `pip install` users and has no intended negative behavior change.
