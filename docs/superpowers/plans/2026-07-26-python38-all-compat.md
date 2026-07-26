# Python 3.8 and Full-Install Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make XEdu-python 2.1 installable on Python 3.8 and include the Python 3.8-compatible Gradio UI in the `all` extra.

**Architecture:** Packaging metadata uses Python-version environment markers so Python 3.8 receives the last compatible NumPy and Matplotlib lines while newer Python versions preserve current requirements. The Notebook locates bundled assets through the imported package path, avoiding the Python 3.9-only `importlib.resources.files` API.

**Tech Stack:** Python packaging (`pyproject.toml`), pytest, Jupyter Notebook JSON, uv dependency resolver, setuptools build.

## Global Constraints

- Support Python `>=3.8` on the 2.1 branch.
- Keep Gradio out of default dependencies and constrain it to `gradio>=4,<5` in `llm` and `all`.
- Preserve Python 3.9+ NumPy `>=1.26,<2` and Matplotlib `>=3.8` requirements.
- Do not change inference behavior or model assets.

---

### Task 1: Lock the packaging contract

**Files:**
- Modify: `tests/test_packaging.py:16-92`
- Modify: `pyproject.toml:10-55`

**Interfaces:**
- Consumes: PEP 621 project metadata and PEP 508 environment markers.
- Produces: Python 3.8-compatible dependency metadata and `llm`/`all` extras.

- [ ] **Step 1: Write failing metadata tests**

Add a Python 3.8-compatible TOML import and tests that require:

```python
try:
    import tomllib
except ImportError:
    import tomli as tomllib

def load_pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)

def test_project_supports_python_38_with_compatible_scientific_dependencies():
    project = load_pyproject()["project"]
    assert project["requires-python"] == ">=3.8"
    assert "numpy>=1.24,<1.25; python_version < '3.9'" in project["dependencies"]
    assert "numpy>=1.26,<2; python_version >= '3.9'" in project["dependencies"]
    assert "matplotlib>=3.7,<3.8; python_version < '3.9'" in project["dependencies"]
    assert "matplotlib>=3.8; python_version >= '3.9'" in project["dependencies"]

def test_gradio_is_available_only_through_optional_full_install():
    project = load_pyproject()["project"]
    extras = project["optional-dependencies"]
    assert "gradio>=4,<5" in extras["llm"]
    assert "gradio>=4,<5" in extras["all"]
    assert all(not item.startswith("gradio") for item in project["dependencies"])
```

- [ ] **Step 2: Verify the tests fail for the intended reasons**

Run: `pytest tests/test_packaging.py -m 'not slow' -v`

Expected: failures show `requires-python` is `>=3.9`, Python 3.8 dependency markers are absent, and `llm` is absent.

- [ ] **Step 3: Implement the packaging metadata**

Set `requires-python = ">=3.8"`, add the Python 3.8 classifier, split NumPy and Matplotlib requirements with the exact markers in Step 1, add `llm = ["gradio>=4,<5"]`, include Gradio in `all`, and set Black's target version to `py38`.

- [ ] **Step 4: Verify packaging tests pass**

Run: `pytest tests/test_packaging.py -m 'not slow' -v`

Expected: all non-slow packaging tests pass.

### Task 2: Make the Notebook resource lookup Python 3.8-compatible

**Files:**
- Modify: `tests/test_getting_started_notebook.py:18-28`
- Modify: `XEdu/examples/getting_started.ipynb` asset setup cell

**Interfaces:**
- Consumes: `XEdu.examples.__file__` from the installed examples package.
- Produces: `ASSET_DIR: pathlib.Path` on Python 3.8+.

- [ ] **Step 1: Update the Notebook regression test first**

Require the source to contain `Path(XEdu.examples.__file__).resolve().parent / 'assets'`, require `import XEdu.examples`, and reject `from importlib.resources import files`.

- [ ] **Step 2: Verify the focused test fails**

Run: `pytest tests/test_getting_started_notebook.py::test_getting_started_notebook_uses_packaged_example_assets -v`

Expected: failure because the Notebook still calls `files('XEdu.examples')`.

- [ ] **Step 3: Change the Notebook asset cell**

Replace the Python 3.9-only import and assignment with:

```python
from pathlib import Path
import numpy as np
import soundfile as sf
import XEdu.examples

ASSET_DIR = Path(XEdu.examples.__file__).resolve().parent / 'assets'
```

- [ ] **Step 4: Verify all Notebook tests pass**

Run: `pytest tests/test_getting_started_notebook.py -v`

Expected: all Notebook regression tests pass.

### Task 3: Align installation documentation

**Files:**
- Modify: `README.md:22-33`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: extras defined in `pyproject.toml`.
- Produces: installation commands that match package metadata.

- [ ] **Step 1: Add README assertions to the packaging test**

Require `Python 3.8` and both `pip install XEdu-python[llm]` and `pip install XEdu-python[all]` in README.

- [ ] **Step 2: Verify the README assertions fail**

Run: `pytest tests/test_packaging.py -m 'not slow' -v`

Expected: failure because Python 3.8 and the `llm` install command are not documented.

- [ ] **Step 3: Document supported installation modes**

State that the 2.1 source branch supports Python 3.8+, add the LLM/Gradio extra command, and describe `all` as OCR + audio + Gradio.

- [ ] **Step 4: Verify focused tests pass**

Run: `pytest tests/test_packaging.py tests/test_getting_started_notebook.py -m 'not slow' -v`

Expected: all focused tests pass.

### Task 4: Verify and deliver the source PR

**Files:**
- Verify: all modified source and documentation files.

**Interfaces:**
- Consumes: completed source changes.
- Produces: tested wheel metadata and a PR targeting `XEduPro/XEdu-python:2.1`.

- [ ] **Step 1: Resolve dependencies for both Python versions**

Run `uv pip compile --python-version 3.8 pyproject.toml --extra all --quiet` and repeat with `--python-version 3.12`.

Expected: both commands exit zero; Python 3.8 selects NumPy 1.24.x, Matplotlib 3.7.x, and Gradio 4.x.

- [ ] **Step 2: Run syntax and test verification**

Run `python3.8 -m compileall -q XEdu`, `pytest -m 'not slow'`, and `pytest tests/test_packaging.py -m slow -v`.

Expected: all commands exit zero.

- [ ] **Step 3: Build and inspect the wheel**

Build with `python -m build --wheel`; inspect `METADATA` for `Requires-Python: >=3.8`, version-marked requirements, and the `all` Gradio extra; inspect the archive for the Notebook and assets.

- [ ] **Step 4: Commit, push, and open the PR**

Use a Lore-format commit documenting compatibility constraints and verification. Push `codex/py38-all-compat` to `XEduPro/XEdu-python` and create a PR with base `2.1`.
