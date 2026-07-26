# -*- coding: utf-8 -*-
"""
Packaging integrity tests.

These guard against a real regression found during manual testing: an
earlier version of pyproject.toml declared `packages = ["XEdu"]` explicitly,
which silently dropped XEdu.LLM, XEdu.utils, XEdu.hub.models and
XEdu.hub.BaseDT from the built wheel (setuptools does not auto-discover
sub-packages when `packages` is given as an explicit list).

Building a wheel is slow, so these tests are marked and skipped unless
explicitly requested, but they are the only thing that would have caught
that regression automatically.
"""

import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.8-3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_SUBPACKAGES = [
    "XEdu/__init__.py",
    "XEdu/hub/__init__.py",
    "XEdu/hub/models/__init__.py",
    "XEdu/hub/BaseDT/__init__.py",
    "XEdu/hub/tokenizer/__init__.py",
    "XEdu/LLM/__init__.py",
    "XEdu/LLM/llms/__init__.py",
    "XEdu/utils/__init__.py",
]

EXPECTED_EXAMPLE_ASSETS = [
    "XEdu/examples/getting_started.ipynb",
    "XEdu/examples/assets/xedu-vision-scene.png",
    "XEdu/examples/assets/xedu-road-scene.png",
    "XEdu/examples/assets/xedu-ocr-poster.png",
]


def load_pyproject():
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


@pytest.mark.slow
def test_wheel_contains_all_subpackages(tmp_path):
    """Build a wheel and assert every known sub-package is present in it.

    This directly reproduces the manual check that caught the packaging
    bug: `packages = ["XEdu"]` in pyproject.toml silently drops
    XEdu.LLM / XEdu.utils / XEdu.hub.models / XEdu.hub.BaseDT.
    """
    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"wheel build failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, "no wheel produced"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())

    missing = [f for f in EXPECTED_SUBPACKAGES + EXPECTED_EXAMPLE_ASSETS if f not in names]
    assert not missing, f"wheel is missing expected files: {missing}"


def test_setup_cfg_uses_automatic_package_discovery():
    """Regression guard without needing a full wheel build.

    Asserts pyproject.toml does NOT hardcode `packages = ["XEdu"]`, which is
    what caused sub-packages to be silently dropped. Automatic discovery via
    [tool.setuptools.packages.find] is required instead.
    """
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["XEdu"]' not in pyproject
    assert "[tool.setuptools.packages.find]" in pyproject


def test_project_supports_python_38_with_compatible_scientific_dependencies():
    project = load_pyproject()["project"]

    assert project["requires-python"] == ">=3.8"
    assert "Programming Language :: Python :: 3.8" in project["classifiers"]
    assert "numpy>=1.24,<1.25; python_version < '3.9'" in project["dependencies"]
    assert "numpy>=1.26,<2; python_version >= '3.9'" in project["dependencies"]
    assert "matplotlib>=3.7,<3.8; python_version < '3.9'" in project["dependencies"]
    assert "matplotlib>=3.8; python_version >= '3.9'" in project["dependencies"]


def test_gradio_is_available_only_through_optional_full_install():
    project = load_pyproject()["project"]
    extras = project["optional-dependencies"]

    assert "gradio>=4,<5" in extras["llm"]
    assert "gradio>=4,<5" in extras["all"]
    assert all(not dependency.startswith("gradio") for dependency in project["dependencies"])


def test_package_data_includes_getting_started_notebook_and_assets():
    package_data = load_pyproject()["tool"]["setuptools"]["package-data"]

    assert "XEdu.examples" in package_data
    assert "getting_started.ipynb" in package_data["XEdu.examples"]
    assert "assets/*.png" in package_data["XEdu.examples"]
