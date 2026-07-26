import json
from pathlib import Path

import XEdu.examples


NOTEBOOK_PATH = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"


def notebook_source():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_examples_package_imports_for_packaged_resources():
    assert XEdu.examples.__file__


def test_getting_started_notebook_uses_packaged_example_assets():
    source = notebook_source()

    assert "Path(XEdu.examples.__file__).resolve().parent / 'assets'" in source
    assert "import XEdu.examples" in source
    assert "from importlib.resources import files" not in source


def test_getting_started_notebook_covers_full_teaching_flow():
    source = notebook_source()

    assert "result['matches'][0]['text']" in source
    assert "hand_boxes, hand_image = hand.inference" in source
    assert "whole = wf(task='pose_wholebody133')" in source
    assert "ocr = wf(task='ocr')" in source
    assert "audio_embedder = wf(task='embedding_audio')" in source
    assert "import soundfile as sf" in source


def test_getting_started_notebook_has_current_install_fallbacks():
    source = notebook_source()

    assert "glob('xedu_python-*.whl')" in source
    assert "folder / 'pyproject.toml'" in source
    assert "git+https://github.com/XEduPro/XEdu-python.git@2.1" in source
    assert "[all]" in source
    assert "--upgrade" in source


def test_first_code_cell_installs_with_the_current_python():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(first_code["source"])

    assert "subprocess.check_call" in source
    assert "sys.executable" in source
