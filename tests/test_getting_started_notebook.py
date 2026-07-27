import json
from pathlib import Path

import XEdu.examples


def test_getting_started_notebook_uses_match_image_text_return_schema():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "result['matches'][0]['text']" in source
    assert "result['best_text']" not in source


def test_getting_started_notebook_uses_packaged_example_assets():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Path(XEdu.examples.__file__).resolve().parent / 'assets'" in source
    assert "import XEdu.examples" in source
    assert "from importlib.resources import files" not in source
    assert "outputs' / 'imagegen" not in source
    assert "find_project_root" not in source
    assert "import sys" in source


def test_examples_package_imports_for_packaged_resources():
    assert XEdu.examples.__file__


def test_hand_and_face_landmark_example_runs_direct_hand_detection():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "hand_boxes, hand_image = hand.inference" in source
    assert "face_boxes, face_image = face.inference" in source


def test_hand_example_uses_wholebody_pose_for_full_scene_asset():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "whole = wf(task='pose_wholebody133')" in source
    assert "left_hand21 = whole_points[91:112]" in source
    assert "right_hand21 = whole_points[112:133]" in source
    assert "Palm ONNX" in source


def test_getting_started_notebook_runs_ocr_and_audio_examples():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "ocr = wf(task='ocr')" in source
    assert "audio_embedder = wf(task='embedding_audio')" in source
    assert "import soundfile as sf" in source


def test_getting_started_notebook_explains_how_to_read_comparison_results():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "行和列的顺序" in source
    assert "候选文本及其分数" in source
    assert "分数不是正确率，也不是百分比" in source
    assert "这只能验证流程可运行" in source


def test_getting_started_notebook_includes_the_full_feature_install_command():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "f'{wheel_path}[all]'" in source
    assert "XEdu-python[all]==2.1.0" in source
    assert "--upgrade" in source
    assert "OCR、音频和 Gradio" in source
    assert "http://8.145.44.54:3000/api/packages/admin/pypi/simple" in source
    assert "https://pypi.tuna.tsinghua.edu.cn/simple" in source
    assert "--trusted-host" in source
    assert "git+https://github.com/XEduPro/XEdu-python.git@2.1" not in source
    assert "folder / 'pyproject.toml'" in source


def test_first_code_cell_can_install_the_local_wheel_with_current_python():
    notebook_path = Path(__file__).parents[1] / "XEdu" / "examples" / "getting_started.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    first_code = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "".join(first_code["source"])

    assert "subprocess.check_call" in source
    assert "sys.executable" in source
    assert "wheel_path" in source
    assert "[all]" in source


def test_readme_documents_the_gitea_install_and_notebook_entrypoint():
    readme_path = Path(__file__).parents[1] / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    notebook_download = (
        "http://8.145.44.54:3000/admin/XEdu-python/releases/download/2.1.0/"
        "getting_started.ipynb.zip"
    )

    assert "XEdu-python[all]==2.1.0" in readme
    assert "http://8.145.44.54:3000/api/packages/admin/pypi/simple" in readme
    assert notebook_download in readme
    assert "Path(XEdu.examples.__file__).with_name('getting_started.ipynb')" in readme
    assert "git+https://github.com/XEduPro/XEdu-python.git@2.1" not in readme
