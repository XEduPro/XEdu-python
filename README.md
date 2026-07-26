# XEdu-python

<div align="center">

做有用的AI教育

[文档](https://xedu.readthedocs.io/zh-cn/master/) | [官网](https://www.openinnolab.org.cn/pjedu/xedu/mmedu) | [快速上手 Notebook](http://8.145.44.54:3000/admin/XEdu-python/src/branch/2.1/XEdu/examples/getting_started.ipynb)

</div>

## 简介

XEdu-python 是一个面向中小学 AI 教学场景的模型推理和工具库，由 OpenXLab 出品。通过简洁的 Python API，学生可以用几行代码调用预训练的深度学习模型，无需深厚的 AI 背景知识。

## 核心特性

- **开箱即用**：20+ 种预训练任务，支持自动模型下载
- **任务丰富**：姿态检测、人脸检测、目标检测、分类、OCR、图像生成、深度估计、embedding、音频/文本/图文匹配等
- **一体化 LLM 支持**：国产大模型（通义、文心、GLM、Kimi、深度求索）和自建 Gradio 网关
- **教学友好**：内置 Gradio 聊天 UI，支持实时演示和交互

## 安装与快速上手

`2.1.0` 支持 Python 3.8 及以上版本。它暂不发布到官方 PyPI，请使用 XEdu
Gitea 软件源安装固定版本：

```bash
python -m pip install --upgrade \
  --trusted-host 8.145.44.54 \
  --index-url http://8.145.44.54:3000/api/packages/admin/pypi/simple \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "XEdu-python[all]==2.1.0"
```

安装后可直接下载并打开 [getting_started.ipynb](http://8.145.44.54:3000/admin/XEdu-python/raw/branch/2.1/XEdu/examples/getting_started.ipynb)。
如果希望打开已安装包内的同名文件，请运行：

```bash
python -c "from pathlib import Path; import XEdu.examples; print(Path(XEdu.examples.__file__).with_name('getting_started.ipynb'))"
```

### 按需安装

如果不需要全部功能，复制上面的 Gitea 安装命令，仅将 `[all]` 替换为：

- `[llm]`：LLM 与 Gradio 聊天界面
- `[ocr]`：OCR 任务
- `[audio]`：音频 embedding / 分类

## 快速开始

### 目标检测

```python
from XEdu.hub import Workflow as wf
import cv2

# 实例化人体检测模型
det = wf(task='det_body')

# 推理
img = cv2.imread('image.jpg')
result, img_output = det.inference(data=img, img_type='cv2', show=True)
```

### 姿态估计

```python
# 人体姿态识别
pose = wf(task='pose_body17')
keypoints, img_output = pose.inference(data=img, img_type='cv2', bbox=result[0])
```

### 人脸关键点

`pose_face` 是人脸关键点任务，2.1 默认使用更新后的轻量 MobileNet 106 点模型；需要更高精度时可以切换到 PIPNet/WFLW-98。

```python
face_pose = wf(task='pose_face')
keypoints, img_output = face_pose.inference(data=img, img_type='cv2')

face_pose_high = wf(
    task='pose_face',
    model_id='pose_face_landmark-pipnet98-wflw',
)
keypoints98 = face_pose_high.inference(data=img)
```

### OCR 识别

```python
# 文字识别
ocr = wf(task='ocr')
result, img_output = ocr.inference(data='ocr_image.jpg', img_type='cv2')
```

### 离线文本分类

`cls_text` 是基于本地文本 embedding 的原型分类任务，不调用 `XEdu.LLM`，也不需要 API key。中文文本可以输入，但当前使用的是 CLIP 文本侧模型，不是中文专用 embedding；正式教学样例建议先用自己的类别原型做相似度边界检查。

```python
from XEdu.hub import Workflow as wf

clf = wf(task='cls_text')
result = clf.inference(
    data='这节课我们学习图像识别',
    prototypes={
        '人工智能': ['图像识别', '机器学习'],
        '体育': ['篮球训练', '跑步比赛'],
    },
)
print(result['label'])
```

### 图文匹配

`match_image_text` 组合本地图像 embedding 与文本 embedding，返回候选文本的相似度排序；它不是 VQA，不能直接回答图像问题。

```python
matcher = wf(task='match_image_text')
result = matcher.inference(
    data='image.jpg',
    texts=['一只猫在桌上', '学生在教室里上课', '一辆车停在路边'],
)
print(result['best_text'], result['best_score'])
```

### 大语言模型

```python
from XEdu.LLM import Client

# 使用通义千问
client = Client(provider="qwen", api_key="your-api-key")
response = client.inference("你好，请介绍一下自己", stream=False)
print(response)

# 启动 Gradio 聊天界面
client.run(host='0.0.0.0', port=7860)
```

## 支持的任务

`Workflow.support_task()` 在 2.1.0 中共返回 38 个任务入口：

| 类别 | 任务名 | 描述 |
|-----|-------|------|
| 检测 | `det_body`, `det_body_l` | 人体检测 |
| 检测 | `det_coco`, `det_coco_l` | COCO 目标检测 |
| 检测 | `det_hand`, `det_face` | 手部与人脸检测 |
| 姿态与关键点 | `pose_body17`, `pose_body17_l`, `pose_body26` | 人体关键点 |
| 姿态与关键点 | `pose_wholebody133`, `pose_hand21` | 全身与手部关键点 |
| 姿态与关键点 | `pose_face`, `pose_face106` | 人脸关键点；`pose_face106` 需显式提供本地 checkpoint |
| 图像理解 | `cls_imagenet`, `ocr` | ImageNet 分类与 OCR |
| 图像理解 | `segment_anything`, `depth_anything` | 图像分割与深度估计 |
| 图像理解 | `drive_perception` | 道路目标、车道线和可行驶区域感知 |
| 图像生成与处理 | `gen_color`, `gen_style` | 图像着色与风格迁移统一入口 |
| 图像生成与处理 | `gen_style_mosaic`, `gen_style_candy` | 内置风格迁移 |
| 图像生成与处理 | `gen_style_rain-princess`, `gen_style_udnie` | 内置风格迁移 |
| 图像生成与处理 | `gen_style_pointilism`, `gen_style_custom` | 点彩与自定义风格迁移 |
| 向量与图文匹配 | `embedding_image`, `embedding_text` | 图像与文本 embedding |
| 向量与图文匹配 | `match_image_text` | 图文匹配（CLIP 相似度，不是 VQA） |
| 文本 | `nlp_qa` | 问答任务 |
| 文本 | `cls_text` | 离线原型文本分类（本地 embedding，不调用 LLM API） |
| 音频 | `embedding_audio`, `cls_audio` | 音频 embedding 与原型分类 |
| 音频 | `det_audio_keyword` | 原型关键词或声音事件检测 |
| 自定义扩展 | `mmedu`, `basenn`, `baseml`, `custom` | 教学平台及自定义模型兼容入口 |

## 文件结构

```
XEdu/
├── hub/              # 核心推理模块（Workflow 主类）
│   ├── workflow.py   # 统一任务接口
│   ├── models/       # 各模型的特殊加载逻辑
│   ├── tokenizer/    # NLP tokenizer
│   └── BaseDT/       # 数据集工具
├── LLM/              # 大语言模型接口
│   ├── client.py     # Client 主类
│   └── llms/         # 各个 LLM provider 实现
└── utils/            # 工具函数（embedding 相似度等）
```

## 环境变量配置

### 模型缓存目录

默认下载到 `~/.cache/XEdu/hub`，可通过环境变量自定义：

```bash
# Linux/macOS
export XEDU_HOME=~/my_xedu_models

# Windows
set XEDU_HOME=C:\my_xedu_models
```

### 模型下载源

内置 ONNX 模型默认从 [ModelScope](https://www.modelscope.cn/models/wht0926/xedu-hub-models) 下载，并保留 OpenXLab/OpenInnoLab 或 GitHub 原始地址作为备用源。下载后会做 SHA256 校验；主源不可用时会自动尝试备用源。

## 常见问题

**Q: 模型下载很慢？**
A: 模型默认从 ModelScope 下载，OpenXLab/OpenInnoLab 或 GitHub 作为备用源。如果网络仍然受限，建议预先在有网络的环境下运行一次，模型会缓存本地。

**Q: 如何使用自定义 ONNX 模型？**
A: 指定 `checkpoint` 参数：
```python
wf_custom = wf(task='custom', checkpoint='path/to/your_model.onnx')
result = wf_custom.inference(data=img)
```

**Q: 支持 GPU 加速吗？**
A: 支持。onnxruntime 会自动使用 GPU（如有 CUDA 环境）。

**Q: 可以在 CPU 上运行吗？**
A: 可以。所有模型都支持 CPU 推理。

## 许可证

MIT License

## 引用

如果你在学术工作中使用 XEdu，请引用：

```bibtex
@software{xedu2024,
  title={XEdu-python: AI Education Toolkit for K-12},
  author={OpenXLab-Edu},
  year={2024},
  url={https://github.com/XEduPro/XEdu-python}
}
```

## 贡献

欢迎提交 Issue 和 PR！

## 联系方式

- 官网：https://www.openinnolab.org.cn/pjedu/xedu/mmedu
- 文档：https://xedu.readthedocs.io
- 邮件：wangbolun@pjlab.org.cn
