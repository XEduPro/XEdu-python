# XEdu-python

<div align="center">

做有用的AI教育

[文档](https://xedu.readthedocs.io/zh-cn/master/) | [官网](https://www.openinnolab.org.cn/pjedu/xedu/mmedu) | [示例](./XEdu/examples/)

</div>

## 简介

XEdu-python 是一个面向中小学 AI 教学场景的模型推理和工具库，由 OpenXLab 出品。通过简洁的 Python API，学生可以用几行代码调用预训练的深度学习模型，无需深厚的 AI 背景知识。

## 核心特性

- **开箱即用**：20+ 种预训练任务，支持自动模型下载
- **任务丰富**：姿态检测、人脸检测、目标检测、分类、OCR、图像生成、深度估计、embedding 等
- **一体化 LLM 支持**：国产大模型（通义、文心、GLM、Kimi、深度求索）和自建 Gradio 网关
- **教学友好**：内置 Gradio 聊天 UI，支持实时演示和交互

## 安装

```bash
pip install XEdu-python
```

### 可选依赖

- OCR 任务：`pip install XEdu-python[ocr]`
- 音频 embedding：`pip install XEdu-python[audio]`
- 全部依赖：`pip install XEdu-python[all]`

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

### OCR 识别

```python
# 文字识别
ocr = wf(task='ocr')
result, img_output = ocr.inference(data='ocr_image.jpg', img_type='cv2')
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

| 类别 | 任务名 | 描述 |
|-----|-------|------|
| 检测 | `det_body`, `det_body_l` | 人体检测 |
| 检测 | `det_coco`, `det_coco_l` | COCO 目标检测 |
| 检测 | `det_face` | 人脸检测（YuNet） |
| 检测 | `det_hand` | 手部检测 |
| 姿态 | `pose_body17`, `pose_body17_l` | 人体姿态（17 关键点） |
| 姿态 | `pose_body26` | 人体姿态（26 关键点） |
| 姿态 | `pose_face106` | 人脸关键点（106 个） |
| 姿态 | `pose_hand21` | 手部关键点（21 个） |
| 姿态 | `pose_wholebody133` | 全身关键点（133 个） |
| 分类 | `cls_imagenet` | ImageNet 分类 |
| OCR | `ocr` | 文字识别和检测 |
| 生成 | `gen_style` | 风格迁移 |
| 生成 | `gen_color` | 图像着色 |
| 分割 | `segment_anything` | SAM 图像分割 |
| 深度估计 | `depth_anything` | 深度估计 |
| 嵌入 | `embedding_image` | 图像嵌入 |
| 嵌入 | `embedding_text` | 文本嵌入 |
| 嵌入 | `embedding_audio` | 音频嵌入 |
| NLP | `nlp_qa` | 问答任务 |
| 感知 | `drive_perception` | 自动驾驶感知 |

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

## 常见问题

**Q: 模型下载很慢？**
A: 模型默认从 openinnolab.org.cn 国内节点下载。如果网络仍然受限，建议预先在有网络的环境下运行一次，模型会缓存本地。

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
  url={https://github.com/OpenXLab-Edu/XEdu-python}
}
```

## 贡献

欢迎提交 Issue 和 PR！

## 联系方式

- 官网：https://www.openinnolab.org.cn/pjedu/xedu/mmedu
- 文档：https://xedu.readthedocs.io
- 邮件：wangbolun@pjlab.org.cn
