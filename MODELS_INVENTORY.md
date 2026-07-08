# XEdu 模型清单

本文档梳理 XEdu 项目中所有已支持的模型及其配置。

## 检测类模型 (Detection)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| det_body | bodydetect.onnx | openinnolab | image | boxes, scores | — | ✓ 支持 | 人体检测，基于 MMDetection |
| det_body_l | bodydetect_l.onnx | openinnolab | image | boxes, scores | — | ✓ 支持 | 人体检测大模型 |
| det_coco | cocodetect.onnx | openinnolab | image | boxes, scores, classes | — | ✓ 支持 | COCO 目标检测（80类），基于 MMDetection |
| det_coco_l | cocodetect_l.onnx | openinnolab | image | boxes, scores, classes | — | ✓ 支持 | COCO 检测大模型 |
| det_hand | handdetect.onnx | openinnolab | image | boxes, scores | — | ✓ 支持 | 手部检测，基于 MMDetection |
| det_face | face_detection_yunet_2023mar.onnx | opencv_zoo (GitHub) | image | boxes, scores | — | ✓ 支持 | 人脸检测，使用 YuNet 2023 模型，支持 Cascade 后兼容 |

## 姿态关键点类模型 (Pose Estimation)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| pose_body17 | body17.onnx | openinnolab | image, bbox | keypoints, scores | — | ✓ 支持 | 人体 17 点关键点，轻量版 |
| pose_body17_l | body17_l.onnx | openinnolab | image, bbox | keypoints, scores | — | ✓ 支持 | 人体 17 点关键点，大模型 |
| pose_body26 | body26.onnx | openinnolab | image, bbox | keypoints, scores | — | ✓ 支持 | 人体 26 点关键点 |
| pose_hand21 | hand21.onnx | openinnolab | image, bbox | keypoints, scores | — | ✓ 支持 | 手部 21 点关键点 |
| pose_face106 | face106.onnx | 缺失 | image, bbox | keypoints, scores | — | ⚠️ 不支持自动下载 | 人脸 106 点关键点。BUG：原代码复用 wholebody133 URL，已修复为显式报错 |
| pose_wholebody133 | whole133.onnx | openinnolab | image, bbox | keypoints, scores | — | ✓ 支持 | 全身 133 点关键点（人体+人脸+手部） |

## 分类类模型 (Classification)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| cls_imagenet | imagenet1k.onnx | openinnolab | image | label, score, top-k | — | ✓ 支持 | ImageNet 1000 类分类 |

## 生成类模型 (Generation)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| gen_style | gen_style_mosaic.onnx (default) | openinnolab | image | styled_image | — | ✓ 支持 | 风格迁移，5 种风格可选 |
| gen_style_mosaic | gen_style_mosaic.onnx | openinnolab | image | styled_image | — | ✓ 支持 | 马赛克风格 |
| gen_style_candy | gen_style_candy.onnx | openinnolab | image | styled_image | — | ✓ 支持 | 糖果风格 |
| gen_style_rain-princess | gen_style_rain-princess.onnx | openinnolab | image | styled_image | — | ✓ 支持 | 雨夜公主风格 |
| gen_style_udnie | gen_style_udnie.onnx | openinnolab | image | styled_image | — | ✓ 支持 | UDNIE 风格 |
| gen_style_pointilism | gen_style_pointilism.onnx | openinnolab | image | styled_image | — | ✓ 支持 | 点彩画风格 |
| gen_color | gen_color.onnx | openinnolab | gray_image | color_image | — | ✓ 支持 | 图像着色（灰度→彩色） |

## 分割类模型 (Segmentation)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| segment_anything | seg_sam_encoder.onnx, seg_sam_decoder.onnx | openinnolab | image | segmentation_mask | — | ✓ 支持 | SAM 通用分割，支持 prompt 引导 |
| depth_anything | depth_anything.onnx | openinnolab | image | depth_map | — | ✓ 支持 | 深度估计，支持输出校验 |

## NLP 类模型 (NLP)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| nlp_qa | nlp_qa.onnx | openinnolab | context, question | answer | — | ✓ 支持 | 问答系统，基于 SQuAD 数据集，BERT tokenizer |

## Embedding 类模型 (Embeddings)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| embedding_image | embedding_image.onnx | openinnolab | image | embedding (512d) | — | ✓ 支持 | CLIP 图像嵌入 |
| embedding_text | embedding_text.onnx | openinnolab | text | embedding (512d) | — | ✓ 支持 | CLIP 文本嵌入 |
| embedding_audio | embedding_audio.onnx | openinnolab | audio | embedding (1024d) | soundfile | ⚠️ 可选 | CLAP 音频嵌入，需要 soundfile 依赖 |

## 场景感知类模型 (Perception)

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| drive_perception | drive_perception.onnx | openinnolab | image | objects, lanes | — | ✓ 支持 | 自动驾驶场景感知 |

## 其他支持模式

| Task | Model File | Source | Input | Output | Optional Deps | Status | Notes |
|------|-----------|--------|-------|--------|---------------|--------|-------|
| ocr | — | rapidocr | image | text, boxes | rapidocr_onnxruntime | ⚠️ 可选 | 文字识别 + 检测，需要 rapidocr_onnxruntime |
| mmedu | 用户自定义 | 用户本地 | 多种 | 多种 | — | ⚠️ 自定义 | MMEdu 导出的 ONNX 模型 |
| baseml | 用户自定义.pkl | 用户本地 | 多种 | 多种 | scikit-learn | ⚠️ 自定义 | BaseML 导出的 sklearn 模型，需要 scikit-learn |
| basenn | 用户自定义 | 用户本地 | 多种 | 多种 | — | ⚠️ 自定义 | BaseNN 导出的模型 |
| custom | 用户自定义.onnx | 用户本地 | 多种 | 多种 | — | ⚠️ 自定义 | 用户自定义 ONNX 模型 + pre/post process |
| repo | ModelScope repo | 远程或本地 | 多种 | 多种 | modelscope | ⚠️ 可选 | ModelScope 模型库支持 |

## 统计

- **内置预训练模型**：27 个（其中 1 个因 bug 不支持自动下载）
- **支持自定义/用户模型的 task**：5 个
- **可选依赖**：
  - `rapidocr_onnxruntime`（用于 ocr）
  - `soundfile`（用于 embedding_audio）
  - `scikit-learn`（用于 baseml）
  - `modelscope`（用于 repo）
  - `gradio`（用于 LLM 的 UI）
  - `gradio-client`（用于 LLM 的 xedu_url）
  - `holidays`（用于 easter egg）

## 数据来源

- 大多数模型托管在 `https://www.openinnolab.org.cn/`
- det_face YuNet 从 GitHub opencv_zoo 下载
- 用户自定义模型可来自本地或 ModelScope

## 已知问题

1. **pose_face106**：原代码在模型下载映射表中复用了 pose_wholebody133 的 URL，会导致下载到错误的模型。已修复为显式 RuntimeError，用户需要手动指定 checkpoint。
2. **segment_anything**：需要两个文件（encoder + decoder），依赖项较多。
3. **embedding_audio**：依赖 soundfile，在没有声卡的服务器环境下可能有兼容性问题。
