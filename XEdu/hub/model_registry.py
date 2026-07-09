# -*- coding: utf-8 -*-
"""
XEdu 模型注册表

统一管理所有预训练模型的元数据，包括：
- 任务名称
- 默认 checkpoint 文件名
- 下载 URL
- 输入输出类型
- 依赖要求
- 模型类型标签
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelMetadata:
    """模型元数据定义"""

    # 基本信息
    task_name: str  # 任务名，如 'pose_body17'
    model_id: str  # 模型唯一 ID，如 'pose_body17-default'
    filename: str  # 本地保存文件名，如 'body17.onnx'

    # 来源信息
    source_url: Optional[str] = None  # 下载 URL
    source_type: str = "remote"  # 来源类型: remote / local / repo

    # 输入输出约定
    input_type: str = "image"  # image / text / audio / image+text / ...
    output_type: str = "detection"  # detection / pose / classification / embedding / ...

    # 依赖与兼容性
    optional_dependencies: List[str] = None  # 可选依赖列表
    providers: List[str] = None  # 支持的 provider: cpu / cuda / tensorrt / ...

    # 行为控制
    auto_download: bool = True  # 是否支持自动下载
    checksum: Optional[str] = None  # sha256 校验和

    # 分类与发现
    tags: List[str] = None  # 标签，如 ['vision', 'pose', 'education']
    description: str = ""  # 简短描述

    # 性能特征（可选）
    latency_tier: Optional[str] = None  # tiny / base / large
    quality_tier: Optional[str] = None  # low / medium / high
    recommended_for: List[str] = None  # classroom / edge / demo / high_accuracy / ...

    def __post_init__(self):
        """初始化默认值"""
        if self.optional_dependencies is None:
            self.optional_dependencies = []
        if self.providers is None:
            self.providers = ["cpu"]
        if self.tags is None:
            self.tags = []
        if self.recommended_for is None:
            self.recommended_for = []


class ModelRegistry:
    """模型注册表，管理所有可用模型"""

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._task_to_models: Dict[str, List[str]] = {}

    def register(self, metadata: ModelMetadata) -> None:
        """注册一个模型"""
        self._models[metadata.model_id] = metadata

        if metadata.task_name not in self._task_to_models:
            self._task_to_models[metadata.task_name] = []
        self._task_to_models[metadata.task_name].append(metadata.model_id)

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """获取单个模型元数据"""
        return self._models.get(model_id)

    def get_models_by_task(self, task_name: str) -> List[ModelMetadata]:
        """获取某个 task 的所有模型"""
        model_ids = self._task_to_models.get(task_name, [])
        return [self._models[mid] for mid in model_ids]

    def get_default_model(self, task_name: str) -> Optional[ModelMetadata]:
        """获取 task 的默认模型（第一个注册的）"""
        models = self.get_models_by_task(task_name)
        return models[0] if models else None

    def list_tasks(self) -> List[str]:
        """列出所有可用的 task"""
        return sorted(self._task_to_models.keys())

    def list_all_models(self) -> List[ModelMetadata]:
        """列出所有已注册模型"""
        return list(self._models.values())


# 全局注册表实例
_global_registry = ModelRegistry()


def register_model(metadata: ModelMetadata) -> None:
    """全局模型注册函数"""
    _global_registry.register(metadata)


def get_model(model_id: str) -> Optional[ModelMetadata]:
    """获取单个模型"""
    return _global_registry.get_model(model_id)


def get_models_by_task(task_name: str) -> List[ModelMetadata]:
    """获取某个 task 的所有模型"""
    return _global_registry.get_models_by_task(task_name)


def get_default_model(task_name: str) -> Optional[ModelMetadata]:
    """获取 task 的默认模型"""
    return _global_registry.get_default_model(task_name)


def list_tasks() -> List[str]:
    """列出所有可用 task"""
    return _global_registry.list_tasks()


def list_all_models() -> List[ModelMetadata]:
    """列出所有已注册模型"""
    return _global_registry.list_all_models()


# ============================================================================
# 内置模型注册
# ============================================================================

# 检测类模型
register_model(ModelMetadata(
    task_name="det_body",
    model_id="det_body-default",
    filename="bodydetect.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/8137e4ca-482d-48fa-b57f-bfa50f7768be.onnx&name=det_body.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "human"],
    description="Human body detection",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="det_body_l",
    model_id="det_body_l-default",
    filename="bodydetect_l.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/d6d5680e-b3ef-4624-9a9f-52f1892f0045.onnx&name=det_body_l.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "human"],
    description="Human body detection (large model)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_coco",
    model_id="det_coco-default",
    filename="cocodetect.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/e4c39ead-ff3b-4810-ab4d-a8f3757ff1bb.onnx&name=det_coco.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "coco"],
    description="COCO object detection (80 classes)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="det_coco_l",
    model_id="det_coco_l-default",
    filename="cocodetect_l.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/8be89312-4ff7-4dc7-ba19-7d759d8e713a.onnx&name=det_coco_l.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "coco"],
    description="COCO object detection large model (80 classes)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_hand",
    model_id="det_hand-default",
    filename="handdetect.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/6172ad06-8a97-4d47-bcc3-cdbdda0c0187.onnx&name=det_hand.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "hand"],
    description="Hand detection",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="det_face",
    model_id="det_face-yunet",
    filename="face_detection_yunet_2023mar.onnx",
    source_url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    source_type="remote",
    input_type="image",
    output_type="detection",
    auto_download=True,
    tags=["vision", "detection", "face"],
    description="Face detection using YuNet 2023",
    latency_tier="tiny",
    recommended_for=["classroom", "edge", "demo"]
))

# 姿态类模型
register_model(ModelMetadata(
    task_name="pose_body17",
    model_id="pose_body17-default",
    filename="body17.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/b94f252e-03de-4491-b9f3-042c57c7671f.onnx&name=pose_body17.onnx",
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "human"],
    description="Human body pose (17 keypoints)",
    latency_tier="tiny",
    recommended_for=["classroom", "edge", "demo"]
))

register_model(ModelMetadata(
    task_name="pose_body17_l",
    model_id="pose_body17_l-default",
    filename="body17_l.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/8e71a720-e87b-42e7-8498-8a6d07473941.onnx&name=pose_body17_l.onnx",
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "human"],
    description="Human body pose large model (17 keypoints)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="pose_body26",
    model_id="pose_body26-default",
    filename="body26.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/2de9dd14-93c7-4b89-ac79-da3231c79d01.onnx&name=pose_body26.onnx",
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "human"],
    description="Human body pose (26 keypoints)",
    latency_tier="base",
    recommended_for=["classroom"]
))

register_model(ModelMetadata(
    task_name="pose_hand21",
    model_id="pose_hand21-default",
    filename="hand21.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/e5e5540b-3475-42f8-be0b-6ea8d46d577b.onnx&name=pose_hand21.onnx",
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "hand"],
    description="Hand pose (21 keypoints)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="pose_wholebody133",
    model_id="pose_wholebody133-default",
    filename="whole133.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/98e010a3-76f4-4209-bba9-33fba2fe1281.onnx&name=pose_wholebody133.onnx",
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "wholebody"],
    description="Whole body pose (133 keypoints: body + face + hand)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="pose_face106",
    model_id="pose_face106-default",
    filename="face106.onnx",
    source_url=None,
    auto_download=False,
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "face"],
    description=(
        "Face pose (106 keypoints). Automatic download is disabled: the "
        "legacy download map incorrectly reused the pose_wholebody133 URL "
        "for this task, which would silently install the wrong model. "
        "A local checkpoint must be provided explicitly, e.g. "
        "wf(task='pose_face106', checkpoint='/path/to/face106.onnx')."
    ),
    latency_tier="base",
    recommended_for=[]
))

# 分类模型
register_model(ModelMetadata(
    task_name="cls_imagenet",
    model_id="cls_imagenet-default",
    filename="imagenet1k.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/09a4c4f4-7034-45a5-a0c1-a747da5a2766.onnx&name=cls_imagenet.onnx",
    input_type="image",
    output_type="classification",
    tags=["vision", "classification"],
    description="ImageNet 1000-class classification",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

# 生成类模型（风格迁移）
# 真实下载地址来自 workflow.py 中的 model_name_map_download，
# 与 workflow.py 保持一致，避免出现占位符 URL。
_GEN_STYLE_ASSET_IDS = {
    "mosaic": "965b190c-6008-43dd-a037-94a99e55f78a",
    "candy": "bc24e059-131a-49d0-b663-45289156bbc9",
    "rain-princess": "f193af6e-8eaf-43c1-913c-85b226da4e47",
    "udnie": "3691c4c2-877b-4137-b621-7eb7dede54e3",
    "pointilism": "9e5fb84f-fcd5-497f-a59f-9359e430e549",
}
for style, asset_id in _GEN_STYLE_ASSET_IDS.items():
    register_model(ModelMetadata(
        task_name=f"gen_style_{style}",
        model_id=f"gen_style_{style}-default",
        filename=f"gen_style_{style}.onnx",
        source_url=f"https://www.openinnolab.org.cn/res/api/v1/file/creator/{asset_id}.onnx&name=gen_style_{style}.onnx",
        input_type="image",
        output_type="image",
        tags=["vision", "generation", "style_transfer"],
        description=f"Style transfer ({style})",
        latency_tier="base",
        recommended_for=["classroom", "demo"]
    ))

# 着色模型
register_model(ModelMetadata(
    task_name="gen_color",
    model_id="gen_color-default",
    filename="gen_color.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/733caa05-0357-4e52-a7b0-ce9a9419f959.onnx&name=gen_color.onnx",
    input_type="image",
    output_type="image",
    tags=["vision", "generation", "colorization"],
    description="Image colorization (grayscale → color)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

# 分割模型
# 注意：segment_anything 需要两个模型文件（encoder + decoder），
# 当前 ModelMetadata 是单文件 schema，因此拆成两条记录挂在同一个 task_name 下，
# 复用 get_models_by_task() 本身支持一对多的机制。
register_model(ModelMetadata(
    task_name="segment_anything",
    model_id="segment_anything-encoder",
    filename="seg_sam_encoder.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/b0baaf01-8673-4762-a99b-f47661454395.onnx&name=seg_sam_encoder.onnx",
    input_type="image",
    output_type="segmentation",
    tags=["vision", "segmentation"],
    description="SAM universal segmentation (encoder half; requires segment_anything-decoder too)",
    latency_tier="large",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="segment_anything",
    model_id="segment_anything-decoder",
    filename="seg_sam_decoder.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/70f02a96-6998-4196-92ac-c61a9a841c66.onnx&name=seg_sam_decoder.onnx",
    input_type="image",
    output_type="segmentation",
    tags=["vision", "segmentation"],
    description="SAM universal segmentation (decoder half; requires segment_anything-encoder too)",
    latency_tier="large",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="depth_anything",
    model_id="depth_anything-default",
    filename="depth_anything.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/ffa0880a-4900-4ef5-8106-562eb14e7e8e.onnx&name=depth_anything.onnx",
    input_type="image",
    output_type="depth",
    tags=["vision", "depth"],
    description="Depth estimation",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

# NLP 模型
register_model(ModelMetadata(
    task_name="nlp_qa",
    model_id="nlp_qa-default",
    filename="nlp_qa.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/b1938fd0-6ffa-4c91-a98e-170cd3b1c520.onnx&name=nlp_qa.onnx",
    input_type="text",
    output_type="text",
    tags=["nlp", "qa"],
    description="Question answering (SQuAD-based, BERT)",
    latency_tier="base",
    recommended_for=["classroom"]
))

# Embedding 模型
register_model(ModelMetadata(
    task_name="embedding_image",
    model_id="embedding_image-clip",
    filename="embedding_image.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/69aebb8e-3202-4022-9618-a64560ffef76.onnx&name=embedding_image.onnx",
    input_type="image",
    output_type="embedding",
    tags=["vision", "embedding"],
    description="CLIP image embedding (512-dim)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="embedding_text",
    model_id="embedding_text-clip",
    filename="embedding_text.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/ce38d2ad-e8be-4e6a-990a-a6d818e5655b.onnx&name=embedding_text.onnx",
    input_type="text",
    output_type="embedding",
    tags=["nlp", "embedding"],
    description="CLIP text embedding (512-dim)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="embedding_audio",
    model_id="embedding_audio-clap",
    filename="embedding_audio.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/3fb25823-aeb7-4866-9617-937f5079af4a.onnx&name=embedding_audio.onnx",
    input_type="audio",
    output_type="embedding",
    optional_dependencies=["soundfile"],
    tags=["audio", "embedding"],
    description="CLAP audio embedding (1024-dim)",
    latency_tier="base",
    recommended_for=["classroom"],
    auto_download=True
))

# 其他模型
register_model(ModelMetadata(
    task_name="drive_perception",
    model_id="drive_perception-default",
    filename="drive_perception.onnx",
    source_url="https://www.openinnolab.org.cn/res/api/v1/file/creator/add78652-51c0-41e3-ab56-a52dd7374e54.onnx&name=drive_perception.onnx",
    input_type="image",
    output_type="detection",
    tags=["vision", "autonomous_driving"],
    description="Autonomous driving perception",
    latency_tier="base",
    recommended_for=["demo"]
))

# 特殊任务（不在自动下载表中）
register_model(ModelMetadata(
    task_name="ocr",
    model_id="ocr-rapidocr",
    filename="rapidocr",
    source_type="external",
    input_type="image",
    output_type="text",
    optional_dependencies=["rapidocr_onnxruntime"],
    auto_download=False,
    tags=["vision", "nlp", "ocr"],
    description="OCR (text detection + recognition)",
    recommended_for=["classroom"]
))
