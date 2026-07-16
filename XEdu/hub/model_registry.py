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
    source_url: Optional[str] = None  # 主下载 URL
    mirror_urls: List[str] = None  # 备用下载 URL
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
        if self.mirror_urls is None:
            self.mirror_urls = []
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



MODELSCOPE_MODEL_REPO = "wht0926/xedu-hub-models"
MODELSCOPE_REVISION = "master"


def modelscope_url(filename: str) -> str:
    return f"https://modelscope.cn/models/{MODELSCOPE_MODEL_REPO}/resolve/{MODELSCOPE_REVISION}/{filename}"


MODEL_CHECKSUMS = {
    "body17.onnx": "010662e1f7830c2e6eaa29f7eeb404a10e37bc4e0574c5113d760785366ce9d1",
    "body17_l.onnx": "1b6bc4313b78df115590d35cf4e5b5ab478bb30f5e289dbe647751be389807e0",
    "body26.onnx": "228ad4bf6d9afd99c1f81fb5d775b442f93a64fc5628c8a7d6dffee037c01aeb",
    "bodydetect.onnx": "4720eda18a18402e7b76d85827d4bc60fba18111fc0b74b559e4d24da9ddc3d2",
    "bodydetect_l.onnx": "874d00f4f6011621fccd10e0aad418c4e9c733a48d840a08782858e9e443992d",
    "cocodetect.onnx": "b8fc9455390a565213ba5d6abcd0cf2edc9bec8b9f4dffd73882f47541f62827",
    "cocodetect_l.onnx": "cc697db9ddd52727746ab42e700239632f28cfc2a5543fa3da3e5e5a9b032c83",
    "depth_anything.onnx": "21e149bdc73d9173c2dfe0dce9a2b7d0d3bcf9be06440ad987a703b7ce30ea54",
    "drive_perception.onnx": "7c24d01112cb924c1a6fc36f854581e1d9e94bcd575537cfb1870a84e4e3adff",
    "embedding_audio.onnx": "2cc80b4aad1eb6b70db9a5f4c2b2a29ef482f68f08539c8a8588c94a76da72cc",
    "embedding_image.onnx": "4962914240f80dbac5b9d33dc3fa321cdcaa4820193d65c51e8edc3cd28109d2",
    "embedding_text.onnx": "c7ef57e5bfd920bd65e288cac200168c5fe4deb5c39bcf4323c8f16ace8d3359",
    "face_landmark106_mobilenet.onnx": "8b730ef412d8db18e6af5602fd2bc052f5afba3d330de5be8fb398ab9a2f6cff",
    "face_landmark98_pipnet_wflw.onnx": "9862838dc6144bc772b6485f6f6d31295c0b1c1ab7293e6ddeb0a439cb10218d",
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "gen_color.onnx": "6388afcc7a6cd1da3c116509e593263af77d3bef1d11f81d6d73348b7635d846",
    "gen_style_candy.onnx": "255ae574d7e6708e41ec9cc3a592c3f0afc1a5dfc9b943b6628d319496033851",
    "gen_style_custom.onnx": "3896c81225a1338d1240fe95104ce093a9571465bc14ecfde22234babbe84dd2",
    "gen_style_mosaic.onnx": "bc8c9078c229c84d7c2b5c767cba4f7ace26c3d97216d4434d1297b66c738124",
    "gen_style_pointilism.onnx": "d40a5822cae890c361f3a0dcef5f3522d6a8cec23184bcd8a1395eddbf132293",
    "gen_style_rain-princess.onnx": "187fd206c79e503590cbf3ba7d3ed83bf8813d78d34d2a6328ffb5ceef72f460",
    "gen_style_udnie.onnx": "7aa72b583dfa383047c292fe28d2451b91445a8ced296a2fd6185c53e6fd47e3",
    "hand21.onnx": "20880f1a7b02b1bed87c9aba91b7018d283967a071228b64cb440b315a301b20",
    "palm_detection_full_inf_post_192x192.onnx": "3530b2b4a50d80173b573cd478b13436a015d3058bcff68e74435c6414ed7de3",
    "handdetect.onnx": "76eef89f110fef03a8f11618d7605ae0bbf13dad6e26ab6e960ca1d8d52ccee0",
    "imagenet1k.onnx": "c1e056d20073708c518decd93af610325d3fa5c55150edd63715002df4b0db00",
    "nlp_qa.onnx": "02cd45105864ad0c8022b520c2cf8a9c52da42e10fcdaca9d0e712d725febe6f",
    "seg_sam_decoder.onnx": "1fb0842eb7b5b78df05f1ed3f2d0e1f4e57002520cc96485de9b0942b6f11179",
    "seg_sam_encoder.onnx": "d0658526091e6916ed5f1f15e0a3b99eccfd1969abe93a46780bac409bac4459",
    "whole133.onnx": "199335b9d010f81a26835d882437693b50536b088c3851b3573ae43d9a8fa2f5",
}


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
    source_url=modelscope_url("bodydetect.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/8137e4ca-482d-48fa-b57f-bfa50f7768be.onnx&name=det_body.onnx'],
    checksum=MODEL_CHECKSUMS["bodydetect.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "human"],
    description="Human body detection",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="det_body",
    model_id="det_body_l-default",
    filename="bodydetect_l.onnx",
    source_url=modelscope_url("bodydetect_l.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/d6d5680e-b3ef-4624-9a9f-52f1892f0045.onnx&name=det_body_l.onnx'],
    checksum=MODEL_CHECKSUMS["bodydetect_l.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "human"],
    description="Human body detection (large model)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_body_l",
    model_id="det_body_l-compat",
    filename="bodydetect_l.onnx",
    source_url=modelscope_url("bodydetect_l.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/d6d5680e-b3ef-4624-9a9f-52f1892f0045.onnx&name=det_body_l.onnx'],
    checksum=MODEL_CHECKSUMS["bodydetect_l.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "human", "compat"],
    description="Human body detection (large model, legacy task alias)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_coco",
    model_id="det_coco-default",
    filename="cocodetect.onnx",
    source_url=modelscope_url("cocodetect.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/e4c39ead-ff3b-4810-ab4d-a8f3757ff1bb.onnx&name=det_coco.onnx'],
    checksum=MODEL_CHECKSUMS["cocodetect.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "coco"],
    description="COCO object detection (80 classes)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="det_coco",
    model_id="det_coco_l-default",
    filename="cocodetect_l.onnx",
    source_url=modelscope_url("cocodetect_l.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/8be89312-4ff7-4dc7-ba19-7d759d8e713a.onnx&name=det_coco_l.onnx'],
    checksum=MODEL_CHECKSUMS["cocodetect_l.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "coco"],
    description="COCO object detection large model (80 classes)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_coco_l",
    model_id="det_coco_l-compat",
    filename="cocodetect_l.onnx",
    source_url=modelscope_url("cocodetect_l.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/8be89312-4ff7-4dc7-ba19-7d759d8e713a.onnx&name=det_coco_l.onnx'],
    checksum=MODEL_CHECKSUMS["cocodetect_l.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "coco", "compat"],
    description="COCO object detection large model (legacy task alias)",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

register_model(ModelMetadata(
    task_name="det_hand",
    model_id="det_hand-palm-onnx",
    filename="palm_detection_full_inf_post_192x192.onnx",
    source_url=modelscope_url("palm_detection_full_inf_post_192x192.onnx"),
    mirror_urls=["https://raw.githubusercontent.com/PINTO0309/hand-gesture-recognition-using-onnx/main/model/palm_detection/palm_detection_full_inf_post_192x192.onnx", "https://raw.githubusercontent.com/betodamian/RealtimeONNXHandtracking/main/palm_detection_full_inf_post_192x192.onnx"],
    checksum=MODEL_CHECKSUMS["palm_detection_full_inf_post_192x192.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "hand"],
    description="Multi-hand palm detection (pure ONNX, MediaPipe-derived)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

# 保留旧模型 ID，便于旧代码显式指定 checkpoint；默认 det_hand 使用上面的 Palm ONNX。
register_model(ModelMetadata(
    task_name="det_hand",
    model_id="det_hand-default",
    filename="handdetect.onnx",
    source_url=modelscope_url("handdetect.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/6172ad06-8a97-4d47-bcc3-cdbdda0c0187.onnx&name=det_hand.onnx'],
    checksum=MODEL_CHECKSUMS["handdetect.onnx"],
    input_type="image",
    output_type="detection",
    tags=["vision", "detection", "hand", "compat"],
    description="Legacy hand detector (compatibility model)",
    latency_tier="base",
    recommended_for=["compatibility"]
))

register_model(ModelMetadata(
    task_name="det_face",
    model_id="det_face-yunet",
    filename="face_detection_yunet_2023mar.onnx",
    source_url=modelscope_url("face_detection_yunet_2023mar.onnx"),
    mirror_urls=['https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'],
    checksum=MODEL_CHECKSUMS["face_detection_yunet_2023mar.onnx"],
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
    source_url=modelscope_url("body17.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/b94f252e-03de-4491-b9f3-042c57c7671f.onnx&name=pose_body17.onnx'],
    checksum=MODEL_CHECKSUMS["body17.onnx"],
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
    source_url=modelscope_url("body17_l.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/8e71a720-e87b-42e7-8498-8a6d07473941.onnx&name=pose_body17_l.onnx'],
    checksum=MODEL_CHECKSUMS["body17_l.onnx"],
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
    source_url=modelscope_url("body26.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/2de9dd14-93c7-4b89-ac79-da3231c79d01.onnx&name=pose_body26.onnx'],
    checksum=MODEL_CHECKSUMS["body26.onnx"],
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
    source_url=modelscope_url("hand21.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/e5e5540b-3475-42f8-be0b-6ea8d46d577b.onnx&name=pose_hand21.onnx'],
    checksum=MODEL_CHECKSUMS["hand21.onnx"],
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
    source_url=modelscope_url("whole133.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/98e010a3-76f4-4209-bba9-33fba2fe1281.onnx&name=pose_wholebody133.onnx'],
    checksum=MODEL_CHECKSUMS["whole133.onnx"],
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

register_model(ModelMetadata(
    task_name="pose_face",
    model_id="pose_face_landmark-mobilenet106",
    filename="face_landmark106_mobilenet.onnx",
    source_url=modelscope_url("face_landmark106_mobilenet.onnx"),
    mirror_urls=[],
    checksum=MODEL_CHECKSUMS["face_landmark106_mobilenet.onnx"],
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "face", "landmark", "mobilenet", "lightweight", "derived"],
    description=(
        "Lightweight face landmark detector (106 points + head pose), "
        "exported from the ModelScope/IIC MobileNet face 2D keypoints model."
    ),
    latency_tier="tiny",
    quality_tier="medium",
    recommended_for=["classroom", "edge", "demo"]
))

register_model(ModelMetadata(
    task_name="pose_face",
    model_id="pose_face_landmark-pipnet98-wflw",
    filename="face_landmark98_pipnet_wflw.onnx",
    source_url=modelscope_url("face_landmark98_pipnet_wflw.onnx"),
    mirror_urls=[
        "https://github.com/yakhyo/pipnet-onnx/releases/download/weights/pipnet_r18_wflw_98.onnx"
    ],
    checksum=MODEL_CHECKSUMS["face_landmark98_pipnet_wflw.onnx"],
    input_type="image",
    output_type="pose",
    tags=["vision", "pose", "face", "landmark", "pipnet", "wflw", "high_accuracy"],
    description="High-accuracy PIPNet face landmark detector (WFLW 98 points, ResNet-18).",
    latency_tier="large",
    quality_tier="high",
    recommended_for=["high_accuracy"]
))

# 分类模型
register_model(ModelMetadata(
    task_name="cls_imagenet",
    model_id="cls_imagenet-default",
    filename="imagenet1k.onnx",
    source_url=modelscope_url("imagenet1k.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/09a4c4f4-7034-45a5-a0c1-a747da5a2766.onnx&name=cls_imagenet.onnx'],
    checksum=MODEL_CHECKSUMS["imagenet1k.onnx"],
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
    "custom": "17b8f5e6-94b6-44a3-b15b-486f6fc6a142",
}
for style, asset_id in _GEN_STYLE_ASSET_IDS.items():
    register_model(ModelMetadata(
        task_name=f"gen_style_{style}",
        model_id=f"gen_style_{style}-default",
        filename=f"gen_style_{style}.onnx",
        source_url=modelscope_url(f"gen_style_{style}.onnx"),
        mirror_urls=[f"https://www.openinnolab.org.cn/res/api/v1/file/creator/{asset_id}.onnx&name=gen_style_{style}.onnx"],
        checksum=MODEL_CHECKSUMS[f"gen_style_{style}.onnx"],
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
    source_url=modelscope_url("gen_color.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/733caa05-0357-4e52-a7b0-ce9a9419f959.onnx&name=gen_color.onnx'],
    checksum=MODEL_CHECKSUMS["gen_color.onnx"],
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
    source_url=modelscope_url("seg_sam_encoder.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/b0baaf01-8673-4762-a99b-f47661454395.onnx&name=seg_sam_encoder.onnx'],
    checksum=MODEL_CHECKSUMS["seg_sam_encoder.onnx"],
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
    source_url=modelscope_url("seg_sam_decoder.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/70f02a96-6998-4196-92ac-c61a9a841c66.onnx&name=seg_sam_decoder.onnx'],
    checksum=MODEL_CHECKSUMS["seg_sam_decoder.onnx"],
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
    source_url=modelscope_url("depth_anything.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/ffa0880a-4900-4ef5-8106-562eb14e7e8e.onnx&name=depth_anything.onnx'],
    checksum=MODEL_CHECKSUMS["depth_anything.onnx"],
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
    source_url=modelscope_url("nlp_qa.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/b1938fd0-6ffa-4c91-a98e-170cd3b1c520.onnx&name=nlp_qa.onnx'],
    checksum=MODEL_CHECKSUMS["nlp_qa.onnx"],
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
    source_url=modelscope_url("embedding_image.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/69aebb8e-3202-4022-9618-a64560ffef76.onnx&name=embedding_image.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_image.onnx"],
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
    source_url=modelscope_url("embedding_text.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/ce38d2ad-e8be-4e6a-990a-a6d818e5655b.onnx&name=embedding_text.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_text.onnx"],
    input_type="text",
    output_type="embedding",
    tags=["nlp", "embedding"],
    description="CLIP text embedding (512-dim)",
    latency_tier="base",
    recommended_for=["classroom", "demo"]
))

register_model(ModelMetadata(
    task_name="cls_text",
    model_id="cls_text-clip-prototype",
    filename="embedding_text.onnx",
    source_url=modelscope_url("embedding_text.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/ce38d2ad-e8be-4e6a-990a-a6d818e5655b.onnx&name=embedding_text.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_text.onnx"],
    input_type="text",
    output_type="classification",
    tags=["nlp", "classification", "embedding", "offline", "prototype"],
    description=(
        "Offline prototype-based text classification using CLIP text embeddings. "
        "Chinese text can be tokenized, but this is not a dedicated Chinese "
        "embedding model; validate margins for Chinese use cases."
    ),
    latency_tier="base",
    recommended_for=["classroom", "demo"],
    auto_download=True
))

register_model(ModelMetadata(
    task_name="embedding_audio",
    model_id="embedding_audio-clap",
    filename="embedding_audio.onnx",
    source_url=modelscope_url("embedding_audio.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/3fb25823-aeb7-4866-9617-937f5079af4a.onnx&name=embedding_audio.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_audio.onnx"],
    input_type="audio",
    output_type="embedding",
    optional_dependencies=["soundfile"],
    tags=["audio", "embedding"],
    description="CLAP audio embedding (1024-dim)",
    latency_tier="base",
    recommended_for=["classroom"],
    auto_download=True
))

register_model(ModelMetadata(
    task_name="cls_audio",
    model_id="cls_audio-clap-prototype",
    filename="embedding_audio.onnx",
    source_url=modelscope_url("embedding_audio.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/3fb25823-aeb7-4866-9617-937f5079af4a.onnx&name=embedding_audio.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_audio.onnx"],
    input_type="audio",
    output_type="classification",
    optional_dependencies=["soundfile"],
    tags=["audio", "classification", "embedding", "prototype"],
    description="CLAP prototype-based audio classification using reference audio examples",
    latency_tier="base",
    recommended_for=["classroom"],
    auto_download=True
))

register_model(ModelMetadata(
    task_name="det_audio_keyword",
    model_id="det_audio_keyword-clap-prototype",
    filename="embedding_audio.onnx",
    source_url=modelscope_url("embedding_audio.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/3fb25823-aeb7-4866-9617-937f5079af4a.onnx&name=embedding_audio.onnx'],
    checksum=MODEL_CHECKSUMS["embedding_audio.onnx"],
    input_type="audio",
    output_type="detection",
    optional_dependencies=["soundfile"],
    tags=["audio", "keyword", "detection", "embedding", "prototype"],
    description="CLAP prototype-based keyword or sound-event detection",
    latency_tier="base",
    recommended_for=["classroom"],
    auto_download=True
))

register_model(ModelMetadata(
    task_name="match_image_text",
    model_id="match_image_text-clip",
    filename="embedding_image.onnx+embedding_text.onnx",
    source_url=None,
    source_type="composite",
    input_type="image+text",
    output_type="multimodal",
    tags=["vision", "nlp", "multimodal", "matching", "clip"],
    description=(
        "Image-text matching task composed from local CLIP image and text "
        "embedding models; no separate multimodal checkpoint is required."
    ),
    latency_tier="base",
    recommended_for=["classroom", "demo"],
    auto_download=True
))

# 其他模型
register_model(ModelMetadata(
    task_name="drive_perception",
    model_id="drive_perception-default",
    filename="drive_perception.onnx",
    source_url=modelscope_url("drive_perception.onnx"),
    mirror_urls=['https://www.openinnolab.org.cn/res/api/v1/file/creator/add78652-51c0-41e3-ab56-a52dd7374e54.onnx&name=drive_perception.onnx'],
    checksum=MODEL_CHECKSUMS["drive_perception.onnx"],
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
