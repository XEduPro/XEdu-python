# -*- coding: utf-8 -*-
"""
模型发现与目录 API

提供用户友好的接口来探索和了解可用的模型及任务
"""

from typing import Dict, List, Any, Optional
from .model_registry import list_tasks, get_models_by_task, list_all_models


def support_tasks(lang: str = "en") -> List[str]:
    """列出所有支持的任务

    Args:
        lang: 语言（暂未实现多语言，预留扩展）

    Returns:
        任务名称列表
    """
    return list_tasks()


def support_models(task_name: Optional[str] = None) -> List[str]:
    """列出所有支持的模型

    Args:
        task_name: 可选，如果指定则返回该任务的所有模型

    Returns:
        模型 ID 列表
    """
    if task_name:
        models = get_models_by_task(task_name)
        return [m.model_id for m in models]
    else:
        models = list_all_models()
        return [m.model_id for m in models]


def describe_task(task_name: str, lang: str = "en") -> Dict[str, Any]:
    """获取任务的详细信息

    Args:
        task_name: 任务名
        lang: 语言

    Returns:
        包含任务描述、输入输出类型、可用模型等信息的字典
    """
    models = get_models_by_task(task_name)

    if not models:
        return {
            "task": task_name,
            "status": "not_found",
            "error": f"Task '{task_name}' not found",
        }

    primary_model = models[0]

    return {
        "task": task_name,
        "description": primary_model.description,
        "input_type": primary_model.input_type,
        "output_type": primary_model.output_type,
        "available_models": [m.model_id for m in models],
        "models": [
            {
                "model_id": m.model_id,
                "filename": m.filename,
                "description": m.description,
                "latency_tier": m.latency_tier,
                "quality_tier": m.quality_tier,
                "auto_download": m.auto_download,
                "mirror_count": len(m.mirror_urls),
                "recommended_for": m.recommended_for,
                "tags": m.tags,
            }
            for m in models
        ],
        "optional_dependencies": list(set(
            dep for m in models for dep in m.optional_dependencies
        )),
        "auto_download_supported": any(m.auto_download for m in models),
        "recommended_for": list(set(
            rec for m in models for rec in m.recommended_for
        )),
        "tags": list(set(tag for m in models for tag in m.tags)),
    }


def describe_model(model_id: str, lang: str = "en") -> Dict[str, Any]:
    """获取模型的详细信息

    Args:
        model_id: 模型 ID
        lang: 语言

    Returns:
        包含模型详细信息的字典
    """
    from .model_registry import get_model

    model = get_model(model_id)

    if not model:
        return {
            "model_id": model_id,
            "status": "not_found",
            "error": f"Model '{model_id}' not found",
        }

    return {
        "model_id": model_id,
        "task": model.task_name,
        "filename": model.filename,
        "description": model.description,
        "input_type": model.input_type,
        "output_type": model.output_type,
        "source_url": model.source_url,
        "mirror_urls": model.mirror_urls,
        "auto_download": model.auto_download,
        "checksum": model.checksum,
        "optional_dependencies": model.optional_dependencies,
        "providers": model.providers,
        "latency_tier": model.latency_tier,
        "quality_tier": model.quality_tier,
        "recommended_for": model.recommended_for,
        "tags": model.tags,
    }


def model_matrix(category: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """获取模型矩阵：按类别和性能层级组织

    Args:
        category: 可选，筛选类别（如 'vision', 'nlp', 'audio'）

    Returns:
        按类别组织的模型信息
    """
    all_models = list_all_models()

    result = {}

    for model in all_models:
        # 按 output_type 分组（作为主类别）
        cat = model.output_type

        if category and category not in model.tags:
            continue

        if cat not in result:
            result[cat] = []

        result[cat].append({
            "model_id": model.model_id,
            "task": model.task_name,
            "latency_tier": model.latency_tier,
            "quality_tier": model.quality_tier,
            "description": model.description,
            "auto_download": model.auto_download,
        })

    # 按 latency_tier 排序
    tier_order = {"tiny": 0, "base": 1, "large": 2, None: 1}
    for cat in result:
        result[cat].sort(key=lambda m: tier_order.get(m["latency_tier"], 1))

    return result
