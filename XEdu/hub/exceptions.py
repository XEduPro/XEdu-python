# -*- coding: utf-8 -*-
"""
XEdu 统一异常定义

所有 XEdu 异常都从 XEduError 继承，便于用户统一捕获和处理。
"""


class XEduError(Exception):
    """XEdu 基础异常类，所有 XEdu 异常的基类。"""
    pass


class XEduConfigError(XEduError):
    """配置相关错误。

    包括：
    - 任务不存在
    - 模型配置缺失
    - 参数不合法
    """
    pass


class XEduTaskNotFoundError(XEduConfigError):
    """指定的任务不存在。

    例如：
        wf(task="nonexistent_task")
    """
    pass


class XEduModelNotFoundError(XEduError):
    """模型文件未找到。

    包括：
    - 本地 checkpoint 不存在
    - 远程模型地址无效
    - 自动下载失败后仍未找到模型
    """
    pass


class XEduModelDownloadError(XEduError):
    """模型下载相关错误。

    包括：
    - 网络连接失败
    - 下载文件大小不匹配
    - 文件完整性校验失败
    """
    pass


class XEduDependencyError(XEduError):
    """依赖包缺失错误。

    当任务需要某个可选依赖（如 rapidocr、soundfile、scikit-learn 等）
    但该依赖未安装时抛出。

    例如：
        from XEdu.hub import Workflow
        wf = Workflow(task='ocr')  # 缺少 rapidocr_onnxruntime
        # 会抛出 XEduDependencyError
    """
    pass


class XEduInputError(XEduError):
    """输入数据错误。

    包括：
    - 输入类型不符合要求
    - 输入维度不匹配
    - 文件格式错误（例如图片读取失败）
    """
    pass


class XEduInferenceError(XEduError):
    """模型推理错误。

    包括：
    - 推理过程异常（例如 ONNX Runtime 错误）
    - 输出格式异常
    - 输出校验失败
    """
    pass


class XEduOutputError(XEduError):
    """输出处理错误。

    包括：
    - 后处理失败
    - 输出格式转换失败
    """
    pass


class XEduAPIError(XEduError):
    """外部 API 调用错误。

    用于 LLM、ModelScope 等外部服务的错误。

    包括：
    - API 请求失败
    - API 返回错误码
    - 认证失败
    """
    pass


class XEduAuthenticationError(XEduAPIError):
    """认证相关错误。

    包括：
    - API key 缺失
    - API key 无效
    - 权限不足
    """
    pass


class XEduRateLimitError(XEduAPIError):
    """API 调用频率限制。

    当请求过于频繁被限流时抛出。
    """
    pass


class XEduNotSupportedError(XEduError):
    """功能不支持错误。

    例如：
    - 在不支持的设备上（如非 CUDA 但要求 GPU）
    - 不支持的参数组合
    - 临时不可用的功能（如 pose_face106 自动下载）
    """
    pass


# 便利函数，用于生成有上下文的错误信息
def make_dependency_error(task: str, required_package: str, install_cmd: str = None) -> XEduDependencyError:
    """生成依赖错误，包含安装建议。

    Args:
        task: 需要该依赖的任务名
        required_package: 缺失的包名
        install_cmd: 安装命令（可选，如果不提供会自动生成）

    Returns:
        XEduDependencyError 异常对象
    """
    if install_cmd is None:
        install_cmd = f"pip install {required_package}"

    msg = (
        f"The '{task}' task requires the optional dependency '{required_package}'. "
        f"Install it with:\n  {install_cmd}"
    )
    return XEduDependencyError(msg)


def make_model_not_found_error(task: str, checkpoint_path: str, suggest_manual: bool = False) -> XEduModelNotFoundError:
    """生成模型未找到错误，包含建议。

    Args:
        task: 任务名
        checkpoint_path: 期望的 checkpoint 路径
        suggest_manual: 是否建议用户手动指定模型

    Returns:
        XEduModelNotFoundError 异常对象
    """
    msg = f"Model for task '{task}' not found at {checkpoint_path}"
    if suggest_manual:
        msg += f". Please provide a valid checkpoint:\n  wf = Workflow(task='{task}', checkpoint='/path/to/model.onnx')"
    return XEduModelNotFoundError(msg)
