# -*- coding: utf-8 -*-
"""
XEdu 模型下载与缓存管理

统一管理模型文件的获取、缓存、校验，支持：
- 多级缓存目录（user/project/custom）
- 自动下载控制
- 完整性校验
- 清晰的错误提示
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .exceptions import (
    XEduModelNotFoundError,
    XEduModelDownloadError,
    XEduDependencyError,
)
from .model_registry import get_model, ModelMetadata


class ModelStore:
    """模型存储与管理"""

    def __init__(self):
        """初始化缓存目录"""
        self.cache_dir = self._resolve_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def _resolve_cache_dir() -> str:
        """确定缓存目录

        优先级：
        1. XEDU_HOME 环境变量
        2. XEDU_CACHE_DIR 环境变量
        3. 平台默认位置
        """
        # 优先用户自定义
        if custom := os.environ.get("XEDU_HOME"):
            return os.path.abspath(os.path.expanduser(custom))

        if custom := os.environ.get("XEDU_CACHE_DIR"):
            return os.path.abspath(os.path.expanduser(custom))

        # 平台默认
        if os.name == "nt":  # Windows
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            return os.path.join(base, "xedu", "models")
        else:  # Linux / macOS
            base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
                os.path.expanduser("~"), ".cache"
            )
            return os.path.join(base, "xedu", "models")

    def get_model_path(
        self, model_id: str, auto_download: bool = True
    ) -> str:
        """获取模型文件路径

        Args:
            model_id: 模型 ID
            auto_download: 如果文件不存在是否自动下载

        Returns:
            模型文件的完整路径

        Raises:
            XEduModelNotFoundError: 模型未找到（无下载 URL 或下载失败）
            XEduModelDownloadError: 下载失败
        """
        metadata = get_model(model_id)
        if not metadata:
            raise XEduModelNotFoundError(f"Model '{model_id}' not found in registry")

        # 构造本地路径
        local_path = os.path.join(self.cache_dir, metadata.filename)

        # 文件已存在
        if os.path.exists(local_path):
            return local_path

        # 文件不存在，检查是否可下载
        if not auto_download:
            raise XEduModelNotFoundError(
                f"Model file not found: {local_path}\n"
                f"Set auto_download=True or provide a local checkpoint"
            )

        if not metadata.auto_download:
            raise XEduModelNotFoundError(
                f"Model '{model_id}' does not support automatic download. "
                f"Please provide a local checkpoint file."
            )

        if not metadata.source_url:
            raise XEduModelNotFoundError(
                f"Model '{model_id}' has no download URL in registry"
            )

        # 下载模型
        self.download(metadata.source_url, local_path, metadata.checksum)

        return local_path

    def download(
        self,
        url: str,
        local_path: str,
        checksum: Optional[str] = None,
        chunk_size: int = 8192,
    ) -> None:
        """下载模型文件

        Args:
            url: 下载 URL
            local_path: 本地保存路径
            checksum: 可选的 SHA256 校验和
            chunk_size: 下载块大小

        Raises:
            XEduModelDownloadError: 下载或校验失败
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            # HEAD 请求获取文件大小
            response = requests.head(url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            # GET 请求下载文件
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            print(f"Downloading {os.path.basename(local_path)}...")

            # 用 tqdm 显示进度
            with open(local_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # 校验
            if checksum:
                self._verify_checksum(local_path, checksum)

            print(f"Model saved to {local_path}")

        except requests.RequestException as e:
            # 清理残留的不完整文件
            if os.path.exists(local_path):
                os.remove(local_path)
            raise XEduModelDownloadError(f"Failed to download model: {e}") from e

    @staticmethod
    def _verify_checksum(file_path: str, expected_checksum: str) -> None:
        """校验文件 SHA256

        Args:
            file_path: 文件路径
            expected_checksum: 期望的 SHA256 校验和

        Raises:
            XEduModelDownloadError: 校验失败
        """
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual = sha256_hash.hexdigest()

        if actual.lower() != expected_checksum.lower():
            os.remove(file_path)
            raise XEduModelDownloadError(
                f"Checksum mismatch for {file_path}\n"
                f"Expected: {expected_checksum}\n"
                f"Got: {actual}"
            )

    def check_dependencies(self, model_id: str) -> None:
        """检查模型的依赖是否满足

        Args:
            model_id: 模型 ID

        Raises:
            XEduDependencyError: 依赖缺失
        """
        metadata = get_model(model_id)
        if not metadata:
            return

        for dep in metadata.optional_dependencies:
            if not self._is_package_available(dep):
                raise XEduDependencyError(
                    f"Model '{model_id}' requires optional dependency '{dep}'. "
                    f"Install it with: pip install {dep}"
                )

    @staticmethod
    def _is_package_available(package_name: str) -> bool:
        """检查包是否已安装"""
        # 处理特殊的包名映射
        import_name = package_name.replace("-", "_")

        try:
            __import__(import_name)
            return True
        except ImportError:
            return False


# 全局模型存储实例
_global_store = None


def get_model_store() -> ModelStore:
    """获取全局模型存储实例"""
    global _global_store
    if _global_store is None:
        _global_store = ModelStore()
    return _global_store


def get_model_path(model_id: str, auto_download: bool = True) -> str:
    """便利函数：获取模型路径"""
    store = get_model_store()
    return store.get_model_path(model_id, auto_download)


def check_dependencies(model_id: str) -> None:
    """便利函数：检查依赖"""
    store = get_model_store()
    return store.check_dependencies(model_id)
