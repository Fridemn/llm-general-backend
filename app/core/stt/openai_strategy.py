import os
import requests
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, BinaryIO, Union

from app import logger
from . import STTProvider


class ProviderOpenAISTT(STTProvider):
    """OpenAI Whisper STT提供者实现"""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        """
        初始化OpenAI STT提供者

        Args:
            provider_config: 提供者配置
            provider_settings: 全局设置
        """
        super().__init__(provider_config, provider_settings)

        # 从配置中获取API密钥和基础URL
        self.api_key = provider_config.get("api_key")
        self.api_base = provider_config.get("base_url")
        self.model = provider_config.get("model")

        # 确保api_base不以斜杠结尾
        if self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]

        # 从提供者名称设置模型
        self.set_model("openai_whisper")

        # logger.info(f"初始化OpenAI STT提供者 - API基础URL: {self.api_base}, 模型: {self.model}")

        if not self.api_key:
            logger.warning("未提供API密钥，可能导致请求失败")

    def set_model(self, model_name: str) -> None:
        """
        设置模型名称

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name

    def _transcribe_sync(self, audio_file: Union[str, Path, BinaryIO], **kwargs) -> Dict[str, Any]:
        """
        同步方式转录音频文件

        Args:
            audio_file: 音频文件路径或文件对象
            **kwargs: 传递给API的其他参数

        Returns:
            API返回的结果
        """
        endpoint = f"{self.api_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # logger.info(f"开始转录，使用端点: {endpoint}")
        # 检查文件是否存在
        if isinstance(audio_file, (str, Path)) and not os.path.exists(audio_file):
            raise FileNotFoundError(f"找不到音频文件: {audio_file}")

        try:
            # 准备文件
            if isinstance(audio_file, (str, Path)):
                with open(audio_file, "rb") as f:
                    # logger.info(f"发送文件: {audio_file}")
                    files = {"file": f}
                    data = {"model": kwargs.pop("model", self.model), **kwargs}
                    response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            else:
                # 假设audio_file是一个已打开的文件对象
                files = {"file": audio_file}
                data = {"model": kwargs.pop("model", self.model), **kwargs}
                response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)

            # logger.info(f"API响应状态码: {response.status_code}")

            if response.status_code != 200:
                error_detail = f"状态码: {response.status_code}, 响应内容: {response.text or '空响应'}"
                logger.error(f"API请求失败: {error_detail}")
                raise Exception(f"API request failed: {error_detail}")

            return response.json()

        except requests.RequestException as e:
            logger.error(f"请求异常: {str(e)}")
            raise Exception(f"网络请求失败: {str(e)}")
        except Exception as e:
            logger.error(f"处理异常: {str(e)}")
            raise

    def _process_sync(
        self, audio_file: Union[str, Path, BinaryIO], return_full_response: bool = False, **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        处理音频文件（同步方式）

        Args:
            audio_file: 音频文件路径或文件对象
            return_full_response: 是否返回完整的API响应
            **kwargs: 传递给API的其他参数

        Returns:
            转录的文本，或完整的API响应
        """
        result = self._transcribe_sync(audio_file, **kwargs)

        if return_full_response:
            return result
        return result.get("text", "")

    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """
        转录音频文件

        Args:
            audio_file: 音频文件路径
            **kwargs: 额外参数
                - model: 可选，要使用的模型名称

        Returns:
            转录文本
        """
        try:
            # 使用 asyncio 将同步操作放入线程池
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: self._process_sync(audio_file, **kwargs))

            if isinstance(result, str):
                return result
            elif isinstance(result, dict) and "text" in result:
                return result["text"]
            else:
                return str(result)

        except Exception as e:
            logger.error(f"转录失败: {str(e)}")
            raise
