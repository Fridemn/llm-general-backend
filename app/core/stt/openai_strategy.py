import os
import requests
import json
import tempfile
import logging
from enum import Enum
from typing import Optional, Dict, Any, BinaryIO, Union
import base64
from pathlib import Path
import asyncio
from . import STTProvider

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STTProcessor")

class STTModel(str, Enum):
    """OpenAI支持的语音转文字模型枚举"""
    WHISPER_1 = "whisper-1"


class STTStrategy(str, Enum):
    """语音转文字策略枚举"""
    STANDARD = "standard"  # 标准转录
    TRANSLATE = "translate"  # 翻译为英语


class STTProcessor:
    """
    语音转文字处理器，使用OpenAI的API
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        strategy: STTStrategy = STTStrategy.STANDARD
    ):
        """
        初始化语音转文字处理器
        
        Args:
            api_key: OpenAI API密钥，如果为None则尝试从环境变量中获取
            api_base: OpenAI API基础URL，如果为None则使用默认URL
            model: 使用的模型
            strategy: 转录策略
        """
        # 使用传入的api_key参数，而不是硬编码的值
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as environment variable OPENAI_API_KEY")
        
        self.api_base = api_base or "https://aihubmix.com/v1"
        # 确保api_base不以斜杠结尾
        if self.api_base.endswith("/"):
            self.api_base = self.api_base[:-1]

        self.model = model
        self.strategy = strategy
        
        logger.info(f"初始化STT处理器 - API基础URL: {self.api_base}, 模型: {self.model}, 策略: {self.strategy}")
        
    def set_api_key(self, api_key: str) -> None:
        """设置API密钥"""
        self.api_key = api_key
        
    def set_api_base(self, api_base: str) -> None:
        """设置API基础URL"""
        self.api_base = api_base
        
    def set_model(self, model: STTModel) -> None:
        """设置模型"""
        self.model = model
        
    def set_strategy(self, strategy: STTStrategy) -> None:
        """设置转录策略"""
        self.strategy = strategy
    
    def transcribe(self, audio_file: Union[str, Path, BinaryIO], **kwargs) -> Dict[str, Any]:
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径或文件对象
            **kwargs: 传递给API的其他参数
            
        Returns:
            API返回的结果
        """
        endpoint = f"{self.api_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        logger.info(f"开始转录，使用端点: {endpoint}")
        # 检查文件是否存在
        if isinstance(audio_file, (str, Path)) and not os.path.exists(audio_file):
            raise FileNotFoundError(f"找不到音频文件: {audio_file}")

        try:
            # 准备文件
            if isinstance(audio_file, (str, Path)):
                with open(audio_file, "rb") as f:
                    logger.info(f"发送文件: {audio_file}")
                    files = {"file": f}
                    data = {
                        "model": self.model,
                        **kwargs
                    }
                    response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            else:
                # 假设audio_file是一个已打开的文件对象
                files = {"file": audio_file}
                data = {
                    "model": self.model,
                    **kwargs
                }
                response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            
            logger.info(f"API响应状态码: {response.status_code}")
            
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
    
    def translate(self, audio_file: Union[str, Path, BinaryIO], **kwargs) -> Dict[str, Any]:
        """
        将音频文件翻译为英语
        
        Args:
            audio_file: 音频文件路径或文件对象
            **kwargs: 传递给API的其他参数
            
        Returns:
            API返回的结果
        """
        endpoint = f"{self.api_base}/audio/translations"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        logger.info(f"开始翻译，使用端点: {endpoint}")
        # 检查文件是否存在
        if isinstance(audio_file, (str, Path)) and not os.path.exists(audio_file):
            raise FileNotFoundError(f"找不到音频文件: {audio_file}")

        try:
            # 准备文件
            if isinstance(audio_file, (str, Path)):
                with open(audio_file, "rb") as f:
                    logger.info(f"发送文件: {audio_file}")
                    files = {"file": f}
                    data = {
                        "model": self.model,
                        **kwargs
                    }
                    response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            else:
                # 假设audio_file是一个已打开的文件对象
                files = {"file": audio_file}
                data = {
                    "model": self.model,
                    **kwargs
                }
                response = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            
            logger.info(f"API响应状态码: {response.status_code}")
            
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
    
    def process(self, audio_file: Union[str, Path, BinaryIO], return_full_response: bool = False, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        根据策略处理音频文件
        
        Args:
            audio_file: 音频文件路径或文件对象
            return_full_response: 是否返回完整的API响应
            **kwargs: 传递给API的其他参数
            
        Returns:
            转录或翻译的文本，或完整的API响应
        """
        if self.strategy == STTStrategy.STANDARD:
            result = self.transcribe(audio_file, **kwargs)
        elif self.strategy == STTStrategy.TRANSLATE:
            result = self.translate(audio_file, **kwargs)
        else:
            raise ValueError(f"不支持的策略: {self.strategy}")
        
        if return_full_response:
            return result
        return result.get("text", "")


# 辅助函数
def audio_bytes_to_file(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    将音频字节转换为临时文件
    
    Args:
        audio_bytes: 音频字节
        suffix: 文件后缀
        
    Returns:
        临时文件路径
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(audio_bytes)
        return f.name


def base64_to_audio(base64_string: str) -> bytes:
    """
    将Base64字符串转换为音频字节
    
    Args:
        base64_string: Base64编码的字符串
        
    Returns:
        音频字节
    """
    return base64.b64decode(base64_string)


class ProviderOpenAISTT(STTProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config, provider_settings)
        
        # 从配置中获取API密钥和基础URL
        api_key = provider_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
        api_base = provider_config.get("api_base")
        model = provider_config.get("model", STTModel.WHISPER_1)
        strategy = provider_config.get("strategy", STTStrategy.STANDARD)
        
        # 创建内部处理器
        self.processor = STTProcessor(
            api_key=api_key,
            api_base=api_base,
            model=model,
            strategy=strategy
        )
        
        self.set_model("openai_stt")
    
    async def process_audio(self, audio_file: Union[str, Path, BinaryIO], return_full_response: bool = False, **kwargs) -> Union[str, Dict[str, Any]]:
        """异步处理音频文件，将处理委托给内部的STTProcessor实例"""
        # 使用run_in_executor来异步运行同步代码
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.processor.process(
                audio_file, 
                return_full_response=return_full_response, 
                **kwargs
            )
        )
