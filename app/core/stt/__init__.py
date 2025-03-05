from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class STTProvider(ABC):
    """STT (语音转文本) 提供者抽象基类"""
    
    def __init__(self, provider_config: Dict[str, Any], provider_settings: Dict[str, Any]):
        """
        初始化STT提供者
        
        Args:
            provider_config: 提供者配置
            provider_settings: 全局设置
        """
        self.provider_config = provider_config
        self.provider_settings = provider_settings
        self.provider_name = provider_config.get("id", "unknown")
    
    @abstractmethod
    async def transcribe(self, audio_file: str, **kwargs) -> str:
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径
            **kwargs: 额外参数，如模型名称、翻译选项等
        
        Returns:
            转录文本
        """
        pass
