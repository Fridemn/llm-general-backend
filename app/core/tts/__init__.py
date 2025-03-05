from typing import Dict, Any
from abc import ABC, abstractmethod

class TTSProvider(ABC):
    """TTS (文本转语音) 提供者抽象基类"""
    
    def __init__(self, provider_config: Dict[str, Any], provider_settings: Dict[str, Any]):
        """
        初始化TTS提供者
        
        Args:
            provider_config: 提供者配置
            provider_settings: 全局设置
        """
        self.provider_config = provider_config
        self.provider_settings = provider_settings
        self.provider_name = provider_config.get("id", "unknown")
    
    @abstractmethod
    async def get_audio(self, text: str, **kwargs) -> str:
        """
        将文本转换为音频
        
        Args:
            text: 要转换的文本
            **kwargs: 额外参数，如语音、语速等
        
        Returns:
            音频文件路径
        """
        pass
        
    def set_model(self, model_name: str) -> None:
        """
        设置TTS模型
        
        Args:
            model_name: 模型名称
        """
        self.provider_name = model_name
