import abc
from typing import Union, Dict, Any, BinaryIO
from pathlib import Path
from app.core.provider import AbstractProvider

class STTProvider(AbstractProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config)
        self.provider_config = provider_config
        self.provider_settings = provider_settings

    @abc.abstractmethod
    async def process_audio(self, audio_file: Union[str, Path, BinaryIO], return_full_response: bool = False, **kwargs) -> Union[str, Dict[str, Any]]:
        """处理音频文件，返回转换后的文本或完整响应"""
        raise NotImplementedError()
