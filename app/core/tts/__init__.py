import abc
from app.core.provider import AbstractProvider

class TTSProvider(AbstractProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config)
        self.provider_config = provider_config
        self.provider_settings = provider_settings

    @abc.abstractmethod
    async def get_audio(self, text: str) -> str:
        """获取文本的音频，返回音频文件路径"""
        raise NotImplementedError()
