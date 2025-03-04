import abc
from dataclasses import dataclass

@dataclass
class ProviderMeta:
    id: str
    model: str
    type: str


class AbstractProvider(abc.ABC):
    def __init__(self, provider_config: dict) -> None:
        super().__init__()
        self.model_name = ""
        self.provider_config = provider_config

    def set_model(self, model_name: str):
        """设置当前使用的模型名称"""
        self.model_name = model_name

    def get_model(self) -> str:
        """获得当前使用的模型名称"""
        return self.model_name

    def meta(self) -> ProviderMeta:
        """获取 Provider 的元数据"""
        return ProviderMeta(
            id=self.provider_config["id"],
            model=self.get_model(),
            type=self.provider_config["type"],
        )


class TTSProvider(AbstractProvider):
    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        super().__init__(provider_config)
        self.provider_config = provider_config
        self.provider_settings = provider_settings

    @abc.abstractmethod
    async def get_audio(self, text: str) -> str:
        """获取文本的音频，返回音频文件路径"""
        raise NotImplementedError()
