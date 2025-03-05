import abc
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProviderMeta:
    id: str
    model: str
    type: str


class AbstractProvider(abc.ABC):
    """
    抽象提供者基类，所有策略提供者的基础
    """
    
    def __init__(self, provider_config: Dict[str, Any]) -> None:
        """
        初始化提供者
        
        Args:
            provider_config: 提供者配置
        """
        self.id = provider_config.get("id")
        self.type = provider_config.get("type")
        self.model = None

    def set_model(self, model: str) -> None:
        """设置模型"""
        self.model = model

    def get_model(self) -> str:
        """获得当前使用的模型名称"""
        return self.model

    def meta(self) -> ProviderMeta:
        """获取 Provider 的元数据"""
        return ProviderMeta(
            id=self.id,
            model=self.get_model(),
            type=self.type,
        )