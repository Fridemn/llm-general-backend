"""
函数工具源代码包
包含所有可用的函数工具实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseFunction(ABC):
    """
    函数工具的抽象基类
    所有函数工具类都应继承此基类并实现其方法
    """

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        执行函数

        Args:
            **kwargs: 函数参数

        Returns:
            函数执行结果
        """
        pass

    @abstractmethod
    def get_definition(self) -> Dict[str, Any]:
        """
        获取函数定义

        Returns:
            包含函数名称、描述、参数等信息的字典
        """
        pass

    @property
    def need_llm(self) -> bool:
        """
        返回函数调用后是否需要LLM处理

        Returns:
            布尔值，表示是否需要LLM处理
        """
        return True

    @property
    def name(self) -> str:
        """
        返回函数名称

        Returns:
            函数名称字符串
        """
        return self.get_definition()["name"]

    @property
    def command_prefixes(self) -> List[str]:
        """
        返回用于调用此函数的命令前缀列表

        Returns:
            命令前缀字符串列表
        """
        # 默认使用函数名作为命令前缀
        return [self.name]
