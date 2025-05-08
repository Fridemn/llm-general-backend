import datetime
from typing import Dict, Any, List

from app.core.funcall.source import BaseFunction


class TestFunction(BaseFunction):
    """测试函数，返回简单消息"""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        简单的测试函数，返回感叹号
        """
        return {
            "success": True,
            "message": "!",
            "description": "测试成功",
            "timestamp": int(datetime.datetime.now().timestamp()),
        }

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "test_function",
            "description": "测试函数，返回一个感叹号",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "need_llm": False,
        }

    @property
    def need_llm(self) -> bool:
        return False

    @property
    def command_prefixes(self) -> List[str]:
        """支持的命令前缀"""
        return ["test", "ping"]
