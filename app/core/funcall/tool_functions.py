from typing import Dict, Any, List, Optional, Type
import importlib
import inspect
import os
import pkgutil
import sys

from app import logger
from app.core.funcall.source import BaseFunction
from app.core.funcall.source.date_functions import CurrentDateFunction, DateDifferenceFunction
from app.core.funcall.source.test_functions import TestFunction

# 实例化所有功能类
function_instances = {
    "get_current_date": CurrentDateFunction(),
    "calculate_date_difference": DateDifferenceFunction(),
    "test_function": TestFunction()
}

# 创建前缀到函数实例的映射，用于快速查找命令
prefix_to_function = {}
for func_name, instance in function_instances.items():
    for prefix in instance.command_prefixes:
        prefix_to_function[prefix.lower()] = instance

# 生成可用函数定义列表
AVAILABLE_FUNCTIONS = [
    func.get_definition() for func in function_instances.values()
]

# 函数名到need_llm标志的映射
FUNCTION_NEED_LLM = {
    func_name: func.need_llm for func_name, func in function_instances.items()
}

def get_available_functions() -> List[Dict[str, Any]]:
    """获取可用函数定义"""
    return AVAILABLE_FUNCTIONS

def find_function_by_command(command: str) -> Optional[BaseFunction]:
    """
    根据命令前缀查找对应的函数实例
    
    Args:
        command: 命令前缀
        
    Returns:
        对应的函数实例，如果未找到则返回None
    """
    return prefix_to_function.get(command.lower())

def execute_function(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行指定的函数
    
    Args:
        function_name: 要执行的函数名称
        arguments: 函数参数
        
    Returns:
        函数执行结果
    """
    if function_name not in function_instances:
        return {
            "success": False,
            "error": f"未找到函数: {function_name}"
        }
    
    try:
        # 执行对应的函数
        result = function_instances[function_name].execute(**arguments)
        return result
    except Exception as e:
        logger.error(f"执行函数 {function_name} 时出错: {str(e)}")
        return {
            "success": False,
            "error": f"函数执行失败: {str(e)}"
        }

def get_function_need_llm(function_name: str) -> bool:
    """
    获取指定函数是否需要LLM处理
    
    Args:
        function_name: 函数名称
        
    Returns:
        是否需要LLM处理
    """
    return FUNCTION_NEED_LLM.get(function_name, True)  # 默认需要LLM处理
