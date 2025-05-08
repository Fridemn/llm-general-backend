import json
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from app import logger
from app.core.funcall.tool_functions import (
    get_available_functions,
    execute_function,
    get_function_need_llm,
    find_function_by_command,
)
from app.core.llm.message import LLMMessage, MessageComponent, MessageType, MessageRole, MessageSender


class FunctionHandler:
    """
    函数调用处理器
    """

    # 命令前缀标识符
    COMMAND_PREFIXES = ["/", "!", "！"]
    # 跳过LLM处理的前缀
    RAW_MODE_PREFIX = "raw"

    @staticmethod
    def get_available_functions() -> List[Dict[str, Any]]:
        """获取可用函数定义列表"""
        return get_available_functions()

    @staticmethod
    def handle_function_call(function_call: Dict[str, Any]) -> Tuple[str, Dict[str, Any], bool]:
        """
        处理函数调用并返回结果

        Args:
            function_call: 函数调用信息，包含name和arguments

        Returns:
            函数名、执行结果和是否需要LLM处理的元组
        """
        try:
            function_name = function_call.get("name", "")
            arguments_str = function_call.get("arguments", "{}")
            need_llm_override = function_call.get("need_llm")

            # 解析参数
            try:
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                elif isinstance(arguments_str, dict):
                    arguments = arguments_str
                else:
                    arguments = {}
                    logger.error(f"无法处理函数参数类型: {type(arguments_str)}")
            except json.JSONDecodeError:
                arguments = {}
                logger.error(f"无法解析函数参数: {arguments_str}")

            # 执行函数
            result = execute_function(function_name, arguments)

            # 确定是否需要LLM处理
            if need_llm_override is not None:
                need_llm = need_llm_override
            else:
                need_llm = get_function_need_llm(function_name)

            return function_name, result, need_llm

        except Exception as e:
            logger.error(f"处理函数调用失败: {str(e)}")
            return "", {"error": f"处理函数调用失败: {str(e)}"}, True

    @staticmethod
    def create_function_message(history_id: str, function_name: str, result: Dict[str, Any]) -> LLMMessage:
        """
        创建包含函数执行结果的消息对象

        Args:
            history_id: 历史ID
            function_name: 函数名称
            result: 函数执行结果

        Returns:
            LLMMessage对象
        """
        # 将结果转为JSON字符串
        result_str = json.dumps(result, ensure_ascii=False)

        # 创建文本组件
        component = MessageComponent(type=MessageType.TEXT, content=result_str, extra={"function_name": function_name})

        # 创建函数消息
        return LLMMessage(
            history_id=history_id,
            sender=MessageSender(role=MessageRole.SYSTEM),
            components=[component],
            message_str=f"函数 {function_name} 的结果: {result_str}",
        )

    @staticmethod
    def parse_command(text: str) -> Tuple[str, Optional[str], List[str]]:
        """
        解析命令文本，提取命令名称和参数

        Args:
            text: 用户输入文本

        Returns:
            (解析后的文本, 命令名称, 参数列表)元组；如果不是命令，命令名称为None
        """
        if not text or not text.strip():
            return text, None, []

        text = text.strip()
        is_command = False

        # 检查是否以命令前缀开始
        for prefix in FunctionHandler.COMMAND_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                is_command = True
                break

        if not is_command:
            return text, None, []

        # 将文本分解为命令和参数
        parts = text.split()
        if not parts:
            return text, None, []

        command = parts[0].lower()
        args = parts[1:]

        return text, command, args

    @staticmethod
    def detect_function_call_intent(text: str) -> Optional[Dict[str, Any]]:
        """
        检测是否有函数调用意图（例如命令式调用）

        Args:
            text: 用户输入的文本

        Returns:
            检测到的函数调用，如无则返回None
        """
        if not text:
            return None

        # 解析命令
        raw_text, command, args = FunctionHandler.parse_command(text)
        if not command:
            return None

        # 检查是否需要跳过LLM处理
        skip_llm = False
        if command.lower() == FunctionHandler.RAW_MODE_PREFIX:
            skip_llm = True
            # 移除raw前缀，重新解析
            if args:
                remaining_text = " ".join(args)
                _, command, args = FunctionHandler.parse_command(remaining_text)
                if not command:  # 如果没有命令，则退出
                    return None
            else:
                return None

        # 查找对应的函数
        function_instance = find_function_by_command(command)
        if not function_instance:
            return None

        # 获取函数的名称和参数
        function_name = function_instance.name

        # 对于特定函数进行参数处理
        arguments = {}
        if function_name == "calculate_date_difference" and args:
            arguments["start_date"] = args[0]
            if len(args) > 1:
                arguments["end_date"] = args[1]

        return {
            "name": function_name,
            "arguments": json.dumps(arguments) if arguments else "{}",
            "need_llm": not skip_llm and function_instance.need_llm,
        }


# 全局函数处理器实例
function_handler = FunctionHandler()
