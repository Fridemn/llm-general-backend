import datetime
from typing import Dict, Any, Optional, List

from app import logger
from app.core.funcall.source import BaseFunction


class CurrentDateFunction(BaseFunction):
    """获取当前日期信息的函数"""

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        获取当前日期和时间信息
        """
        try:
            now = datetime.datetime.now()
            return {
                "success": True,
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "weekday_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()],
                "year": now.year,
                "month": now.month,
                "day": now.day,
                "timestamp": int(now.timestamp()),
            }
        except Exception as e:
            logger.error(f"获取日期时出错: {str(e)}")
            return {"success": False, "error": f"获取日期出错: {str(e)}"}

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "get_current_date",
            "description": "获取当前的日期和时间信息",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "need_llm": True,
        }

    @property
    def need_llm(self) -> bool:
        return True

    @property
    def command_prefixes(self) -> List[str]:
        """支持的命令前缀"""
        return ["date", "today", "now", "time"]


class DateDifferenceFunction(BaseFunction):
    """计算日期差异的函数"""

    def execute(self, start_date: str, end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        计算两个日期之间的差异

        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD，如不提供则使用当前日期

        Returns:
            包含计算结果的字典
        """
        try:
            start = datetime.datetime.strptime(start_date, "%Y-%m-%d")

            if end_date:
                end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end = datetime.datetime.now()
                end_date = end.strftime("%Y-%m-%d")

            diff = end - start

            return {
                "success": True,
                "days": diff.days,
                "seconds": diff.seconds,
                "total_seconds": int(diff.total_seconds()),
                "start_date": start_date,
                "end_date": end_date,
            }
        except Exception as e:
            logger.error(f"计算日期差异时出错: {str(e)}")
            return {"success": False, "error": f"计算日期差异出错: {str(e)}"}

    def get_definition(self) -> Dict[str, Any]:
        return {
            "name": "calculate_date_difference",
            "description": "计算两个日期之间的差异，如果不提供结束日期则使用当前日期",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "起始日期，格式为YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期，格式为YYYY-MM-DD，可选参数"},
                },
                "required": ["start_date"],
            },
            "need_llm": True,
        }

    @property
    def need_llm(self) -> bool:
        return True

    @property
    def command_prefixes(self) -> List[str]:
        """支持的命令前缀"""
        return ["diff", "datediff", "date-diff"]
