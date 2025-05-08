"""
ws_utils.py
存放WebSocket相关的工具函数，专注于支持语音助手功能
"""

import json
import traceback
from typing import Dict, Any, Callable, Awaitable
import time
import asyncio

from app import logger
from inspect import currentframe, getframeinfo
from starlette.websockets import WebSocketState, WebSocketDisconnect

# 类型定义
WebSocketSender = Callable[[Dict[str, Any]], Awaitable[None]]
SessionData = Dict[str, Any]


async def send_message(websocket, message: Dict[str, Any]):
    """发送消息到客户端

    Args:
        websocket: FastAPI WebSocket连接
        message: 要发送的消息字典

    Returns:
        bool: 是否成功发送消息
    """
    try:
        text = json.dumps(message)

        # 检查连接是否打开
        if hasattr(websocket, "client_state") and websocket.client_state != WebSocketState.CONNECTED:
            logger.warning(
                f"WebSocket未连接，状态为: {websocket.client_state}，尝试发送的消息类型: {message.get('type', 'unknown')}"
            )
            return False

        # 添加消息队列缓存逻辑，保证一定间隔
        if not hasattr(websocket, "_last_send_time"):
            setattr(websocket, "_last_send_time", 0)

        # 如果消息过于密集，添加小延迟防止消息阻塞
        last_send = getattr(websocket, "_last_send_time", 0)
        if time.time() - last_send < 0.05:  # 50ms的最小间隔
            await asyncio.sleep(0.01)  # 10ms延迟

        if hasattr(websocket, "_close_called"):
            setattr(websocket, "_close_called", False)  # 重置关闭标志

        # 发送消息
        await websocket.send_text(text)
        setattr(websocket, "_last_send_time", time.time())

        return True
    except WebSocketDisconnect as e:
        # 客户端已断开，记录更多细节
        frame = currentframe()
        frameinfo = getframeinfo(frame) if frame else None
        location = f"{frameinfo.filename}:{frameinfo.lineno}" if frameinfo else "unknown"
        logger.debug(f"发送消息失败：客户端已断开连接, 代码: {e.code}, 位置: {location}")
        return False
    except RuntimeError as e:
        if "Cannot call" in str(e) and "close message has been sent" in str(e):
            # 连接已经关闭，不再记录错误
            logger.debug(f"发送消息失败：连接已关闭, 错误: {str(e)}")
            return False
        elif "WebSocket is not connected" in str(e):
            # 连接未建立，不再记录错误
            logger.debug(f"发送消息失败：连接未建立, 错误: {str(e)}")
            return False
        # 其他Runtime错误
        error_trace = traceback.format_exc()
        frame = currentframe()
        frameinfo = getframeinfo(frame) if frame else None
        location = f"{frameinfo.filename}:{frameinfo.lineno}" if frameinfo else "unknown"
        logger.error(f"发送消息异常: {e}, 位置: {location}\n{error_trace}")
        return False
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"发送消息异常: {e}\n{error_trace}")
        return False
