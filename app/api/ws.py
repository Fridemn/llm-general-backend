"""
ws.py
存放与WebSocket相关的接口，如实时语音识别、声纹验证等
"""

import uuid
import json
import traceback
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse

from app import logger
from app.core.ws.ws_utils import (
    send_message, process_message, cleanup_client,
    process_voice_chat_message
)

api_ws = APIRouter()

# 存储活跃连接和客户端会话
active_connections: Dict[str, WebSocket] = {}
client_sessions: Dict[str, Dict[str, Any]] = {}
voice_chat_connections: Dict[str, WebSocket] = {}
voice_chat_sessions: Dict[str, Dict[str, Any]] = {}

@api_ws.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """语音识别WebSocket接口"""
    client_id = str(uuid.uuid4())
    
    try:
        # 接受WebSocket连接
        await websocket.accept()
        logger.info(f"WebSocket连接成功接受: {client_id}")
        active_connections[client_id] = websocket
        
        # 初始化客户端会话
        client_sessions[client_id] = create_client_session(client_id)
        
        # 发送连接成功消息
        await send_message(websocket, {
            "type": "connection",
            "client_id": client_id,
            "message": "连接成功"
        })
        
        # 处理客户端消息
        logger.info(f"开始处理客户端 {client_id} 的消息")
        while True:
            data = await websocket.receive_text()
            logger.debug(f"从客户端 {client_id} 收到消息: {data[:100]}...")
            await process_message(client_id, websocket, data, client_sessions)
            
    except WebSocketDisconnect:
        logger.info(f"客户端断开连接: {client_id}")
        # 清理资源
        await cleanup_client(client_id, client_sessions, active_connections)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理WebSocket连接异常: {e}\n{error_trace}")
        try:
            await send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            logger.error(f"无法向客户端 {client_id} 发送错误消息")
        # 清理资源
        await cleanup_client(client_id, client_sessions, active_connections)

@api_ws.websocket("/voice-chat")
async def voice_chat_endpoint(websocket: WebSocket):
    """语音聊天WebSocket接口 - 结合语音识别、LLM对话和TTS功能"""
    client_id = str(uuid.uuid4())
    
    try:
        # 接受WebSocket连接
        await websocket.accept()
        logger.info(f"语音聊天WebSocket连接成功接受: {client_id}")
        voice_chat_connections[client_id] = websocket
        
        # 初始化语音聊天会话
        voice_chat_sessions[client_id] = create_voice_chat_session(client_id)
        
        # 发送连接成功消息
        await send_message(websocket, {
            "type": "connection",
            "client_id": client_id,
            "message": "语音聊天连接成功"
        })
        
        # 处理客户端消息
        logger.info(f"开始处理语音聊天客户端 {client_id} 的消息")
        while True:
            data = await websocket.receive_text()
            logger.debug(f"从语音聊天客户端 {client_id} 收到消息: {data[:100]}...")
            await process_voice_chat_message(client_id, websocket, data, voice_chat_sessions)
            
    except WebSocketDisconnect:
        logger.info(f"语音聊天客户端断开连接: {client_id}")
        # 清理资源
        await cleanup_voice_chat_client(client_id)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理语音聊天WebSocket连接异常: {e}\n{error_trace}")
        try:
            await send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            logger.error(f"无法向语音聊天客户端 {client_id} 发送错误消息")
        # 清理资源
        await cleanup_voice_chat_client(client_id)

def create_client_session(client_id: str) -> Dict[str, Any]:
    """创建客户端会话数据"""
    return {
        "client_id": client_id,
        "asr_client": None,
        "recording_active": False,
        "recognition_results": [],
        "final_result": None
    }

def create_voice_chat_session(client_id: str) -> Dict[str, Any]:
    """创建语音聊天会话数据"""
    return {
        "client_id": client_id,
        "asr_client": None,
        "recording_active": False,
        "recognition_results": [],
        "final_result": None,
        "history_id": None,  # 用于跟踪对话历史
        "model": None,       # LLM模型
        "user_id": None,     # 用户ID
        "llm_processing_started": False  # 避免重复处理
    }

async def cleanup_voice_chat_client(client_id: str):
    """清理语音聊天客户端资源"""
    if client_id in voice_chat_connections:
        del voice_chat_connections[client_id]
        
    if client_id in voice_chat_sessions:
        session = voice_chat_sessions[client_id]
        if session.get("asr_client"):
            await session["asr_client"].close()
        del voice_chat_sessions[client_id]
        
    logger.info(f"语音聊天客户端资源已清理: {client_id}")

# 添加调试端点，帮助排查连接问题
@api_ws.get("/status")
async def api_status():
    """API状态检查端点"""
    return {
        "status": "online",
        "active_connections": len(active_connections),
        "voice_chat_connections": len(voice_chat_connections),
        "active_sessions": len(client_sessions),
        "voice_chat_sessions": len(voice_chat_sessions)
    }

# 启动和关闭事件
@api_ws.on_event("startup")
async def startup_event():
    logger.info("WebSocket服务启动")
    logger.info("语音识别WebSocket服务可通过 ws://[host]/ws/asr 接口访问")
    logger.info("语音聊天WebSocket服务可通过 ws://[host]/ws/voice-chat 接口访问")

@api_ws.on_event("shutdown")
async def shutdown_event():
    # 关闭所有会话
    for client_id in list(client_sessions.keys()):
        await cleanup_client(client_id, client_sessions, active_connections)
    for client_id in list(voice_chat_sessions.keys()):
        await cleanup_voice_chat_client(client_id)
    logger.info("WebSocket服务已关闭")
