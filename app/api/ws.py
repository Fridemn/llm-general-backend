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
import time
import asyncio
import os

from app import logger
from app.core.ws.ws_utils import (
    send_message, process_message, cleanup_client,
    process_voice_chat_message
)
from app.core.ws.ws_tts import process_tts_message, manager as tts_manager

api_ws = APIRouter()

# 存储活跃连接和客户端会话
active_connections: Dict[str, WebSocket] = {}
client_sessions: Dict[str, Dict[str, Any]] = {}
voice_chat_connections: Dict[str, WebSocket] = {}
voice_chat_sessions: Dict[str, Dict[str, Any]] = {}
tts_connections: Dict[str, WebSocket] = {}  # 新增TTS连接存储
tts_sessions: Dict[str, Dict[str, Any]] = {}  # 新增TTS会话存储

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

@api_ws.websocket("/tts")
async def tts_endpoint(websocket: WebSocket):
    """文本转语音WebSocket接口"""
    client_id = str(uuid.uuid4())
    
    try:
        # 禁用自动关闭连接
        websocket.auto_close = False
        
        # 设置更长的WebSocket空闲超时时间
        websocket_timeout = 600  # 10分钟
        
        # 接受WebSocket连接
        await tts_manager.connect(websocket, client_id)
        
        # 发送连接成功消息
        await tts_manager.send_message(websocket, {
            "message": "TTS连接已建立", 
            "status": "connected"
        })
        
        # 处理客户端消息
        logger.info(f"开始处理TTS客户端 {client_id} 的消息")
        
        # 设置断开标志，只有在用户显式请求断开连接时才设置为True
        disconnect_requested = False
        connection_active = True
        
        # 开始心跳检测
        last_heartbeat = time.time()
        heartbeat_interval = 30  # 30秒发送一次心跳
        
        while connection_active and not disconnect_requested:
            try:
                # 计算下一次心跳前的等待时间
                wait_timeout = min(websocket_timeout, heartbeat_interval - (time.time() - last_heartbeat))
                if wait_timeout <= 0:
                    # 发送心跳
                    try:
                        await tts_manager.send_message(websocket, {"ping_check": True, "timestamp": time.time()})
                        logger.debug(f"发送心跳检测至 TTS客户端 {client_id}")
                        last_heartbeat = time.time()
                        # 继续接收消息
                        continue
                    except Exception as e:
                        logger.warning(f"发送心跳失败，客户端可能已断开: {str(e)}")
                        connection_active = False
                        break
                
                # 使用带超时的receive_text
                try:
                    # 接收消息并更新活动时间
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=wait_timeout)
                    
                    # 记录收到的消息以便调试（排除ping消息）
                    if not (len(data) < 100 and ('"ping":true' in data or '"keep_alive":true' in data)):
                        message_preview = data[:50] + ('...' if len(data) > 50 else '')
                        logger.debug(f"收到来自 {client_id} 的消息: {message_preview}")
                    
                    # 检查是否是断开连接请求
                    if len(data) < 100 and ('"disconnect":true' in data or '"disconnect": true' in data):
                        logger.info(f"TTS客户端 {client_id} 请求断开连接")
                        disconnect_requested = True
                        # 回复确认断开消息
                        await tts_manager.send_message(websocket, {
                            "message": "连接将关闭", 
                            "status": "closing"
                        })
                        break
                    
                    # 检查是否是心跳响应
                    if len(data) < 100 and ('"ping":true' in data or '"keep_alive":true' in data):
                        # 重置心跳计时
                        last_heartbeat = time.time()
                    
                    # 处理正常消息
                    await process_tts_message(client_id, websocket, data)
                    
                except asyncio.TimeoutError:
                    # 超时但未到心跳时间，继续等待
                    continue
                    
            except WebSocketDisconnect as e:
                logger.info(f"TTS客户端断开连接: {client_id}, 代码: {e.code if hasattr(e, 'code') else 'unknown'}")
                connection_active = False
                break
            except Exception as e:
                logger.error(f"处理TTS消息错误: {str(e)}", exc_info=True)
                try:
                    await tts_manager.send_message(websocket, {
                        "error": "服务器内部错误", 
                        "status": "error"
                    })
                    # 不立即断开连接，继续尝试处理后续消息
                except Exception:
                    # 如果发送失败，则可能连接已断开
                    logger.warning(f"无法发送错误消息，连接可能已断开: {client_id}")
                    connection_active = False
                    break
    except WebSocketDisconnect as e:
        logger.info(f"TTS客户端断开连接: {client_id}, 代码: {e.code if hasattr(e, 'code') else 'unknown'}")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理TTS WebSocket连接异常: {e}\n{error_trace}")
        try:
            await tts_manager.send_message(websocket, {
                "error": str(e),
                "status": "error"
            })
        except:
            pass
    finally:
        # 清理资源
        tts_manager.disconnect(websocket, client_id)
        logger.info(f"TTS客户端资源已清理: {client_id}")

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

def create_tts_session(client_id: str) -> Dict[str, Any]:
    """创建TTS会话数据"""
    return {
        "client_id": client_id,
        "processing": False,  # 标识是否正在处理TTS请求
        "last_activity": time.time()  # 添加最后活动时间
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

async def cleanup_tts_client(client_id: str):
    """清理TTS客户端资源"""
    if client_id in tts_connections:
        # 先尝试关闭连接
        try:
            await tts_connections[client_id].close()
        except Exception:
            pass
        del tts_connections[client_id]
        
    if client_id in tts_sessions:
        # 检查是否有临时文件需要清理
        try:
            temp_dir = "static/audio"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if "gsvi_tts_" in file:
                        try:
                            os.remove(os.path.join(temp_dir, file))
                            logger.debug(f"已删除临时文件: {file}")
                        except Exception as e:
                            logger.error(f"删除临时文件失败 {file}: {str(e)}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
        
        del tts_sessions[client_id]
        
    logger.info(f"TTS客户端资源已清理: {client_id}")

# 添加调试端点，帮助排查连接问题
@api_ws.get("/status")
async def api_status():
    """API状态检查端点"""
    return {
        "status": "online",
        "active_connections": len(active_connections),
        "voice_chat_connections": len(voice_chat_connections),
        "tts_connections": len(tts_manager.active_connections),  # 使用新的管理器
        "active_sessions": len(client_sessions),
        "voice_chat_sessions": len(voice_chat_sessions),
        "tts_sessions": len(tts_manager.client_sessions)  # 使用新的管理器
    }

# 启动和关闭事件
@api_ws.on_event("startup")
async def startup_event():
    logger.info("WebSocket服务启动")
    logger.info("语音识别WebSocket服务可通过 ws://[host]/ws/asr 接口访问")
    logger.info("语音聊天WebSocket服务可通过 ws://[host]/ws/voice-chat 接口访问")
    logger.info("文本转语音WebSocket服务可通过 ws://[host]/ws/tts 接口访问")  # 新增

@api_ws.on_event("shutdown")
async def shutdown_event():
    # 关闭所有会话
    for client_id in list(client_sessions.keys()):
        await cleanup_client(client_id, client_sessions, active_connections)
    for client_id in list(voice_chat_sessions.keys()):
        await cleanup_voice_chat_client(client_id)
    for client_id in list(tts_sessions.keys()):  # 新增
        await cleanup_tts_client(client_id)
    logger.info("WebSocket服务已关闭")
