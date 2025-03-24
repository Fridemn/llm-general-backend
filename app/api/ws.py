"""
ws.py
存放与WebSocket相关的接口，如实时语音识别、声纹验证等
"""

import os
import asyncio
import json
import uuid
import logging
import traceback
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import app_config, logger
from app.api.user import get_current_user
from app.core.ws.audio_websocket_client import AudioWebSocketClient

api_ws = APIRouter()

# 创建静态文件目录
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)
templates = Jinja2Templates(directory=STATIC_DIR)

# 存储活跃连接和客户端会话
active_connections: Dict[str, WebSocket] = {}
client_sessions: Dict[str, Dict[str, Any]] = {}

class RecordingRequest(BaseModel):
    """录音请求模型"""
    duration: Optional[int] = 5
    check_voiceprint: Optional[bool] = True
    batch_mode: Optional[bool] = False
    server_url: Optional[str] = "ws://127.0.0.1:8765"

@api_ws.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """语音识别WebSocket接口"""
    client_id = str(uuid.uuid4())
    
    # 优化WebSocket连接处理，添加错误日志
    try:
        await websocket.accept()
        logger.info(f"接受WebSocket连接: {client_id}")
        active_connections[client_id] = websocket
    except Exception as e:
        logger.error(f"接受WebSocket连接失败: {client_id}, 错误: {str(e)}")
        return
    
    # 创建客户端会话数据
    client_sessions[client_id] = {
        "client_id": client_id,
        "asr_client": None,
        "recording_active": False,
        "recognition_results": [],
        "final_result": None
    }
    
    try:
        # 发送连接成功消息
        await send_message(websocket, {
            "type": "connection",
            "client_id": client_id,
            "message": "连接成功"
        })
        
        # 处理客户端消息
        while True:
            data = await websocket.receive_text()
            await process_message(client_id, websocket, data)
            
    except WebSocketDisconnect:
        logger.info(f"客户端断开连接: {client_id}")
        # 清理资源
        await cleanup_client(client_id)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理WebSocket连接异常: {e}\n{error_trace}")
        try:
            await send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        # 清理资源
        await cleanup_client(client_id)

async def send_message(websocket: WebSocket, message: Dict[str, Any]):
    """发送消息到客户端"""
    try:
        await websocket.send_text(json.dumps(message))
    except Exception as e:
        logger.error(f"发送消息异常: {e}")

async def process_message(client_id: str, websocket: WebSocket, data_text: str):
    """处理客户端消息"""
    try:
        data = json.loads(data_text)
        command = data.get("command")
        
        if not command:
            await send_message(websocket, {
                "type": "error",
                "message": "缺少command字段"
            })
            return
        
        # 处理不同的命令
        if command == "start_recording":
            await handle_start_recording(client_id, websocket, data)
        elif command == "stop_recording":
            await handle_stop_recording(client_id, websocket)
        elif command == "get_voiceprints":
            await handle_get_voiceprints(client_id, websocket)
        elif command == "check_status":
            await handle_check_status(client_id, websocket)
        else:
            await send_message(websocket, {
                "type": "error",
                "message": f"未知命令: {command}"
            })
    
    except json.JSONDecodeError:
        await send_message(websocket, {
            "type": "error",
            "message": "无效的JSON格式"
        })
    except Exception as e:
        logger.error(f"处理消息异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"处理消息异常: {str(e)}"
        })

async def handle_start_recording(client_id: str, websocket: WebSocket, data: Dict[str, Any]):
    """处理开始录音命令"""
    session = client_sessions[client_id]
    
    if session["recording_active"]:
        await send_message(websocket, {
            "type": "error",
            "message": "录音已在进行中"
        })
        return
    
    # 获取参数
    duration = int(data.get("duration", 5))
    check_voiceprint = data.get("check_voiceprint", True)
    batch_mode = data.get("batch_mode", False)
    server_url = data.get("server_url", "ws://127.0.0.1:8765")
    
    try:
        # 创建ASR客户端
        asr_client = AudioWebSocketClient(server_url)
        session["asr_client"] = asr_client
        
        # 连接到语音服务器
        if await asr_client.connect():
            session["recording_active"] = True
            session["recognition_results"] = []
            session["final_result"] = None
            
            # 发送录音开始消息
            await send_message(websocket, {
                "type": "recording",
                "status": "started",
                "duration": duration,
                "batch_mode": batch_mode
            })
            
            # 启动异步录音任务
            if batch_mode:
                # 批量模式：录音完成后再识别
                asyncio.create_task(perform_batch_recording(client_id, websocket, duration, check_voiceprint))
            else:
                # 流式模式：边录边识别
                asyncio.create_task(perform_recording(client_id, websocket, duration, check_voiceprint))
        else:
            await send_message(websocket, {
                "type": "error",
                "message": "无法连接到语音服务器"
            })
            
    except Exception as e:
        logger.error(f"启动录音异常: {e}")
        if session["asr_client"]:
            await session["asr_client"].close()
            session["asr_client"] = None
            
        await send_message(websocket, {
            "type": "error",
            "message": f"启动录音异常: {str(e)}"
        })

async def handle_stop_recording(client_id: str, websocket: WebSocket):
    """处理停止录音命令"""
    session = client_sessions[client_id]
    
    if not session["recording_active"]:
        await send_message(websocket, {
            "type": "error",
            "message": "当前没有进行中的录音"
        })
        return
    
    # 标记停止录音
    session["recording_active"] = False
    
    # 发送停止消息
    await send_message(websocket, {
        "type": "recording",
        "status": "stopped"
    })

async def handle_get_voiceprints(client_id: str, websocket: WebSocket):
    """处理获取声纹列表命令"""
    try:
        # 创建临时客户端获取声纹列表
        asr_client = AudioWebSocketClient("ws://127.0.0.1:8765")
        if await asr_client.connect():
            response = await asr_client.list_voiceprints()
            
            if response.get("success") and "voiceprints" in response:
                await send_message(websocket, {
                    "type": "voiceprints",
                    "data": response["voiceprints"]
                })
            else:
                await send_message(websocket, {
                    "type": "error",
                    "message": "获取声纹列表失败"
                })
                
            # 关闭临时客户端
            await asr_client.close()
        else:
            await send_message(websocket, {
                "type": "error",
                "message": "无法连接到语音服务器"
            })
    
    except Exception as e:
        logger.error(f"获取声纹列表异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"获取声纹列表异常: {str(e)}"
        })

async def handle_check_status(client_id: str, websocket: WebSocket):
    """处理检查状态命令"""
    session = client_sessions[client_id]
    
    status = {
        "client_id": client_id,
        "recording_active": session["recording_active"],
        "has_results": len(session.get("recognition_results", [])) > 0,
        "has_final_result": session.get("final_result") is not None
    }
    
    # 如果有最终结果，一并发回前端
    if session.get("final_result"):
        await send_message(websocket, {
            "type": "recording",
            "status": "completed",
            "result": session["final_result"]
        })
    else:
        await send_message(websocket, {
            "type": "status",
            "data": status
        })

async def cleanup_client(client_id: str):
    """清理客户端资源"""
    if client_id in active_connections:
        del active_connections[client_id]
        
    if client_id in client_sessions:
        session = client_sessions[client_id]
        if session["asr_client"]:
            await session["asr_client"].close()
        del client_sessions[client_id]

async def perform_recording(client_id: str, websocket: WebSocket, duration: int, check_voiceprint: bool):
    """执行录音和识别任务"""
    session = client_sessions[client_id]
    asr_client = session["asr_client"]
    
    try:
        # 进度更新任务
        async def send_progress_updates():
            for i in range(1, duration + 1):
                if not session["recording_active"]:
                    break
                await asyncio.sleep(1)
                await send_message(websocket, {
                    "type": "recording",
                    "status": "progress",
                    "current": i,
                    "total": duration,
                    "percentage": round(i / duration * 100)
                })
        
        # 先报告正在录音
        await send_message(websocket, {
            "type": "recording",
            "status": "info",
            "message": "开始录音..."
        })
        
        # 启动进度更新任务
        progress_task = asyncio.create_task(send_progress_updates())
        
        # 录制音频
        audio_data = await asr_client.record_audio(duration)
        
        # 确保进度更新任务完成
        if not progress_task.done():
            try:
                # 等待进度更新任务完成，但最多等待0.5秒
                await asyncio.wait_for(progress_task, timeout=0.5)
            except asyncio.TimeoutError:
                # 如果超时，取消任务
                progress_task.cancel()
        
        # 报告录音完成
        await send_message(websocket, {
            "type": "recording",
            "status": "info",
            "message": "录音完成，处理中..."
        })
        
        # 如果需要验证声纹
        if check_voiceprint:
            await send_message(websocket, {
                "type": "recording",
                "status": "info",
                "message": "验证声纹中..."
            })
            
            # 发送声纹验证请求
            await asr_client.match_voiceprint(audio_data)
            # 给服务器一些处理时间
            await asyncio.sleep(0.5)
        
        # 直接识别整段音频
        await send_message(websocket, {
            "type": "recording",
            "status": "info",
            "message": "识别录音内容中..."
        })
        
        logger.info("开始识别录制的音频")
        
        # 方法1：直接从终端调用方式获取结果
        # 清除旧的结果
        asr_client.final_result = None
        asr_client.streaming_results = []
        asr_client.last_error = None
        
        try:
            # 发送识别请求
            await asr_client.recognize_audio(audio_data, check_voiceprint)
            
            # 等待足够的时间获取结果
            await asyncio.sleep(2.5)
            
            # 检查是否有结果返回
            if asr_client.final_result:
                # 使用收到的结果
                result = asr_client.final_result
                
                # 调试日志，打印完整识别结果
                logger.debug(f"完整识别结果: {json.dumps(result)}")
                
                # 检查结果中是否有错误信息
                if "error" in result:
                    # 有错误但仍然返回结果，让前端决定如何显示
                    logger.info(f"识别完成但有错误: {result.get('error')}")
                    success = True
                else:
                    success = True
                    logger.info(f"成功获取识别结果: {result.get('text', '')[:30]}...")
                    
                # 确保结果中有实际文本，不是等待消息
                if result.get("text") == "等待识别结果...":
                    # 尝试检查是否有其他识别结果可用
                    if asr_client.streaming_results:
                        combined_text = " ".join([r["text"] for r in asr_client.streaming_results])
                        result["text"] = combined_text
                        result["segments"] = asr_client.streaming_results
                        logger.info(f"更新为实际识别结果: {combined_text[:30]}...")
            else:
                # 方法2：尝试流式处理数据方案
                logger.info("尝试流式处理方案...")
                await asr_client.start_streaming(check_voiceprint)
                await asyncio.sleep(0.5)
                
                # 分段发送音频数据
                chunk_size = 8192  # 8KB
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i+chunk_size]
                    is_end = i + chunk_size >= len(audio_data)
                    await asr_client.send_audio_data(chunk, is_end)
                    await asyncio.sleep(0.01)  # 短暂延迟，避免发送过快
                
                # 等待处理完成
                await asyncio.sleep(3.0)
                
                if asr_client.final_result:
                    result = asr_client.final_result
                    success = True
                    # 即使有错误也返回结果
                    if "error" in result:
                        logger.info(f"流式处理完成但有错误: {result.get('error')}")
                    else:
                        logger.info(f"流式处理成功: {result.get('text', '')[:30]}...")
                else:
                    # 方法3：从原始音频数据创建固定格式的结果
                    logger.warning("无法获取有效识别结果，创建默认结果")
                    # 检查是否有声纹错误
                    if asr_client.last_error:
                        result = {
                            "text": "语音已识别，但用户声纹验证失败",
                            "user": asr_client.last_error.get("user", "Unknown"),
                            "error": asr_client.last_error.get("error"),
                            "similarity": asr_client.last_error.get("similarity", 0),
                            "segments": [{"text": "语音已识别，但用户声纹验证失败", "user": asr_client.last_error.get("user", "Unknown")}]
                        }
                    else:
                        result = {
                            "text": "无法识别语音内容，请重试",
                            "user": "Unknown",
                            "segments": [{"text": "无法识别语音内容，请重试", "user": "Unknown"}]
                        }
                    success = False
        
        except Exception as e:
            logger.error(f"识别处理异常: {e}")
            result = {
                "text": f"处理出错: {str(e)}",
                "user": "Error",
                "segments": [{"text": f"处理出错: {str(e)}", "user": "Error"}]
            }
            success = False
        
        # 处理最终结果
        if success:
            # 保存最终结果
            session["final_result"] = result
            
            # 确保结果中有实际文本内容
            if result.get("text") == "等待识别结果...":
                # 如果没有找到实际文本，尝试从响应中获取
                logger.warning("结果中没有实际文本，尝试替换为默认文本")
                result["text"] = "未能获取有效的识别文本，但语音已处理"
            
            # 确保segments字段存在
            if "segments" not in result or not isinstance(result["segments"], list):
                result["segments"] = [{"text": result["text"], "user": result.get("user", "Unknown")}]
            
            # 发送完成消息
            await send_message(websocket, {
                "type": "recording",
                "status": "completed",
                "result": result
            })
            logger.info(f"成功完成识别并返回结果: {result['text'][:30]}...")
        else:
            # 如果仍然有一些结果，即使失败也发送出去
            if result and "text" in result:
                await send_message(websocket, {
                    "type": "recording",
                    "status": "completed",
                    "result": result,
                    "warning": True
                })
                logger.info(f"返回有警告的结果: {result['text'][:30]}...")
            else:
                error_msg = "未能获取有效识别结果"
                await send_message(websocket, {
                    "type": "error",
                    "message": f"语音识别失败: {error_msg}"
                })
                logger.error(f"语音识别失败: {error_msg}")
    
    except Exception as e:
        logger.error(f"录音过程异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"录音过程异常: {str(e)}"
        })
    
    finally:
        # 清理资源
        session["recording_active"] = False
        if asr_client:
            await asr_client.close()
            session["asr_client"] = None
        logger.info("识别会话已完成并清理")

async def perform_batch_recording(client_id: str, websocket: WebSocket, duration: int, check_voiceprint: bool):
    """执行批量录音和识别任务"""
    session = client_sessions[client_id]
    asr_client = session["asr_client"]
    
    try:
        # 进度更新任务
        async def send_progress_updates():
            for i in range(1, duration + 1):
                if not session["recording_active"]:
                    break
                await asyncio.sleep(1)
                await send_message(websocket, {
                    "type": "recording",
                    "status": "progress",
                    "current": i,
                    "total": duration,
                    "percentage": round(i / duration * 100)
                })
        
        # 通知客户端开始录音
        await send_message(websocket, {
            "type": "recording",
            "status": "info",
            "message": "录音中，完成后将进行整体识别..."
        })
        
        # 开始录制音频
        logger.info(f"开始批量录制 {duration} 秒音频")
        
        # 启动进度更新任务
        progress_task = asyncio.create_task(send_progress_updates())
        
        # 直接调用录音函数，录制整段音频
        audio_data = await asr_client.record_audio(duration)
        logger.info("录音完成，开始一次性识别")
        
        # 等待进度更新任务完成
        if not progress_task.done():
            await asyncio.wait_for(progress_task, timeout=0.5)
        
        # 通知客户端录音完成，开始识别
        await send_message(websocket, {
            "type": "recording", 
            "status": "info", 
            "message": "录音完成，开始识别..."
        })
        
        # 如果需要检查声纹
        if check_voiceprint:
            await send_message(websocket, {
                "type": "recording",
                "status": "info",
                "message": "进行声纹验证..."
            })
            
            # 发送声纹匹配请求
            await asr_client.match_voiceprint(audio_data)
            
            # 给服务器一些处理时间
            await asyncio.sleep(0.5)
            
            await send_message(websocket, {
                "type": "recording",
                "status": "info",
                "message": "声纹验证完成，开始识别语音内容..."
            })
        
        # 一次性识别音频
        logger.info("开始识别录音内容")
        
        # 发送识别请求
        await asr_client.recognize_audio(audio_data, check_voiceprint)
        
        # 等待服务器处理并返回结果
        await asyncio.sleep(2.0)
        
        # 获取结果
        if asr_client.final_result:
            result = asr_client.final_result
            
            # 保存最终结果
            session["final_result"] = result
            
            # 确保结果中有实际文本内容
            if result.get("text") == "等待识别结果...":
                logger.warning("结果中没有实际文本，尝试替换为默认文本")
                result["text"] = "未能获取有效的识别文本，但语音已处理"
            
            # 为了保持一致性，制作一个segments格式，但只包含一个段落
            if "segments" not in result:
                result["segments"] = [{"text": result["text"], "user": result.get("user", "unknown")}]
            
            # 发送完成消息
            await send_message(websocket, {
                "type": "recording",
                "status": "completed",
                "result": result
            })
        else:
            error = "未收到识别结果"
            await send_message(websocket, {
                "type": "error",
                "message": f"语音识别失败: {error}"
            })
    
    except Exception as e:
        logger.error(f"批量录音识别异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"批量录音识别异常: {str(e)}"
        })
    
    finally:
        # 清理资源
        session["recording_active"] = False
        if asr_client:
            await asr_client.close()
            session["asr_client"] = None

@api_ws.get("/status")
async def api_status():
    """API状态检查端点"""
    return {
        "status": "online",
        "active_connections": len(active_connections),
        "active_sessions": len(client_sessions)
    }

# 启动和关闭事件
@api_ws.on_event("startup")
async def startup_event():
    logger.info("语音识别WebSocket服务启动")
    logger.info("WebSocket服务可通过 ws://[host]/ws/asr 接口访问")
    logger.info("前端界面可独立部署，使用static/index.html")

@api_ws.on_event("shutdown")
async def shutdown_event():
    # 关闭所有会话
    for client_id in list(client_sessions.keys()):
        await cleanup_client(client_id)
    logger.info("语音识别WebSocket服务已关闭")
