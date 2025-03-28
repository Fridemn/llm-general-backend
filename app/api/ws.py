"""
ws.py
存放与WebSocket相关的接口，主要是语音助手功能
"""

import uuid
import json
import traceback
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import time
import asyncio
import os
from inspect import currentframe, getframeinfo
from collections import deque
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app import logger
from app.core.ws.ws_utils import send_message, process_voice_chat_message

api_ws = APIRouter()

# 在api_ws路由之前添加CORS配置
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 语音助手连接和会话存储
voice_assistant_connections: Dict[str, WebSocket] = {}
voice_assistant_sessions: Dict[str, Dict[str, Any]] = {}

@api_ws.websocket("/voice-assistant")
async def voice_assistant_endpoint(websocket: WebSocket):
    """语音助手WebSocket接口 - 将语音输入转为LLM文字回复，再转为语音输出"""
    client_id = str(uuid.uuid4())
    
    try:
        # 删除多余的accept调用，只保留一次
        await websocket.accept()
        
        # 禁用自动关闭连接
        websocket.auto_close = False
        
        logger.info(f"语音助手WebSocket连接成功接受: {client_id}")
        voice_assistant_connections[client_id] = websocket
        
        # 初始化语音助手会话
        voice_assistant_sessions[client_id] = create_voice_assistant_session(client_id)
        
        # 发送连接成功消息，增加延迟确保消息发送成功
        await asyncio.sleep(0.1)
        await send_message(websocket, {
            "type": "connection",
            "client_id": client_id,
            "message": "语音助手连接成功",
            "timestamp": time.time()
        })
        
        # 记录上次接收消息的时间
        last_activity = time.time()
        
        # 处理客户端消息
        logger.info(f"开始处理语音助手客户端 {client_id} 的消息")
        while True:
            try:
                # 在接收前记录连接状态
                conn_state = getattr(websocket, "client_state", None)
                logger.debug(f"[DIAG] 准备接收消息，客户端状态: {conn_state}, 连接ID: {client_id}")
                
                # 使用带超时的receive_text
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 发送保活消息
                    if await send_message(websocket, {"type": "ping", "timestamp": time.time()}):
                        continue
                    else:
                        logger.warning(f"保活消息发送失败，客户端 {client_id} 可能已断开")
                        break

                # 更新活动时间
                last_activity = time.time()
                
                # 记录接收到的消息
                is_ping = '"ping":true' in data and len(data) < 100
                if not is_ping:  # 只记录非ping消息
                    logger.debug(f"从语音助手客户端 {client_id} 收到消息: {data[:100]}...")
                
                # 更新会话的最后活动时间
                voice_assistant_sessions[client_id]["last_activity"] = time.time()
                
                # 检查连接状态
                if hasattr(websocket, "client_state"):
                    logger.debug(f"[DIAG] 消息接收后连接状态: {websocket.client_state}, 客户端ID: {client_id}")
                
                # 处理消息
                await process_voice_assistant_message(client_id, websocket, data, voice_assistant_sessions)
                
            except WebSocketDisconnect as e:
                # 记录更详细的断开信息
                frame = currentframe()
                frameinfo = getframeinfo(frame) if frame else None
                location = f"{frameinfo.filename}:{frameinfo.lineno}" if frameinfo else "unknown"
                logger.info(f"语音助手客户端断开连接: {client_id}, 代码: {e.code}, 位置: {location}")
                break
            
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"处理语音助手消息异常: {e}\n{error_trace}")
                # 尝试发送错误消息，如果失败则跳出循环
                try:
                    await send_message(websocket, {
                        "type": "error",
                        "message": f"处理消息异常: {str(e)}"
                    })
                except:
                    logger.warning("无法发送错误消息，客户端可能已断开")
                    break
    
    except WebSocketDisconnect as e:
        # 记录更详细的断开信息
        frame = currentframe()
        frameinfo = getframeinfo(frame) if frame else None
        location = f"{frameinfo.filename}:{frameinfo.lineno}" if frameinfo else "unknown"
        logger.info(f"语音助手客户端断开连接: {client_id}, 代码: {e.code}, 位置: {location}")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理语音助手WebSocket连接异常: {e}\n{error_trace}")
        try:
            await send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            # 忽略发送错误时的异常
            pass
    finally:
        # 清理资源前记录状态
        logger.debug(f"[DIAG] 准备清理资源，客户端ID: {client_id}, 连接状态: {getattr(websocket, 'client_state', 'unknown')}")
        # 清理资源
        await cleanup_voice_assistant_client(client_id)

async def process_voice_assistant_message(client_id: str, websocket: WebSocket, data: str, sessions: Dict[str, Dict[str, Any]]):
    """处理语音助手消息，整合ASR、LLM和TTS功能"""
    session = sessions.get(client_id)
    if not session:
        logger.warning(f"未找到语音助手会话: {client_id}")
        return
    
    try:
        message = json.loads(data)
        command = message.get("command", "")
        message_type = message.get("type", "")  # 兼容旧格式
        
        # 初始化TTS队列
        if not session.get("tts_queue"):
            session["tts_queue"] = deque()
            session["tts_task"] = None

        # 处理心跳消息
        if message.get("ping") is True or message.get("keep_alive") is True:
            # 记录保活消息
            is_playback = message.get("playback_active", False)
            playback_status = ""
            
            # 获取更多播放状态信息
            if is_playback:
                if message.get("playback_starting"):
                    playback_status = " (开始播放)"
                elif message.get("playback_started"):
                    playback_status = f" (播放已开始，时长：{message.get('duration', '未知')}秒)"
                elif message.get("playback_loaded"):
                    playback_status = " (音频已加载)"
            
            logger_fn = logger.info if is_playback else logger.debug
            logger_fn(f"[DIAG] 收到{' 音频播放中的' if is_playback else ''}保活消息{playback_status}，客户端ID: {client_id}")
            
            # 更新会话状态和活动时间
            session["last_activity"] = time.time()
            if is_playback:
                # 记录播放状态
                session["playback_active"] = True
                session["playback_timestamp"] = time.time()
                if message.get("duration"):
                    session["playback_duration"] = message["duration"]
            
            # 发送确认响应
            await send_message(websocket, {
                "pong": True, 
                "timestamp": time.time(),
                "server_status": "active",
                "server_message": "接收到保活信号"
            })
            return
            
        # 处理播放完成通知
        if command == "playback_complete":
            logger.info(f"客户端 {client_id} 通知音频播放已完成")
            
            # 清理会话中的播放状态标记
            for key in ["playback_active", "playback_timestamp", "playback_duration"]:
                if key in session:
                    session.pop(key)
                    
            # 发送确认消息
            await send_message(websocket, {
                "type": "playback_ack", 
                "message": "服务器已收到播放完成通知",
                "status": "ok"
            })
            return
            
        # 处理播放错误通知
        if command == "playback_error":
            logger.warning(f"客户端 {client_id} 报告音频播放错误: {message.get('error', '未知错误')}")
            
            # 清理会话中的播放状态标记
            for key in ["playback_active", "playback_timestamp", "playback_duration"]:
                if key in session:
                    session.pop(key)
            
            # 确认收到错误报告
            await send_message(websocket, {
                "type": "playback_ack", 
                "message": "服务器已收到播放错误通知",
                "status": "error_acknowledged"
            })
            return
            
        # 处理心跳消息
        if message.get("ping") is True:
            # 记录更多上下文信息
            conn_state = getattr(websocket, "client_state", "unknown")
            logger.debug(f"[DIAG] 收到心跳消息，客户端ID: {client_id}, 连接状态: {conn_state}, 时间戳: {message.get('timestamp', 'none')}")
            
            if not (hasattr(websocket, "_last_ping_time") and 
                    time.time() - getattr(websocket, "_last_ping_time", 0) < 1.0):
                # 避免在1秒内处理多个ping
                setattr(websocket, "_last_ping_time", time.time())
                logger.debug(f"收到来自 {client_id} 的心跳消息")
                
                # 发送前检查连接状态
                conn_state = getattr(websocket, "client_state", "unknown")
                logger.debug(f"[DIAG] 发送pong前连接状态: {conn_state}, 客户端ID: {client_id}")
                
                await send_message(websocket, {"pong": True, "timestamp": time.time()})
                
                # 发送后再次检查连接状态
                conn_state = getattr(websocket, "client_state", "unknown")
                logger.debug(f"[DIAG] 发送pong后连接状态: {conn_state}, 客户端ID: {client_id}")
            else:
                logger.debug(f"[DIAG] 忽略频繁心跳消息，客户端ID: {client_id}")
            return
            
        # 处理保活消息
        if message.get("keep_alive") is True or message_type == "keep_alive":
            logger.debug(f"收到来自 {client_id} 的保活消息")
            await send_message(websocket, {"pong": True, "timestamp": time.time()})
            return
            
        # 处理断开连接请求
        if command == "disconnect":
            logger.info(f"客户端 {client_id} 请求断开连接")
            await send_message(websocket, {
                "type": "disconnect_ack",
                "message": "服务器确认断开请求"
            })
            raise WebSocketDisconnect(code=1000, reason="Client requested disconnection")
        
        # 处理连接信息
        if command == "connection_info":
            logger.info(f"收到客户端 {client_id} 连接信息: {message.get('client', 'unknown')} {message.get('version', 'unknown')}")
            session["client_info"] = {
                "client": message.get("client", "unknown"),
                "version": message.get("version", "unknown"),
                "user_agent": message.get("user_agent", "unknown")
            }
            await send_message(websocket, {
                "type": "connection_info_received",
                "message": "服务器已收到连接信息"
            })
            return
        
        # 处理配置命令
        if command == "set_params" or message_type == "config":
            # 保存模型参数
            if "model" in message:
                session["model"] = message["model"]
            if "history_id" in message:
                session["history_id"] = message["history_id"]
            if "user_id" in message:
                session["user_id"] = message["user_id"]
                
            # 保存TTS相关参数
            if "provider" in message:
                session["provider"] = message["provider"]
            if "api_base" in message:
                session["api_base"] = message["api_base"]
            if "character" in message:
                session["character"] = message["character"]
            if "emotion" in message:
                session["emotion"] = message["emotion"]
                
            # 发送配置成功响应
            await send_message(websocket, {
                "type": "params",
                "status": "updated"
            })
            return
            
        # 处理开始语音聊天命令
        elif command == "start_voice_chat":
            # 重置LLM响应缓冲区
            session["llm_response_buffer"] = []
            session["llm_response_complete"] = False
            # 委托给语音聊天处理函数
            await process_voice_chat_message(client_id, websocket, data, sessions)
            return
        
        # 处理停止录音命令    
        elif command == "stop_recording":
            # 委托给语音聊天处理函数
            await process_voice_chat_message(client_id, websocket, data, sessions)
            return
            
        # 处理旧格式的语音数据
        elif message_type == "audio_data":
            # 将消息格式转换为command格式
            new_data = json.dumps({
                "command": "send_audio",
                "audio": message.get("audio"),
                "encoding": message.get("encoding", "wav")
            })
            # 委托给语音聊天处理函数
            await process_voice_chat_message(client_id, websocket, new_data, sessions)
            return
            
        # 处理旧格式的识别结束命令
        elif message_type == "recognition_end":
            # 将消息格式转换为command格式
            new_data = json.dumps({
                "command": "stop_recording"
            })
            # 委托给语音聊天处理函数
            await process_voice_chat_message(client_id, websocket, new_data, sessions)
            return
            
        # 处理流式输出LLM完成
        elif command == "llm_stream_end" or message_type == "llm_stream_end":
            # 获取完整文本
            text = message.get("text", "") or "".join(session.get("llm_response_buffer", []))
            
            if not text:
                await send_message(websocket, {
                    "type": "error",
                    "message": "未收集到LLM响应文本"
                })
                return
                
            # 只记录完成消息，不进行额外的TTS处理
            logger.info(f"LLM流式响应完成，总字符数: {len(text)}")
            
            # 仅发送完成通知
            await send_message(websocket, {
                "type": "llm_stream_complete",
                "text": text,
                "history_id": message.get("history_id"),
                "message_id": message.get("message_id")
            })
            return

        # 处理完整句子
        elif command == "llm_sentence_complete" or message_type == "llm_sentence_complete":
            text = message.get("text", "")
            if text:
                # 将句子加入TTS处理队列
                session["tts_queue"].append(text)
                
                # 如果没有正在运行的TTS任务，启动一个
                if not session.get("tts_task") or session["tts_task"].done():
                    session["tts_task"] = asyncio.create_task(
                        process_tts_queue(client_id, websocket, session)
                    )
            return
            
        else:
            logger.warning(f"未知的命令或消息类型: {command or message_type}")
            await send_message(websocket, {
                "type": "error",
                "message": f"未知的命令: {command or message_type}"
            })
            
    except json.JSONDecodeError:
        logger.error(f"无效的JSON数据: {data[:100]}...")
        await send_message(websocket, {
            "type": "error",
            "message": "无效的JSON数据"
        })
    except WebSocketDisconnect as e:
        # 检查是否在播放音频中断开
        if session.get("playback_active", False):
            logger.warning(f"客户端 {client_id} 在音频播放期间断开连接，代码: {e.code}")
        else:
            # 记录普通断开
            logger.info(f"客户端 {client_id} 主动断开连接，代码: {e.code}")
        raise  # 重新抛出异常，让上层处理
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理语音助手消息异常: {e}\n{error_trace}")
        await send_message(websocket, {
            "type": "error",
            "message": str(e)
        })

async def send_to_tts(client_id: str, websocket: WebSocket, text: str, session: Dict[str, Any]):
    """将文本发送给TTS服务并将音频返回给客户端"""
    try:
        # 文本去重处理
        text = remove_duplicate_text(text)
        logger.info(f"发送文本到TTS服务(去重后): {text[:50]}...")
        
        # 设置会话标志，表明我们正在处理TTS
        session["tts_processing"] = True
        
        # 为防止网络不稳定导致连接关闭，先发送一个心跳保持连接
        await asyncio.sleep(0.1)  # 短暂暂停，确保消息发送顺序正确
        
        # 通知客户端TTS处理开始
        if not await send_message(websocket, {
            "type": "tts_start",
            "message": "开始生成语音"
        }):
            logger.warning(f"发送TTS开始消息失败，客户端可能已断开连接")
            session.pop("tts_processing", None)
            return
        
        # 禁用保持连接机制的超时时间
        # 设置session标记表明连接应该永久保持
        session["permanent_connection"] = True
        session["never_close"] = True
        
        # 这里处理时间可能较长，可能导致连接超时，添加保活机制
        tts_start_time = time.time()
        
        # 启动一个任务定期发送保活消息
        async def send_keepalive():
            while "tts_processing" in session:
                try:
                    await send_message(websocket, {
                        "type": "tts_processing",
                        "timestamp": time.time(),
                        "elapsed": time.time() - tts_start_time
                    })
                    await asyncio.sleep(2)  # 每2秒发送一次保活消息
                except Exception:
                    break  # 如果发送失败，退出循环

        # 创建保活任务
        keepalive_task = asyncio.create_task(send_keepalive())
        
        try:
            # 使用GSVI TTS服务
            audio_path = await process_gsvi_tts(text, session)
            
            # 记录TTS处理完成时间
            tts_duration = time.time() - tts_start_time
            logger.debug(f"[DIAG] TTS处理完成，耗时: {tts_duration:.2f}秒, 客户端ID: {client_id}")
            
            if audio_path:
                # 发送音频URL给客户端
                audio_url = f"/static/audio/{os.path.basename(audio_path)}"
                
                # 设置客户端状态为播放中，服务器应该等待播放完成而不是立即关闭
                session["playback_expected"] = True
                session["permanent_connection"] = True  # 添加这个标记表示永久连接
                
                # 发送完整响应
                success = await send_message(websocket, {
                    "type": "tts_complete",
                    "audio_url": audio_url,
                    "message": "语音生成完成",
                    "keep_connection": True,
                    "never_close": True,  # 添加标记告诉客户端永不关闭连接
                    "status": "success",
                    "text_processed": text[:30] + ("..." if len(text) > 30 else ""),
                    "timestamp": time.time()
                })
                
                if success:
                    logger.info(f"成功发送TTS完成消息：{audio_url}")
                else:
                    logger.warning("发送TTS完成消息失败，可能客户端已断开连接")
                    
                # 为客户端播放提供足够时间，添加会话标记
                session["audio_sent_time"] = time.time()
            else:
                # 尝试发送错误消息
                await send_message(websocket, {
                    "type": "error",
                    "message": "TTS处理失败，未返回音频路径"
                })
                
        finally:
            # 停止保活任务
            session.pop("tts_processing", None)
            keepalive_task.cancel()
            
    except WebSocketDisconnect as e:
        # 捕获但不重新抛出异常，允许处理完成
        logger.warning(f"发送TTS结果时客户端 {client_id} 已断开连接，代码: {e.code}")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"TTS处理异常: {e}\n{error_trace}")
        # 尝试发送错误消息
        await send_message(websocket, {
            "type": "error",
            "message": f"TTS处理失败: {str(e)}"
        })

def remove_duplicate_text(text: str) -> str:
    """去除重复的文本片段"""
    # 分句
    sentences = text.split('。')
    # 过滤空句
    sentences = [s.strip() + '。' for s in sentences if s.strip()]
    # 去重
    unique_sentences = []
    for sentence in sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    # 合并
    return ''.join(unique_sentences)

async def process_tts_queue(client_id: str, websocket: WebSocket, session: Dict[str, Any]):
    """处理TTS队列中的句子"""
    queue = session["tts_queue"]
    processed_texts = set()  # 用于跟踪已处理的文本
    
    try:
        while queue:
            # 获取下一个要处理的句子
            text = queue.popleft()
            
            # 检查是否已处理过相同的文本
            if text in processed_texts:
                logger.debug(f"跳过重复文本: {text[:30]}...")
                continue
                
            # 添加到已处理集合
            processed_texts.add(text)
            
            # 处理TTS
            audio_path = await process_gsvi_tts(text, session)
            
            if audio_path:
                # 发送音频URL给客户端
                audio_url = f"/static/audio/{os.path.basename(audio_path)}"
                
                await send_message(websocket, {
                    "type": "tts_sentence_complete",
                    "audio_url": audio_url,
                    "text": text,
                    "timestamp": time.time()
                })
            
            # 短暂等待，避免TTS请求过于频繁
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"处理TTS队列异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"TTS处理异常: {str(e)}"
        })

async def process_gsvi_tts(text: str, session: Dict[str, Any]):
    """使用GSVI TTS服务生成语音"""
    from app.core.tts.gsvi_tts_service import GSVITTSService
    
    try:
        # 创建TTS服务实例
        api_base = session.get("api_base", "http://127.0.0.1:5000")
        tts_service = GSVITTSService(api_base=api_base)
        
        # 生成唯一的输出文件名
        output_filename = f"gsvi_tts_assistant_{uuid.uuid4()}.mp3"
        output_path = os.path.join("static/audio", output_filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 提取可选参数
        character = session.get("character", None)
        emotion = session.get("emotion", None)
        
        # 进行TTS转换
        await tts_service.synthesize(
            text=text,
            output_file=output_path,
            character=character,
            emotion=emotion
        )
        
        logger.info(f"GSVI TTS生成完成: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"GSVI TTS处理异常: {str(e)}", exc_info=True)
        raise

async def cleanup_voice_assistant_client(client_id: str):
    """清理语音助手客户端资源"""
    if client_id in voice_assistant_connections:
        del voice_assistant_connections[client_id]
        
    if client_id in voice_assistant_sessions:
        session = voice_assistant_sessions[client_id]
        if session.get("asr_client"):
            await session["asr_client"].close()
        # 清理临时文件
        try:
            temp_dir = "static/audio"
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if f"gsvi_tts_assistant_{client_id}" in file:
                        try:
                            os.remove(os.path.join(temp_dir, file))
                            logger.debug(f"已删除临时文件: {file}")
                        except Exception as e:
                            logger.error(f"删除临时文件失败 {file}: {str(e)}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
            
        del voice_assistant_sessions[client_id]
        
    logger.info(f"语音助手客户端资源已清理: {client_id}")

def create_voice_assistant_session(client_id: str) -> Dict[str, Any]:
    """创建语音助手会话数据"""
    return {
        "client_id": client_id,
        "asr_client": None,
        "recording_active": False,
        "recognition_results": [],
        "final_result": None,
        "history_id": None,  # 用于跟踪对话历史
        "model": None,       # LLM模型
        "user_id": None,     # 用户ID
        "llm_processing_started": False,  # 避免重复处理
        "llm_response_buffer": [],  # 用于收集LLM响应文本
        "llm_response_complete": False,  # 标记LLM响应是否完成
        "provider": "gsvi",  # 默认TTS提供者
        "api_base": "http://127.0.0.1:5000",  # 默认API基础URL
        "last_activity": time.time()  # 添加最后活动时间
    }

# 添加调试端点，帮助排查连接问题
@api_ws.get("/status")
async def api_status():
    """API状态检查端点"""
    return {
        "status": "online",
        "voice_assistant_connections": len(voice_assistant_connections),
        "voice_assistant_sessions": len(voice_assistant_sessions)
    }

# 启动和关闭事件
@api_ws.on_event("startup")
async def startup_event():
    logger.info("WebSocket服务启动")
    logger.info("语音助手WebSocket服务可通过 ws://[host]/ws/voice-assistant 接口访问")

@api_ws.on_event("shutdown")
async def shutdown_event():
    # 关闭所有会话
    for client_id in list(voice_assistant_sessions.keys()):
        await cleanup_voice_assistant_client(client_id)
    logger.info("WebSocket服务已关闭")