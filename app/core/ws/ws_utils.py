"""
ws_utils.py
存放WebSocket相关的工具函数，专注于支持语音助手功能
"""

import json
import traceback
from typing import Dict, Any, Callable, Awaitable, Optional
import time
import asyncio
import os
import uuid

from app import logger
from app.core.ws.audio_service import AudioService
from app.core.ws.audio_websocket_client import AudioWebSocketClient
from app.core.pipeline.chat_process import chat_process
from app.core.llm.message import LLMMessage, MessageRole
from app.core.pipeline.text_process import text_process
from app.core.tts.gsvi_tts_service import GSVITTSService  # 添加这行导入
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
        logger.debug(f"发送消息: {text[:100]}...")
        
        # 检查并记录WebSocket状态
        if hasattr(websocket, "client_state"):
            logger.debug(f"[DIAG] 发送前WebSocket状态: {websocket.client_state}")
        
        # 检查连接是否打开
        if hasattr(websocket, "client_state") and websocket.client_state != WebSocketState.CONNECTED:
            logger.warning(f"WebSocket未连接，状态为: {websocket.client_state}，尝试发送的消息类型: {message.get('type', 'unknown')}")
            return False
            
        # 添加消息队列缓存逻辑，保证一定间隔
        if not hasattr(websocket, "_last_send_time"):
            setattr(websocket, "_last_send_time", 0)
            
        # 如果消息过于密集，添加小延迟防止消息阻塞
        last_send = getattr(websocket, "_last_send_time", 0)
        if time.time() - last_send < 0.05:  # 50ms的最小间隔
            await asyncio.sleep(0.01)  # 10ms延迟
        
        # 在发送任何消息前，禁用可能导致连接关闭的功能
        if hasattr(websocket, "_close_called"):
            setattr(websocket, "_close_called", False)  # 重置关闭标志
        
        # 发送消息
        await websocket.send_text(text)
        setattr(websocket, "_last_send_time", time.time())
        
        # 检查并记录发送后WebSocket状态
        if hasattr(websocket, "client_state"):
            logger.debug(f"[DIAG] 发送后WebSocket状态: {websocket.client_state}")
        
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

# 语音聊天处理函数 - 这是/voice-assistant接口需要的核心功能
async def process_voice_chat_message(client_id: str, websocket, data_text: str, 
                                    client_sessions: Dict[str, SessionData]):
    """处理语音聊天客户端消息"""
    try:
        data = json.loads(data_text)
        command = data.get("command")
        
        if not command:
            await send_message(websocket, {
                "type": "error",
                "message": "缺少command字段"
            })
            return
        
        # 命令处理映射
        voice_command_handlers = {
            "start_voice_chat": handle_start_voice_chat,
            "stop_recording": handle_stop_recording,
            "set_params": handle_set_params,
            "send_audio": handle_send_audio
        }
        
        # 执行对应的处理函数
        if command in voice_command_handlers:
            await voice_command_handlers[command](client_id, websocket, data, client_sessions)
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
        error_trace = traceback.format_exc()
        logger.error(f"处理语音聊天消息异常: {e}\n{error_trace}")
        await send_message(websocket, {
            "type": "error",
            "message": f"处理消息异常: {str(e)}"
        })

async def handle_start_voice_chat(client_id: str, websocket, data: Dict[str, Any], 
                                 client_sessions: Dict[str, SessionData]):
    """处理开始语音聊天命令"""
    session = client_sessions[client_id]
    
    if session.get("recording_active"):
        await send_message(websocket, {
            "type": "error",
            "message": "录音已在进行中"
        })
        return
    
    # 获取参数
    params = extract_voice_chat_parameters(data)
    
    # 验证必要参数
    if not params["model"] or not params["history_id"]:
        missing = []
        if not params["model"]: missing.append("模型参数")
        if not params["history_id"]: missing.append("对话历史ID")
        
        await send_message(websocket, {
            "type": "error",
            "message": f"缺少必要参数: {', '.join(missing)}"
        })
        return
    
    # 更新会话信息
    update_session_with_params(session, params)
    
    logger.info(f"语音聊天客户端 {client_id} 开始录音: 时长={params['duration']}秒, 模型={params['model']}, 历史ID={params['history_id']}, 用户ID={params['user_id']}")
    
    try:
        # 创建ASR客户端并连接
        asr_client = await create_and_connect_asr_client(
            client_id, 
            params["server_url"], 
            session
        )
        if not asr_client:
            await send_message(websocket, {
                "type": "error",
                "message": "无法连接到语音服务器"
            })
            return
            
        # 发送录音开始消息
        await send_message(websocket, {
            "type": "recording",
            "status": "started",
            "duration": params["duration"]
        })
        
        # 创建消息发送器函数及录音完成回调
        message_sender, recording_task = await create_voice_chat_tasks(
            client_id, 
            websocket, 
            session, 
            params["duration"]
        )
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"语音聊天客户端 {client_id} 启动录音异常: {e}\n{error_trace}")
        await cleanup_asr_client(session)
        await send_message(websocket, {
            "type": "error",
            "message": f"启动录音异常: {str(e)}"
        })

async def create_and_connect_asr_client(client_id: str, server_url: str, session: SessionData) -> Optional[AudioWebSocketClient]:
    """创建并连接ASR客户端"""
    logger.debug(f"为客户端 {client_id} 创建ASR客户端, 连接到 {server_url}")
    asr_client = AudioWebSocketClient(server_url)
    session["asr_client"] = asr_client
    
    # 连接到语音服务器
    logger.debug(f"客户端 {client_id} 正在尝试连接到语音服务器")
    if await asr_client.connect():
        logger.info(f"客户端 {client_id} 成功连接到语音服务器")
        session["recording_active"] = True
        session["recognition_results"] = []
        session["final_result"] = None
        return asr_client
    else:
        logger.error(f"客户端 {client_id} 无法连接到语音服务器")
        await cleanup_asr_client(session)
        return None

async def cleanup_asr_client(session: SessionData):
    """清理ASR客户端资源"""
    if session.get("asr_client"):
        await session["asr_client"].close()
        session["asr_client"] = None

def extract_voice_chat_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """提取语音聊天参数"""
    return {
        "duration": int(data.get("duration", 10)),
        "model": data.get("model"),
        "history_id": data.get("history_id"),
        "user_id": data.get("user_id"),
        "server_url": data.get("server_url", "ws://127.0.0.1:8765")
    }

def update_session_with_params(session: SessionData, params: Dict[str, Any]):
    """更新会话信息"""
    session["model"] = params["model"]
    session["history_id"] = params["history_id"]
    session["user_id"] = params["user_id"]
    session["final_text"] = ""  # 用于收集流式识别的文本
    session["llm_processing_started"] = False

async def create_voice_chat_tasks(client_id: str, websocket, session: SessionData, duration: int):
    """创建语音聊天任务"""
    import asyncio
    
    # 创建消息发送器函数
    async def message_sender(message):
        await send_message(websocket, message)
        
        # 检测到录音完成时，手动调用处理函数
        if message.get("status") == "completed" and message.get("result"):
            # 创建一个任务来处理ASR结果，避免阻塞当前函数
            asyncio.create_task(
                handle_voice_chat_asr_complete(
                    client_id, 
                    websocket, 
                    message.get("result"), 
                    {client_id: session}
                )
            )
    
    # 定义回调函数，在录音任务完成时检查是否有有效文本
    async def recording_completed(task):
        try:
            task.result()  # 获取结果，如有异常会抛出
            logger.debug(f"语音聊天客户端 {client_id} 录音任务正常完成")
            
            # 如果没有收到completed消息但收集了final_text，手动处理
            if not session.get("llm_processing_started", False) and session.get("final_text"):
                logger.info(f"录音任务结束，手动触发文本处理: {session['final_text']}")
                # 创建最终结果
                final_result = {
                    "text": session.get("final_text", ""),
                    "user": "未知用户"
                }
                # 处理识别结果
                await handle_voice_chat_asr_complete(
                    client_id, 
                    websocket, 
                    final_result, 
                    {client_id: session}
                )
        except asyncio.CancelledError:
            logger.warning(f"语音聊天 {client_id} 录音任务被取消")
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"语音聊天 {client_id} 录音任务异常: {e}\n{error_trace}")
            await send_message(websocket, {
                "type": "error",
                "message": f"录音处理异常: {str(e)}"
            })
        finally:
            # 确保会话被重置
            session["recording_active"] = False
    
    # 启动录音任务
    recording_task = asyncio.create_task(AudioService.perform_recording(
        session, message_sender, duration, False
    ))
    
    # 添加完成回调
    recording_task.add_done_callback(
        lambda t: asyncio.create_task(recording_completed(t))
    )
    
    return message_sender, recording_task

async def handle_stop_recording(client_id: str, websocket, data: Dict[str, Any], 
                               client_sessions: Dict[str, SessionData]):
    """处理停止录音命令"""
    session = client_sessions[client_id]
    
    if not session.get("recording_active"):
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

async def handle_set_params(client_id: str, websocket, data: Dict[str, Any], 
                           client_sessions: Dict[str, SessionData]):
    """设置语音聊天参数"""
    session = client_sessions[client_id]
    
    # 更新会话参数
    if data.get("model"):
        session["model"] = data["model"]
    if data.get("history_id"):
        session["history_id"] = data["history_id"]
    if data.get("user_id"):
        session["user_id"] = data["user_id"]
    if "stream_mode" in data:
        session["stream_mode"] = data["stream_mode"]
    
    # 回复参数更新状态
    await send_message(websocket, {
        "type": "params",
        "status": "updated",
        "model": session.get("model"),
        "history_id": session.get("history_id"),
        "user_id": session.get("user_id"),
        "stream_mode": session.get("stream_mode", True)
    })

async def handle_send_audio(client_id: str, websocket, data: Dict[str, Any], 
                           client_sessions: Dict[str, SessionData]):
    """处理客户端发送的音频数据"""
    await send_message(websocket, {
        "type": "error",
        "message": "直接发送音频数据功能尚未实现"
    })

async def handle_voice_chat_asr_complete(client_id: str, websocket, result, client_sessions):
    """语音识别完成后的回调，将文本发送给LLM处理"""
    session = client_sessions[client_id]
    
    # 更新会话状态
    session["recording_active"] = False
    session["final_result"] = result
    
    # 提取文本内容
    text = result.get("text", "")
    
    # 记录收到的识别结果
    logger.info(f"语音聊天收到ASR结果: {text}")
    
    # 检查文本是否有效
    if not is_valid_asr_text(text):
        logger.warning(f"识别结果为空或是等待消息: '{text}'")
        await send_message(websocket, {
            "type": "error",
            "message": "语音识别失败或没有识别到内容"
        })
        return
    
    # 避免重复处理
    if session.get("llm_processing_started", False):
        logger.warning(f"已经在处理此识别结果，避免重复: {text[:30]}...")
        return
    
    # 设置处理标志
    session["llm_processing_started"] = True
    
    try:
        # 发送识别结果给客户端
        logger.info(f"语音聊天客户端 {client_id} 识别到文本: {text}")
        await send_message(websocket, {
            "type": "asr_result",
            "text": text
        })
        
        # 获取LLM处理参数
        llm_params = get_llm_processing_params(session)
        
        # 验证必要参数
        validation_result = validate_llm_params(llm_params)
        if validation_result:
            await send_message(websocket, {
                "type": "error",
                "message": validation_result
            })
            session["llm_processing_started"] = False
            return
        
        # 处理LLM请求
        logger.info(f"语音聊天客户端 {client_id} 准备发送文本到LLM: '{text}', 历史ID: {llm_params['history_id']}, 流式模式: {llm_params['stream_mode']}")
        
        if llm_params["stream_mode"]:
            await handle_stream_response(client_id, websocket, llm_params["model"], text, llm_params["history_id"], llm_params["user_id"], session)
        else:
            await handle_normal_response(client_id, websocket, llm_params["model"], text, llm_params["history_id"], llm_params["user_id"], session)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"处理LLM请求异常: {e}\n{error_trace}")
        await send_message(websocket, {
            "type": "error",
            "message": f"LLM处理异常: {str(e)}"
        })
    finally:
        # 重置处理标志
        session["llm_processing_started"] = False

def is_valid_asr_text(text: str) -> bool:
    """检查ASR文本是否有效"""
    wait_messages = ["等待识别结果...", "识别结果将在服务器处理完成后显示"]
    return bool(text and text.strip() and text not in wait_messages)

def get_llm_processing_params(session: SessionData) -> Dict[str, Any]:
    """获取LLM处理参数"""
    return {
        "model": session.get("model"),
        "history_id": session.get("history_id"),
        "user_id": session.get("user_id"),
        "stream_mode": session.get("stream_mode", True)
    }

def validate_llm_params(params: Dict[str, Any]) -> Optional[str]:
    """验证LLM处理参数"""
    if not params["model"]:
        return "缺少模型参数，无法处理LLM请求"
    if not params["history_id"]:
        return "缺少对话历史ID，无法处理LLM请求"
    return None

async def handle_stream_response(client_id: str, websocket, model, text, history_id, user_id, session):
    """处理流式LLM响应"""
    logger.info(f"开始流式LLM请求: model={model}, history_id={history_id}, message='{text}'")
    try:
        # 重置状态
        setattr(websocket, "_text_buffer", "")
        setattr(websocket, "_processed_sentences", set())
        
        await send_message(websocket, {
            "type": "llm_stream_start",
            "history_id": history_id
        })
        
        full_text = ""
        message_id = None
        
        # 处理LLM流式响应
        input_message = LLMMessage.from_text(
            text=text,
            history_id=history_id or "",
            role=MessageRole.USER
        )
        
        generator = text_process.process_message_stream(
            model=model,
            message=input_message,
            history_id=history_id,
            user_id=user_id
        )
        
        # 传入session作为参数
        async for chunk in generator:
            await process_stream_chunk(
                chunk, websocket, history_id, full_text, message_id, session
            )
            if isinstance(chunk, str) and not chunk.startswith("__TOKEN_INFO__"):
                full_text += chunk
        
        # 处理缓冲区中可能剩余的文本
        final_buffer = getattr(websocket, "_text_buffer", "").strip()
        if final_buffer:
            audio_path = await process_gsvi_tts(final_buffer, session)
            if audio_path:
                audio_url = f"/static/audio/{os.path.basename(audio_path)}"
                await send_message(websocket, {
                    "type": "tts_sentence_complete",
                    "text": final_buffer,
                    "audio_url": audio_url,
                    "history_id": history_id,
                    "final": True
                })
        
        # 清理状态
        delattr(websocket, "_text_buffer")
        delattr(websocket, "_processed_sentences")
        
        # 发送流式响应结束消息
        await send_message(websocket, {
            "type": "llm_stream_end",
            "text": full_text,
            "message_id": message_id,
            "history_id": history_id
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"流式处理异常: {e}\n{error_trace}")
        await send_message(websocket, {
            "type": "error",
            "message": f"流式处理异常: {str(e)}"
        })

def split_text_to_sentences(text: str) -> list[str]:
    """将文本切分成句子，保留标点符号"""
    # 定义分句标点符号
    punctuations = ['。', '，', '！', '？', '，', ',', '!', '?', '.']
    
    sentences = []
    current = ""
    
    for char in text:
        current += char
        if char in punctuations:
            if current.strip():
                sentences.append(current.strip())
            current = ""
            
    # 处理最后可能没有标点的部分
    if current.strip():
        sentences.append(current.strip())
        
    return sentences

async def process_stream_chunk(chunk, websocket, history_id, full_text, message_id, session):
    """处理流式响应的单个数据块，支持实时句子检测和TTS处理"""
    if isinstance(chunk, str):
        if chunk.startswith("__TOKEN_INFO__"):
            try:
                token_data = json.loads(chunk[14:])
                message_id = token_data.get("message_id")
            except Exception as e:
                logger.error(f"解析token信息失败: {e}")
            return

        # 初始化或获取缓冲区
        if not hasattr(websocket, "_text_buffer"):
            setattr(websocket, "_text_buffer", "")
            setattr(websocket, "_processed_sentences", set())  # 记录已处理的句子
        
        buffer = getattr(websocket, "_text_buffer") + chunk
        processed_sentences = getattr(websocket, "_processed_sentences")
        
        # 检查是否可以切分句子
        while True:
            split_result = check_complete_sentence(buffer)
            if not split_result:
                break
            
            sentence, remaining = split_result
            sentence = sentence.strip()
            
            # 检查句子是否已处理过
            if sentence and sentence not in processed_sentences:
                logger.debug(f"检测到完整句子: {sentence}")
                processed_sentences.add(sentence)
                
                # 发送句子进行TTS处理
                audio_path = await process_gsvi_tts(sentence, session)
                if audio_path:
                    audio_url = f"/static/audio/{os.path.basename(audio_path)}"
                    await send_message(websocket, {
                        "type": "tts_sentence_complete",
                        "text": sentence,
                        "audio_url": audio_url,
                        "history_id": history_id
                    })
            
            buffer = remaining
        
        # 更新缓冲区
        setattr(websocket, "_text_buffer", buffer)
        
        # 发送内容到前端显示
        await send_message(websocket, {
            "type": "llm_stream_chunk",
            "content": chunk,
            "history_id": history_id
        })

    elif isinstance(chunk, dict):
        if "text" in chunk:
            content = chunk["text"]
            await send_message(websocket, {
                "type": "llm_stream_chunk",
                "content": content,
                "history_id": history_id
            })
        elif "message_id" in chunk:
            message_id = chunk["message_id"]
        elif "token_info" in chunk and "message_id" in chunk["token_info"]:
            message_id = chunk["token_info"]["message_id"]

def check_complete_sentence(text: str) -> Optional[tuple[str, str]]:
    """检查并返回完整的句子
    
    Args:
        text: 要检查的文本
        
    Returns:
        Optional[tuple[str, str]]: (完整句子, 剩余文本) 或 None
    """
    # 定义句子结束标记
    end_marks = ['。', '！', '？', '，', ',', '!', '?', '.']
    
    for i, char in enumerate(text):
        if char in end_marks:
            sentence = text[:i + 1]
            remaining = text[i + 1:]
            return sentence, remaining
            
    return None

async def handle_normal_response(client_id: str, websocket, model, text, history_id, user_id, session):
    """处理普通LLM响应"""
    # 使用chat_process处理LLM请求
    response = await chat_process.handle_request(
        model=model,
        message=text,
        history_id=history_id,
        role=MessageRole.USER,
        stream=False,
        stt=False,
        tts=True,  # 开启TTS
        audio_file=None,
        user_id=user_id
    )
    
    # 检查LLM响应
    if response.get("success"):
        # 从响应中提取文本和音频URL
        llm_text = response.get("message", "")
        audio_url = response.get("audio_url")
        
        # 发送LLM响应给客户端
        await send_message(websocket, {
            "type": "llm_response",
            "text": llm_text,
            "audio_url": audio_url,
            "message_id": response.get("message_id"),
            "history_id": history_id
        })
    else:
        # 处理错误情况
        await send_message(websocket, {
            "type": "error",
            "message": f"LLM处理失败: {response.get('error', '未知错误')}"
        })

async def process_gsvi_tts(text: str, session: Dict[str, Any]) -> Optional[str]:
    """使用GSVI TTS服务生成语音

    Args:
        text: 要转换的文本
        session: 包含TTS配置的会话数据

    Returns:
        str: 生成的音频文件路径，失败则返回None
    """
    try:
        # 创建TTS服务实例
        api_base = session.get("api_base", "http://127.0.0.1:5000")
        tts_service = GSVITTSService(api_base=api_base)
        
        # 生成唯一的输出文件名
        output_filename = f"gsvi_tts_assistant_{uuid.uuid4()}.mp3"
        output_path = os.path.join("static/audio", output_filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 提取TTS参数
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
        return None
