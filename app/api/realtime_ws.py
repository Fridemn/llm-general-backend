from fastapi import APIRouter, WebSocket, WebSocketDisconnect, FastAPI
import asyncio
import json
import os
import uuid
import time  # 添加time模块导入
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from app import logger
from app.core.ws.ws_utils import send_message
from app.core.ws.realtime_voice_client import RealtimeVoiceClient
from app.core.pipeline.chat_process import chat_process
from app.core.llm.message import LLMMessage, MessageRole

from app.core.pipeline.text_process import text_process
from app.core.tts.gsvi_tts_service import GSVITTSService  # 添加这行导入
from starlette.websockets import WebSocketDisconnect

api_realtime = APIRouter()

# 存储连接和会话
realtime_connections: Dict[str, WebSocket] = {}
realtime_sessions: Dict[str, Dict[str, Any]] = {}

# 创建一个新的FastAPI应用实例
app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载WebSocket路由
app.include_router(api_realtime, prefix="/ws")

@api_realtime.websocket("/realtime-voice-chat")
async def realtime_voice_endpoint(websocket: WebSocket):
    """实时语音聊天WebSocket接口"""
    client_id = str(uuid.uuid4())
    
    try:
        # 简化连接建立过程
        await websocket.accept()
        websocket.auto_close = False
        
        # 保存连接
        realtime_connections[client_id] = websocket
        realtime_sessions[client_id] = create_realtime_session(client_id)
        
        # 发送连接成功消息
        await send_message(websocket, {
            "type": "connection",
            "client_id": client_id,
            "message": "实时语音助手连接成功"
        })
        
        # 启动心跳检测
        heartbeat_task = asyncio.create_task(heartbeat_check(client_id, websocket))
        
        try:
            while True:
                try:
                    data = await websocket.receive_text()
                    # 处理心跳消息
                    if '"type":"ping"' in data or '"keep_alive":true' in data:
                        await send_message(websocket, {"type": "pong"})
                        continue
                    await process_realtime_message(client_id, websocket, data)
                except WebSocketDisconnect:
                    break
                except asyncio.TimeoutError:
                    # 发送心跳检查
                    if not await send_message(websocket, {"type": "ping"}):
                        break
                    continue
                
        finally:
            if 'heartbeat_task' in locals():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            
    except Exception as e:
        logger.error(f"WebSocket处理异常: {str(e)}")
    finally:
        await cleanup_realtime_client(client_id)

async def heartbeat_check(client_id: str, websocket: WebSocket):
    """心跳检测"""
    while True:
        try:
            await asyncio.sleep(15)  # 15秒检查一次
            if not await send_message(websocket, {"type": "ping"}):
                break
        except Exception as e:
            logger.error(f"心跳检测异常: {str(e)}")
            break

async def process_realtime_message(client_id: str, websocket: WebSocket, data: str):
    """处理实时语音消息"""
    try:
        message = json.loads(data)
        session = realtime_sessions[client_id]
        command = message.get("command")
        
        if command == "start":
            # 启动语音客户端
            await handle_start_command(client_id, websocket, message, session)
        elif command == "stop":
            # 停止语音客户端
            await handle_stop_command(client_id, websocket, session)
        elif command == "set_params":
            # 设置参数
            await handle_set_params(client_id, websocket, message, session)
            
    except json.JSONDecodeError:
        await send_message(websocket, {
            "type": "error",
            "message": "无效的JSON数据"
        })
    except Exception as e:
        logger.error(f"处理消息异常: {e}")
        await send_message(websocket, {
            "type": "error", 
            "message": str(e)
        })
        
async def handle_start_command(client_id: str, websocket: WebSocket, 
                             data: Dict[str, Any], session: Dict[str, Any]):
    """处理开始命令"""
    # 强化参数检查
    if not session.get("model"):
        await send_message(websocket, {
            "type": "error",
            "message": "请先选择模型"
        })
        return
        
    if not session.get("history_id"):
        await send_message(websocket, {
            "type": "error",
            "message": "请先创建或设置对话历史ID"
        })
        return
        
    try:
        # 创建语音客户端
        voice_client = RealtimeVoiceClient(data.get("server_url", "ws://127.0.0.1:8765"))
        session["voice_client"] = voice_client
        
        # 连接语音服务器
        if await voice_client.connect():
            # 启动音频流
            await voice_client.start_stream()
            
            # 启动结果处理任务
            session["result_task"] = asyncio.create_task(
                handle_recognition_results(client_id, websocket, session)
            )
            
            await send_message(websocket, {
                "type": "start",
                "status": "success"
            })
        else:
            await send_message(websocket, {
                "type": "error",
                "message": "无法连接到语音服务器"
            })
            
    except Exception as e:
        logger.error(f"启动语音客户端异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": str(e)
        })
        
async def handle_stop_command(client_id: str, websocket: WebSocket, 
                            session: Dict[str, Any]):
    """处理停止命令"""
    voice_client = session.get("voice_client")
    if voice_client:
        await voice_client.stop_stream()
        await voice_client.close()
        session["voice_client"] = None
        
    if session.get("result_task"):
        session["result_task"].cancel()
        session["result_task"] = None
        
    await send_message(websocket, {
        "type": "stop",
        "status": "success"
    })
    
async def handle_set_params(client_id: str, websocket: WebSocket,
                          data: Dict[str, Any], session: Dict[str, Any]):
    """处理参数设置"""
    # 更新会话参数
    if "model" in data:
        session["model"] = data["model"]
    if "history_id" in data:
        session["history_id"] = data["history_id"]
    if "user_id" in data:
        session["user_id"] = data["user_id"]
        
    await send_message(websocket, {
        "type": "params",
        "status": "updated"
    })
    
async def handle_recognition_results(client_id: str, websocket: WebSocket, 
                                  session: Dict[str, Any]):
    """处理识别结果"""
    voice_client = session["voice_client"]
    
    while True:
        try:
            # 等待识别结果
            result = await voice_client.wait_for_result()
            
            if not result:
                logger.warning("未收到识别结果")
                continue
                
            # 处理错误情况
            if "error" in result:
                logger.error(f"识别错误: {result['error']}")
                await send_message(websocket, {
                    "type": "error",
                    "message": f"识别失败: {result['error']}"
                })
                continue
                
            # 如果有有效的文本内容
            if result.get("text"):
                # 发送识别结果
                await send_message(websocket, {
                    "type": "recognition_result",
                    "text": result["text"],
                    "user": result.get("user", "Unknown")
                })
                
                # 调用LLM处理
                try:
                    # 判断是否使用流式响应模式
                    use_stream = True  # 默认使用流式响应
                    
                    if use_stream:
                        # 启动流式响应处理
                        await handle_stream_llm_response(
                            client_id,
                            websocket,
                            session["model"],
                            result["text"],
                            session["history_id"],
                            session.get("user_id"),
                            session
                        )
                    else:
                        # 普通响应处理
                        response = await chat_process.handle_request(
                            model=session["model"],
                            message=result["text"],
                            history_id=session["history_id"],
                            role=MessageRole.USER,
                            stream=False,
                            stt=False,  # 不需要语音识别
                            tts=True,   # 需要文本转语音
                            audio_file=None,
                            user_id=session.get("user_id")
                        )
                        
                        if response.get("success"):
                            # 发送完整响应
                            await send_message(websocket, {
                                "type": "llm_response",
                                "text": response["message"],
                                "message_id": response.get("message_id"),
                                "audio_url": response.get("audio_url")
                            })
                        else:
                            await send_message(websocket, {
                                "type": "error",
                                "message": f"LLM处理失败: {response.get('error', '未知错误')}"
                            })
                except Exception as e:
                    logger.error(f"LLM处理异常: {e}")
                    await send_message(websocket, {
                        "type": "error",
                        "message": f"LLM处理失败: {str(e)}"
                    })
                    
        except asyncio.CancelledError:
            logger.info("识别结果处理被取消")
            break
        except Exception as e:
            logger.error(f"处理识别结果异常: {e}")
            await send_message(websocket, {
                "type": "error",
                "message": f"处理识别结果异常: {str(e)}"
            })

async def handle_stream_llm_response(client_id: str, websocket: WebSocket, 
                                   model: str, text: str, history_id: str, 
                                   user_id: str, session: Dict[str, Any]):
    """处理LLM的流式响应并支持TTS"""
    try:
        # 通知客户端流式响应开始
        await send_message(websocket, {
            "type": "llm_stream_start",
            "history_id": history_id
        })
        
        # 准备从LLM获取流式响应
        input_message = LLMMessage.from_text(
            text=text,
            history_id=history_id,
            role=MessageRole.USER
        )
        
        # 获取流式响应生成器
        generator = text_process.process_message_stream(
            model=model,
            message=input_message,
            history_id=history_id,
            user_id=user_id
        )
        
        # 处理流式响应
        full_text = ""
        message_id = None
        current_sentence = ""
        
        # 缓存已经处理过的句子
        processed_sentences = set()
        
        # 句子结束标记
        sentence_endings = ["。", "！", "？", ".", "!", "?", ",", "，"]
        
        async for chunk in generator:
            # 处理控制消息
            if isinstance(chunk, dict):
                if "token_info" in chunk and "message_id" in chunk["token_info"]:
                    message_id = chunk["token_info"]["message_id"]
                continue
                
            # 处理文本块
            if isinstance(chunk, str) and chunk:
                # 跳过特殊令牌信息
                if chunk.startswith("__TOKEN_INFO__"):
                    try:
                        token_data = json.loads(chunk[14:])
                        if "message_id" in token_data:
                            message_id = token_data["message_id"]
                    except:
                        pass
                    continue
                    
                # 发送流式块到前端
                await send_message(websocket, {
                    "type": "llm_stream_chunk",
                    "content": chunk
                })
                
                # 累积文本
                full_text += chunk
                current_sentence += chunk
                
                # 检测完整句子
                for ending in sentence_endings:
                    if ending in current_sentence:
                        parts = current_sentence.split(ending, 1)
                        if len(parts) > 1:
                            # 构建完整句子（包含标点）
                            completed_sentence = parts[0] + ending
                            # 更新剩余部分
                            current_sentence = parts[1]
                            
                            # 处理完整句子的TTS
                            if completed_sentence.strip() and completed_sentence not in processed_sentences:
                                processed_sentences.add(completed_sentence)
                                await process_tts_for_sentence(completed_sentence, websocket, history_id)
                
        # 处理最后剩余的文本
        if current_sentence.strip() and current_sentence not in processed_sentences:
            await process_tts_for_sentence(current_sentence, websocket, history_id)
            
        # 通知客户端流式响应结束
        await send_message(websocket, {
            "type": "llm_stream_end",
            "text": full_text,
            "message_id": message_id,
            "history_id": history_id
        })
            
    except Exception as e:
        logger.error(f"流式LLM响应处理异常: {e}")
        await send_message(websocket, {
            "type": "error",
            "message": f"流式响应处理异常: {str(e)}"
        })

async def process_tts_for_sentence(sentence: str, websocket: WebSocket, history_id: str):
    """为单句话处理TTS并发送结果给客户端"""
    try:
        # 创建TTS服务客户端
        tts_service = GSVITTSService(api_base="http://127.0.0.1:5000")
        
        # 生成唯一的音频文件名
        output_filename = f"tts_response_{uuid.uuid4()}.mp3"
        output_path = os.path.join("static/audio", output_filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 生成语音
        await tts_service.synthesize(
            text=sentence,
            output_file=output_path
        )
        
        # 创建相对URL路径
        audio_url = f"/static/audio/{output_filename}"
        
        # 发送TTS结果给客户端
        await send_message(websocket, {
            "type": "tts_sentence_complete",
            "text": sentence,
            "audio_url": audio_url,
            "history_id": history_id
        })
        
    except Exception as e:
        logger.error(f"TTS处理异常: {e}")
        # 不中断流程，只记录错误

def create_realtime_session(client_id: str) -> Dict[str, Any]:
    """创建会话数据"""
    return {
        "client_id": client_id,
        "voice_client": None,
        "result_task": None,
        "model": None,
        "history_id": None,
        "user_id": None,
        "last_heartbeat": time.time(),  # 现在可以正确使用time模块
        "reconnect_attempts": 0
    }
    
async def cleanup_realtime_client(client_id: str):
    """清理客户端资源"""
    if client_id in realtime_connections:
        del realtime_connections[client_id]
        
    if client_id in realtime_sessions:
        session = realtime_sessions[client_id]
        if session.get("voice_client"):
            await session["voice_client"].close()
        if session.get("result_task"):
            session["result_task"].cancel()
        del realtime_sessions[client_id]
