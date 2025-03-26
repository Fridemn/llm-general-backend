"""
ws_tts.py
处理WebSocket TTS相关的核心功能
"""

import os
import uuid
import urllib.parse
from typing import Dict, Any, Optional, List
import aiohttp
import base64
from fastapi import WebSocket
from app import logger
from starlette.websockets import WebSocketDisconnect
import json
import time
import asyncio

class ConnectionManager:
    """WebSocket 连接管理器"""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_sessions: Dict[str, Dict[str, Any]] = {}
        self.last_activity: Dict[str, float] = {}
        self.socket_to_client: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """建立新的 WebSocket 连接"""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            self.last_activity[client_id] = time.time()
            self.client_sessions[client_id] = self.create_session(client_id)
            self.socket_to_client[websocket] = client_id
            client = f"{websocket.client.host}:{websocket.client.port}"
            logger.info(f"TTS接受连接: {client}, 客户端ID: {client_id}, 当前活跃连接数: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"接受TTS连接时出错: {str(e)}")
            raise

    def disconnect(self, websocket: WebSocket, client_id: str):
        """关闭 WebSocket 连接"""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                
                connection_duration = None
                if client_id in self.last_activity:
                    connection_duration = time.time() - self.last_activity[client_id]
                    del self.last_activity[client_id]
                
                if client_id in self.client_sessions:
                    del self.client_sessions[client_id]
                
                if websocket in self.socket_to_client:
                    del self.socket_to_client[websocket]
                
                client = getattr(websocket, 'client', None)
                if client:
                    duration_str = f"连接时长: {connection_duration:.2f}秒" if connection_duration else "连接时长未知"
                    logger.info(f"断开TTS连接: {client.host}:{client.port}, 客户端ID: {client_id}, {duration_str}, 当前活跃连接数: {len(self.active_connections)}")
                else:
                    logger.info(f"断开未知TTS连接, 客户端ID: {client_id}, 当前活跃连接数: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"断开TTS连接时出错: {str(e)}")

    def create_session(self, client_id: str) -> Dict[str, Any]:
        """创建TTS会话数据"""
        return {
            "client_id": client_id,
            "processing": False,
            "last_activity": time.time(),
            "audio_playing": False,
            "last_audio_sent": 0,
            "last_audio_played": 0
        }

    async def send_message(self, websocket: WebSocket, message: Dict):
        """发送任意消息给客户端，不会关闭连接"""
        try:
            await websocket.send_json(message)
            
            client_id = self.socket_to_client.get(websocket)
            if client_id and client_id in self.last_activity:
                self.last_activity[client_id] = time.time()
                
        except WebSocketDisconnect as e:
            logger.warning(f"发送消息时客户端已断开: {e.code}")
            raise
        except RuntimeError as e:
            if "websocket connection is closed" in str(e).lower():
                logger.warning("尝试发送消息时发现连接已关闭")
            else:
                logger.error(f"发送消息时出现RuntimeError: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"发送消息时出错: {str(e)}")
            raise


class TTSService:
    """TTS 服务类"""
    def __init__(self):
        os.makedirs("static/audio", exist_ok=True)
    
    async def generate_audio(self, text: str, api_base: str, character: str = None, emotion: str = None) -> str:
        """生成语音文件并返回路径"""
        path = f"static/audio/gsvi_tts_{uuid.uuid4()}.wav"
        params = {"text": text}

        if character:
            params["character"] = character
        if emotion:
            params["emotion"] = emotion

        query_parts = []
        for key, value in params.items():
            encoded_value = urllib.parse.quote(str(value))
            query_parts.append(f"{key}={encoded_value}")

        api_base = api_base.rstrip("/")
        url = f"{api_base}/tts?{'&'.join(query_parts)}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(path, "wb") as f:
                            f.write(await response.read())
                        return path
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"TTS API 请求失败，状态码: {response.status}，错误: {error_text}"
                        )
        except Exception as e:
            raise Exception(f"TTS 服务调用失败: {str(e)}")


manager = ConnectionManager()
tts_service = TTSService()


async def process_tts_message(client_id: str, websocket: WebSocket, data: str):
    """处理TTS WebSocket消息"""
    try:
        if len(data) < 50:
            try:
                message = json.loads(data)
                
                if "ping" in message and message["ping"] is True:
                    if client_id in manager.last_activity:
                        manager.last_activity[client_id] = time.time()
                    await manager.send_message(websocket, {
                        "pong": True,
                        "timestamp": time.time()
                    })
                    return
                
                if "keep_alive" in message and message["keep_alive"] is True:
                    if client_id in manager.last_activity:
                        manager.last_activity[client_id] = time.time()
                    await manager.send_message(websocket, {
                        "pong": True,
                        "keep_alive": True
                    })
                    return
                    
                if "init" in message and message["init"] is True:
                    client_agent = message.get("client", message.get("userAgent", "未知"))
                    logger.info(f"客户端初始化: {client_id} - {client_agent}")
                    await manager.send_message(websocket, {
                        "status": "initialized", 
                        "message": "客户端初始化成功",
                        "client_id": client_id
                    })
                    return
                    
                if "disconnect" in message and message["disconnect"] is True:
                    logger.info(f"客户端 {client_id} 请求断开连接")
                    await manager.send_message(websocket, {"message": "连接将关闭", "status": "closing"})
                    return
            except json.JSONDecodeError:
                logger.warning(f"接收到无效的短消息: {data}")
                return
        
        message = json.loads(data)
        
        if "init" in message or "ping" in message or "keep_alive" in message:
            return
        
        text = message.get("text", "")
        character = message.get("character")
        emotion = message.get("emotion")
        
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            logger.warning(f"来自 {client_id} 的请求缺少文本内容")
            await manager.send_message(websocket, {
                "error": "缺少文本内容或文本为空",
                "status": "error",
                "details": "请提供要转换为语音的文本内容"
            })
            return
            
        logger.info(f"处理TTS请求: client_id={client_id}, 文本内容='{text[:30]}{'...' if len(text) > 30 else ''}'")
        
        api_base = message.get("config", {}).get("api_base", "http://127.0.0.1:5000")
        if not api_base or not isinstance(api_base, str):
            api_base = "http://127.0.0.1:5000"
            
        logger.info(f"处理TTS请求: client_id={client_id}, 文本长度={len(text)}, character={character}, emotion={emotion}")
        
        if client_id in manager.client_sessions:
            manager.client_sessions[client_id]["processing"] = True
        
        try:
            await manager.send_message(websocket, {
                "status": "processing",
                "message": "正在处理TTS请求"
            })
        except Exception as e:
            logger.error(f"发送处理状态消息失败: {str(e)}")
            if client_id in manager.client_sessions:
                manager.client_sessions[client_id]["processing"] = False
            return
            
        try:
            audio_path = await tts_service.generate_audio(text, api_base, character, emotion)
            
            if client_id not in manager.client_sessions:
                logger.warning(f"客户端已断开连接，放弃发送TTS响应: {client_id}")
                return
            
            audio_data = ""
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                    file_size = len(audio_bytes)
                    logger.info(f"音频文件大小: {file_size} 字节")
                    audio_data = base64.b64encode(audio_bytes).decode("utf-8")
                    logger.info(f"Base64编码后大小: {len(audio_data)} 字符")
            except Exception as e:
                logger.error(f"读取音频文件失败: {str(e)}")
                await manager.send_message(websocket, {
                    "error": f"读取音频文件失败: {str(e)}",
                    "status": "error"
                })
                return
            
            if client_id not in manager.client_sessions:
                logger.warning(f"发送音频前客户端已断开连接: {client_id}")
                return
                
            try:
                await manager.send_message(websocket, {
                    "status": "ready",
                    "message": "音频数据已准备好",
                    "audio_size": len(audio_data),
                    "text": text[:30] + ("..." if len(text) > 30 else "")
                })
                
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"发送预备消息失败: {str(e)}")
                return
                
            try:
                audio_filename = os.path.basename(audio_path)
                await manager.send_message(websocket, {
                    "audio_path": f"/static/audio/{audio_filename}",
                    "status": "success",
                    "keep_connection": True,
                    "text_processed": text[:20] + "..." if len(text) > 20 else text,
                    "audio": audio_data,
                    "format": "wav" 
                })
            except Exception as e:
                logger.error(f"发送音频数据失败: {str(e)}")
                return
            
            if client_id in manager.client_sessions:
                manager.client_sessions[client_id]["last_audio_sent"] = time.time()
                manager.client_sessions[client_id]["audio_playing"] = True
            
            try:
                await manager.send_message(websocket, {
                    "message": "处理完成，连接保持中",
                    "status": "ready",
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.error(f"发送完成消息失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"TTS处理错误: {str(e)}")
            try:
                await manager.send_message(websocket, {
                    "error": f"TTS处理错误: {str(e)}",
                    "status": "error",
                    "keep_connection": True
                })
            except Exception as send_err:
                logger.error(f"发送错误消息失败: {str(send_err)}")
    except json.JSONDecodeError:
        logger.warning(f"来自 {client_id} 的无效JSON数据: {data[:100]}...")
        try:
            await manager.send_message(websocket, {"error": "无效的请求格式", "status": "error"})
        except Exception as e:
            logger.error(f"发送错误消息失败: {str(e)}")
    except Exception as e:
        logger.error(f"处理TTS消息异常: {str(e)}", exc_info=True)
        try:
            await manager.send_message(websocket, {
                "error": f"处理异常: {str(e)}",
                "status": "error"
            })
        except:
            pass
    finally:
        # 重置处理状态
        if client_id in manager.client_sessions:
            manager.client_sessions[client_id]["processing"] = False


def cleanup_temp_file(file_path: str) -> None:
    """安全地清理临时文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"已删除临时文件: {file_path}")
    except Exception as e:
        logger.error(f"删除临时文件失败 {file_path}: {str(e)}")
