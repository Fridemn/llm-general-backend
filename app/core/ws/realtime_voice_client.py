import asyncio
import websockets
import json
import base64
import pyaudio
import numpy as np
import wave
import os
import uuid
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import time
import queue

from app.core.config.voice_config import get_voice_config, get_voice_config_section

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realtime_voice")

# 从配置文件加载音频参数
audio_config = get_voice_config_section("audio_config")
RATE = audio_config.get("rate", 16000)
CHANNELS = audio_config.get("channels", 1)
CHUNK = audio_config.get("chunk", 1600)
FORMAT = getattr(pyaudio, audio_config.get("format", "paInt16"))

# 从配置文件加载VAD参数
vad_config = get_voice_config_section("vad_config")
VAD_WINDOW = vad_config.get("window", 30)
VAD_THRESHOLD = vad_config.get("threshold", 0.3)

# VAD模型初始化
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps = utils[0]
    logger.info("VAD模型加载成功")
except Exception as e:
    logger.error(f"VAD模型加载失败: {e}")
    raise RuntimeError("无法初始化VAD模型")

class AudioStream:
    """音频流处理类"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.vad_buffer = []
        self.speech_frames = []
        self.is_speaking = False
        self.silence_frames = 0
        
        # 从配置加载VAD参数
        self.MAX_SILENCE_FRAMES = vad_config.get("max_silence_frames", 15)
        self.min_speech_frames = vad_config.get("min_speech_frames", 5)
        self.audio_rms_threshold = vad_config.get("audio_rms_threshold", 200)
        self.last_process_time = 0
        
    def start_stream(self):
        """开始录音"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        self.is_recording = True
        logger.info("开始录音...")
        
    def stop_stream(self):
        """停止录音"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        logger.info("停止录音")
        
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    def detect_speech(self, audio_data):
        """检测是否有语音"""
        try:
            # 转换为numpy数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算RMS音量
            audio_rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
            
            # 音量过低，可能是背景噪音
            if audio_rms < self.audio_rms_threshold:
                return False
                
            # 归一化
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # 添加到VAD缓冲区
            self.vad_buffer.append(audio_float)
            
            # 保持VAD缓冲区大小
            if len(self.vad_buffer) > VAD_WINDOW:
                self.vad_buffer.pop(0)
                
            # 只有当累积足够的帧才进行VAD
            if len(self.vad_buffer) >= 3:
                # 合并音频片段
                audio_concat = np.concatenate(self.vad_buffer)
                # 检测语音
                speech_timestamps = get_speech_timestamps(
                    audio_concat, vad_model, 
                    threshold=VAD_THRESHOLD,
                    sampling_rate=RATE
                )
                # 如果检测到语音，且音量足够
                if len(speech_timestamps) > 0:
                    return True
        except Exception as e:
            logger.error(f"VAD处理异常: {e}")
        return False

    def preprocess_audio(self, audio_frames):
        """预处理音频数据以提高质量"""
        try:
            # 合并所有音频帧
            audio_data = b''.join(audio_frames)
            
            # 转换为numpy数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 如果音量太小，放大音频
            audio_max = np.max(np.abs(audio_np))
            if audio_max < 2000:  # 音量太小
                gain = min(32767 / (audio_max + 1), 5.0)  # 限制最大增益为5倍
                audio_np = np.clip(audio_np * gain, -32768, 32767).astype(np.int16)
            
            # 检查处理后的音频是否有效
            if np.max(np.abs(audio_np)) < 500:
                logger.warning("预处理后的音频音量仍然太低，可能无法识别")
                
            # 转回字节流
            return audio_np.tobytes()
            
        except Exception as e:
            logger.error(f"音频预处理异常: {e}")
            return b''.join(audio_frames)  # 出错时返回原始音频

    def process_frame(self):
        """处理一帧音频"""
        if self.audio_queue.empty():
            return None
        
        # 获取音频数据
        audio_data = self.audio_queue.get()
        
        # VAD检测
        is_speech = self.detect_speech(audio_data)
        
        # 状态转换逻辑
        if is_speech:
            # 如果检测到语音
            self.speech_frames.append(audio_data)
            self.silence_frames = 0
            if not self.is_speaking and len(self.speech_frames) >= self.min_speech_frames:
                audio_np = np.frombuffer(b''.join(self.speech_frames), dtype=np.int16)
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                if rms > self.audio_rms_threshold * 1.5:  # 确保音量足够
                    self.is_speaking = True
                    logger.info(f"检测到语音开始... (RMS: {rms:.2f})")
        elif self.is_speaking:
            # 如果正在说话但当前无语音
            self.speech_frames.append(audio_data)  # 还是添加进来，可能是短暂停顿
            self.silence_frames += 1
            
            # 如果静音持续一定时间，结束此次语音
            if self.silence_frames >= self.MAX_SILENCE_FRAMES:
                # 预处理音频以提高质量
                complete_audio = self.preprocess_audio(self.speech_frames)
                
                # 添加调试信息
                audio_np = np.frombuffer(complete_audio, dtype=np.int16)
                duration = len(audio_np) / RATE
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                logger.info(f"检测到语音结束 - 时长: {duration:.2f}秒, RMS音量: {rms:.2f}, 帧数: {len(self.speech_frames)}")
                
                # 检查音频数据是否有效（太短或音量太低则丢弃）
                if len(self.speech_frames) < 8 or rms < self.audio_rms_threshold:  # 语音过短或音量过低
                    logger.warning(f"丢弃无效语音: 长度过短或音量过低")
                    self.speech_frames = []
                    self.is_speaking = False
                    self.silence_frames = 0
                    return None
                
                # 检查是否包含有效语音（使用VAD再次验证）
                audio_float = audio_np.astype(np.float32) / 32768.0
                speech_timestamps = get_speech_timestamps(
                    audio_float, vad_model,
                    threshold=VAD_THRESHOLD,
                    sampling_rate=RATE
                )
                
                if len(speech_timestamps) == 0:
                    logger.warning("发送前VAD检测未检出语音，放弃发送")
                    self.speech_frames = []
                    self.is_speaking = False
                    self.silence_frames = 0
                    return None
                    
                self.speech_frames = []
                self.is_speaking = False
                self.silence_frames = 0
                return complete_audio
        
        return None

class RealtimeVoiceClient:
    def __init__(self, server_url: str = None):
        # 从配置获取ASR服务器URL
        asr_config = get_voice_config_section("asr_service")
        self.server_url = server_url or asr_config.get("server_url", "ws://127.0.0.1:8765")
        self.websocket = None
        self.is_connected = False
        
        # 使用AudioStream进行音频处理
        self.audio_stream = None
        self.is_streaming = False
        
        # 事件和状态
        self.response_received = asyncio.Event()
        self.current_response = None
        self._text_ready = asyncio.Event()
        self.final_result = None
        self._final_text = None
        
        # 从配置获取API密钥
        self.api_key = asr_config.get("api_key", "dd91538f5918826f2bdf881e88fe9956")
        self.timeout = asr_config.get("timeout", 10.0)

    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            # 创建WebSocket连接
            self.websocket = await websockets.connect(self.server_url)
            
            # 等待认证请求
            auth_msg = await self.websocket.recv()
            auth_data = json.loads(auth_msg)
            
            if (auth_data.get("type") == "event" and 
                auth_data.get("event") == "auth_required"):
                # 发送认证信息
                await self.websocket.send(json.dumps({
                    "type": "auth",
                    "api_key": self.api_key
                }))
                
                # 等待认证结果
                auth_result = await self.websocket.recv()
                result_data = json.loads(auth_result)
                
                if (result_data.get("type") == "event" and 
                    result_data.get("event") == "connection_established"):
                    logger.info("认证成功")
                    self.is_connected = True
                    # 启动消息处理循环
                    asyncio.create_task(self.message_handler())
                    return True
                else:
                    logger.error(f"认证失败: {result_data}")
                    return False
            else:
                logger.error(f"未收到认证请求: {auth_data}")
                return False
                
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            return False
            
    async def message_handler(self):
        """处理服务器消息"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                if message == '{"type":"ping"}':
                    await self.websocket.send('{"type":"pong"}')
                    continue
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket连接已关闭")
            self.is_connected = False
        except Exception as e:
            logger.error(f"消息处理异常: {str(e)}")
            self.is_connected = False

    async def handle_message(self, message: str):
        """处理单条消息"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "response":
                # 记录完整响应数据，便于调试
                logger.info(f"收到完整响应: {data}")
                
                if data.get("success"):
                    if "text" in data:
                        response = {
                            "text": data["text"],
                            "user": data.get("user", "Unknown"),
                            "voice_match": data.get("voice_match", False)
                        }
                        self.current_response = response
                        self.response_received.set()
                        logger.info(f"收到识别响应: {response}")
                else:
                    error = data.get("error", "未知错误")
                    error_code = data.get("code", "UNKNOWN_ERROR")
                    
                    # 处理不同类型的错误
                    if error_code == "UNREGISTERED_USER":
                        # 未注册用户仍然可以识别文本
                        logger.info(f"未注册用户语音: {error}")
                        text = data.get("text", "")
                        if text:
                            response = {
                                "text": text,
                                "user": "Unknown",
                                "voice_match": False
                            }
                            self.current_response = response
                            self.response_received.set()
                        else:
                            self.current_response = {"error": error, "code": error_code}
                            self.response_received.set()
                    elif error_code == "ASR_FAILED" and "No speech detected" in error:
                        logger.info("服务器VAD未检测到语音，可能是音量过低")
                        self.current_response = {"error": error, "code": error_code}
                        self.response_received.set()
                    else:
                        logger.warning(f"识别失败: [{error_code}] {error}")
                        self.current_response = {"error": error, "code": error_code}
                        self.response_received.set()
                
            elif data.get("type") == "error":
                logger.error(f"服务器错误: {data.get('message', '未知错误')}")
                self.current_response = {"error": data.get("message", "服务器错误")}
                self.response_received.set()
                
        except Exception as e:
            logger.error(f"处理消息异常: {e}")
            self.current_response = {"error": f"处理消息异常: {str(e)}"}
            self.response_received.set()

    async def start_stream(self):
        """开始音频流"""
        if self.is_streaming:
            return
            
        # 创建AudioStream实例
        self.audio_stream = AudioStream()
        self.audio_stream.start_stream()
        self.is_streaming = True
        logger.info("开始音频流")
        
        # 启动音频处理循环
        asyncio.create_task(self._process_audio())

    async def _process_audio(self):
        """处理音频流"""
        try:
            while self.is_streaming:
                # 使用AudioStream的process_frame方法处理音频
                complete_audio = self.audio_stream.process_frame()
                
                if complete_audio:
                    # 发送音频进行识别
                    await self.send_audio(complete_audio)
                
                await asyncio.sleep(0.01)  # 避免CPU过载
                
        except Exception as e:
            logger.error(f"音频处理循环异常: {e}")

    async def stop_stream(self):
        """停止音频流"""
        self.is_streaming = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream = None
        logger.info("停止音频流")
            
    async def send_audio(self, audio_data: bytes):
        """发送音频进行识别"""
        if not self.websocket or not self.is_connected:
            logger.error("WebSocket未连接")
            return
            
        try:
            # 重置事件状态
            self.response_received.clear()
            self.current_response = None
            
            # 音频质量检查
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
            duration = len(audio_np) / RATE
            logger.info(f"发送音频数据 - 时长: {duration:.2f}秒, RMS音量: {rms:.2f}")
            
            # Base64编码
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 重新连接检查
            if not self.is_connected:
                logger.warning("连接已断开，尝试重新连接")
                if not await self.connect():
                    raise ConnectionError("重新连接失败")

            # 记录开始时间
            start_time = time.time()
            
            # 发送请求
            request = {
                "type": "request",
                "command": "recognize_audio",
                "data": {
                    "audio_data": audio_base64,
                    "check_voiceprint": True,
                    "only_register_user": False,  # 允许识别所有用户
                    "identify_unregistered": True  # 启用识别未注册用户的语音
                },
                "request_id": str(uuid.uuid4())
            }
            
            await self.websocket.send(json.dumps(request))
            logger.info(f"已发送音频数据进行识别,大小: {len(audio_data)} bytes")
            
            # 等待响应
            try:
                await asyncio.wait_for(self.response_received.wait(), timeout=self.timeout)
                if self.current_response:
                    response = self.current_response
                    self.current_response = None  # 清除当前响应
                    
                    if "text" in response:
                        # 获取用户信息并格式化文本
                        recognized_text = response["text"]
                        user = response.get("user", "Unknown")
                        voice_match = response.get("voice_match", False)
                        
                        # 打印详细的用户识别信息
                        logger.info(f"识别结果 - 用户: {user}, 声纹匹配: {voice_match}")
                        
                        # 格式化文本（添加用户标签）
                        if user and user != "Unknown" and user.lower() != "unknown":
                            # 已注册用户
                            formatted_text = f"{recognized_text}[{user}]"
                            logger.info(f"已识别为注册用户: {user}")
                        else:
                            # 未注册用户
                            formatted_text = f"{recognized_text}[访客]"
                            logger.info("已识别为访客")
                        
                        # 更新带有用户标识的文本
                        response["text"] = formatted_text
                        self.final_result = response
                        self._final_text = formatted_text
                        self._text_ready.set()
                        logger.info(f"最终识别结果: {formatted_text}")
                    elif "error" in response:
                        error_code = response.get("code", "UNKNOWN_ERROR")
                        error_msg = response.get("error", "未知错误")
                        
                        if error_code == "UNREGISTERED_USER" and "text" in response:
                            # 未注册用户但有识别文本
                            recognized_text = response["text"]
                            formatted_text = f"{recognized_text}[访客]"
                            response["text"] = formatted_text
                            self.final_result = response
                            self._final_text = formatted_text
                            self._text_ready.set()
                            logger.info(f"未注册用户识别结果: {formatted_text}")
                        else:
                            logger.warning(f"识别错误: {error_code}: {error_msg}")
                            self.final_result = {"error": error_msg, "code": error_code}
                            self._text_ready.set()
                    else:
                        logger.warning("响应中未包含文本内容或错误信息")
                        self.final_result = {"error": "未收到有效内容"}
                        self._text_ready.set()
                else:
                    logger.warning("未收到有效响应")
                    self.final_result = {"error": "未收到响应"}
                    self._text_ready.set()
            except asyncio.TimeoutError:
                logger.error(f"等待响应超时，耗时: {time.time() - start_time:.2f}秒")
                self.final_result = {"error": "等待响应超时"}
                self._text_ready.set()
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"发送音频时连接关闭: {e}")
            self.is_connected = False
            self.final_result = {"error": "连接已关闭"}
            self._text_ready.set()
        except Exception as e:
            logger.error(f"发送音频异常: {e}")
            self.final_result = {"error": str(e)}
            self._text_ready.set()

    async def wait_for_result(self, timeout: float = None) -> Optional[Dict[str, Any]]:
        """等待识别结果"""
        # 如果未指定超时，使用配置值
        if timeout is None:
            timeout = self.timeout
            
        try:
            # 增加超时时间并添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await asyncio.wait_for(self._text_ready.wait(), timeout=timeout)
                    if self.final_result and "error" in self.final_result:
                        if attempt < max_retries - 1:
                            logger.warning(f"等待结果失败，尝试重试 ({attempt + 1}/{max_retries})")
                            self._text_ready.clear()
                            continue
                    return self.final_result
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        logger.warning(f"等待超时，尝试重试 ({attempt + 1}/{max_retries})")
                        continue
                    logger.warning("等待识别结果最终超时")
                    return {"error": "识别超时"}
        except Exception as e:
            logger.error(f"等待结果异常: {e}")
            return {"error": str(e)}
        finally:
            self._text_ready.clear()
            
    async def close(self):
        """关闭客户端"""
        try:
            await self.stop_stream()
            
            if self.websocket:
                await self.websocket.close()
                self.is_connected = False
                
            logger.info("客户端已关闭")
        except Exception as e:
            logger.error(f"关闭客户端异常: {e}")
