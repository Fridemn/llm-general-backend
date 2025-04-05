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

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("realtime_voice")

# 音频配置
RATE = 16000
CHANNELS = 1
CHUNK = 1600  # 100ms
FORMAT = pyaudio.paInt16
VAD_WINDOW = 30  # VAD窗口大小(帧数)

# VAD模型初始化
try:
    vad_model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps = utils[0]
    VAD_THRESHOLD = 0.3  # 降低VAD阈值以提高灵敏度
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
        self.MAX_SILENCE_FRAMES = 10  
        self.min_speech_frames = 3
        self.audio_rms_threshold = 180
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
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
            
            if audio_rms < self.audio_rms_threshold:
                return False
                
            audio_float = audio_np.astype(np.float32) / 32768.0
            self.vad_buffer.append(audio_float)
            
            if len(self.vad_buffer) > VAD_WINDOW:
                self.vad_buffer.pop(0)
                
            if len(self.vad_buffer) >= 3:
                audio_concat = np.concatenate(self.vad_buffer)
                speech_timestamps = get_speech_timestamps(
                    audio_concat, vad_model,
                    threshold=VAD_THRESHOLD,
                    sampling_rate=RATE
                )
                return len(speech_timestamps) > 0
        except Exception as e:
            logger.error(f"VAD处理异常: {e}")
        return False

    def preprocess_audio(self, audio_frames):
        """预处理音频数据以提高质量"""
        try:
            audio_data = b''.join(audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            audio_max = np.max(np.abs(audio_np))
            if audio_max < 2000:
                gain = min(32767 / (audio_max + 1), 3.0)
                audio_np = np.clip(audio_np * gain, -32768, 32767).astype(np.int16)
            
            return audio_np.tobytes()
        except Exception as e:
            logger.error(f"音频预处理异常: {e}")
            return b''.join(audio_frames)

    def process_frame(self):
        """处理一帧音频"""
        if self.audio_queue.empty():
            return None
        
        audio_data = self.audio_queue.get()
        is_speech = self.detect_speech(audio_data)
        
        if is_speech:
            self.speech_frames.append(audio_data)
            self.silence_frames = 0
            if not self.is_speaking and len(self.speech_frames) >= self.min_speech_frames:
                audio_np = np.frombuffer(b''.join(self.speech_frames), dtype=np.int16)
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                if rms > self.audio_rms_threshold:
                    self.is_speaking = True
                    logger.info(f"检测到语音开始... (RMS: {rms:.2f})")
        elif self.is_speaking:
            self.speech_frames.append(audio_data)
            self.silence_frames += 1
            
            if self.silence_frames >= self.MAX_SILENCE_FRAMES:
                complete_audio = b''.join(self.speech_frames)
                
                audio_np = np.frombuffer(complete_audio, dtype=np.int16)
                duration = len(audio_np) / 16000
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                logger.info(f"检测到语音结束 - 时长: {duration:.2f}秒, RMS音量: {rms:.2f}, 帧数: {len(self.speech_frames)}")
                
                if len(self.speech_frames) < 5 or rms < self.audio_rms_threshold * 0.7:
                    logger.warning(f"丢弃无效语音: 长度过短或音量过低")
                    self.speech_frames = []
                    self.is_speaking = False
                    self.silence_frames = 0
                    return None
                    
                audio_data = self.preprocess_audio(self.speech_frames)
                self.speech_frames = []
                self.is_speaking = False
                self.silence_frames = 0
                return audio_data
        
        return None

class RealtimeVoiceClient:
    def __init__(self, server_url: str = "ws://127.0.0.1:8765"):
        self.server_url = server_url
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
        
        # API密钥
        self.api_key = "dd91538f5918826f2bdf881e88fe9956"

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
                if data.get("success"):
                    if "text" in data:
                        response = {
                            "text": data["text"],
                            "user": data.get("user", "Unknown")
                        }
                        self.current_response = response
                        self.response_received.set()
                        logger.info(f"收到识别响应: {response}")
                else:
                    error = data.get("error", "未知错误")
                    logger.warning(f"识别失败: {error}")
                    self.current_response = {"error": error}
                    self.response_received.set()
                
            elif data.get("type") == "error":
                logger.error(f"服务器错误: {data.get('message', '未知错误')}")
                self.current_response = {"error": data.get("message", "服务器错误")}
                self.response_received.set()
                
        except Exception as e:
            logger.error(f"处理消息异常: {e}")
            self.current_response = {"error": f"处理消息异常: {str(e)}"}
            self.response_received.set()

    def detect_speech(self, audio_data: bytes) -> bool:
        """检测是否有语音"""
        try:
            # 转换为numpy数组并计算RMS音量
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
            
            # 音量过低判断
            if audio_rms < self.audio_rms_threshold:
                return False
                
            # 归一化并处理VAD
            audio_float = audio_np.astype(np.float32) / 32768.0
            self.vad_buffer.append(audio_float)
            
            # 保持VAD窗口大小
            if len(self.vad_buffer) > 30:  # VAD窗口
                self.vad_buffer.pop(0)
                
            # 执行VAD检测
            if len(self.vad_buffer) >= 3:
                audio_concat = np.concatenate(self.vad_buffer)
                speech_timestamps = self.get_speech_timestamps(
                    audio_concat,
                    self.vad_model,
                    threshold=self.vad_threshold,
                    sampling_rate=self.rate
                )
                return len(speech_timestamps) > 0
                
        except Exception as e:
            logger.error(f"VAD处理异常: {e}")
            
        return False
        
    async def process_frame(self, frame_data: bytes) -> Optional[bytes]:
        """处理单帧音频"""
        try:
            # VAD检测
            is_speech = self.detect_speech(frame_data)
            
            if is_speech:
                # 检测到语音
                self.speech_frames.append(frame_data)
                self.silence_frames = 0
                
                if not self.is_speaking and len(self.speech_frames) >= self.min_speech_frames:
                    audio_np = np.frombuffer(b''.join(self.speech_frames), dtype=np.int16)
                    rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                    
                    if rms > self.audio_rms_threshold:
                        self.is_speaking = True
                        logger.info(f"检测到语音开始... (RMS: {rms:.2f})")
                        
            elif self.is_speaking:
                # 当前无语音但正在说话状态
                self.speech_frames.append(frame_data)
                self.silence_frames += 1
                
                # 静音超过阈值,结束此次语音
                if self.silence_frames >= self.max_silence_frames:
                    complete_audio = b''.join(self.speech_frames)
                    
                    # 添加调试信息
                    audio_np = np.frombuffer(complete_audio, dtype=np.int16) 
                    duration = len(audio_np) / self.rate
                    rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                    logger.info(f"检测到语音结束 - 时长: {duration:.2f}秒, RMS: {rms:.2f}")
                    
                    # 简化有效性检查
                    if len(self.speech_frames) < 5 or rms < self.audio_rms_threshold * 0.7:
                        logger.info("丢弃无效语音: 长度过短或音量过低")
                        self.reset_speech_state()
                        return None
                        
                    self.reset_speech_state()
                    return self.preprocess_audio(complete_audio)
                    
        except Exception as e:
            logger.error(f"处理音频帧异常: {e}")
            
        return None

    def preprocess_audio(self, audio_data: bytes) -> bytes:
        """预处理音频数据以提高质量"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 简化处理逻辑，只进行基本增益调整
            audio_max = np.max(np.abs(audio_np))
            if (audio_max < 2000):  # 音量太小
                gain = min(32767 / (audio_max + 1), 3.0)  # 降低最大增益为3倍
                audio_np = np.clip(audio_np * gain, -32768, 32767).astype(np.int16)
            
            return audio_np.tobytes()
        except Exception as e:
            logger.error(f"音频预处理异常: {e}")
            return audio_data

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
        
    async def handle_audio_frame(self, frame_data: bytes):
        """处理音频回调帧"""
        try:
            complete_audio = await self.process_frame(frame_data)
            
            if complete_audio:
                # 发送音频进行识别
                await self.send_audio(complete_audio)
                
        except Exception as e:
            logger.error(f"处理音频帧异常: {e}")
            
    async def send_audio(self, audio_data: bytes):
        """发送音频进行识别"""
        if not self.websocket or not self.is_connected:
            logger.error("WebSocket未连接")
            return
            
        try:
            # 重置事件状态
            self.response_received.clear()
            self.current_response = None
            
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
                    "only_register_user": True
                },
                "request_id": str(uuid.uuid4())  # 添加请求ID
            }
            
            await self.websocket.send(json.dumps(request))
            logger.info(f"已发送音频数据进行识别,大小: {len(audio_data)} bytes")
            
            # 等待响应
            try:
                await asyncio.wait_for(self.response_received.wait(), timeout=5.0)
                if self.current_response:
                    response = self.current_response
                    self.current_response = None  # 清除当前响应
                    if "text" in response:
                        self.final_result = response
                        self._final_text = response["text"]
                        self._text_ready.set()
                        logger.info(f"收到识别结果: {response['text']}")
                    else:
                        logger.warning("响应中未包含文本内容")
                        self.final_result = {"error": "未收到有效文本"}
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

    async def wait_for_result(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """等待识别结果"""
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
                
            if self.audio:
                self.audio.terminate()
                
            logger.info("客户端已关闭")
        except Exception as e:
            logger.error(f"关闭客户端异常: {e}")
