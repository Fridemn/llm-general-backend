import asyncio
import base64
import json
import logging
import numpy as np
import pyaudio
import websockets
import time
import uuid
import queue
import threading
from datetime import datetime
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("asr_client")

# WebSocket服务器配置
WS_SERVER = "ws://localhost:8765"
API_KEY = "dd91538f5918826f2bdf881e88fe9956"  # 替换为您的API密钥

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
    VAD_THRESHOLD = 0.3  # 降低VAD阈值以提高灵敏度（原值0.5）
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
        self.MAX_SILENCE_FRAMES = 10  # 进一步减少静音帧阈值，更快结束检测（原值15）
        self.min_speech_frames = 3  # 减少最少帧数要求，更快开始检测（原值5）
        self.audio_rms_threshold = 180  # 稍微降低RMS阈值，提高灵敏度（原值200）
        self.last_process_time = 0  # 用于跟踪上次处理时间
        
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
        """预处理音频数据以提高质量
        
        Args:
            audio_frames: 音频帧列表
            
        Returns:
            bytes: 处理后的音频数据
        """
        try:
            # 合并所有音频帧
            audio_data = b''.join(audio_frames)
            
            # 转换为numpy数组
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # 简化处理逻辑，只进行最基本的增益调整
            audio_max = np.max(np.abs(audio_np))
            if audio_max < 2000:  # 音量太小
                gain = min(32767 / (audio_max + 1), 3.0)  # 降低最大增益为3倍（原来是5倍）
                audio_np = np.clip(audio_np * gain, -32768, 32767).astype(np.int16)
            
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
                if rms > self.audio_rms_threshold:  # 降低音量要求（去掉了*1.5）
                    self.is_speaking = True
                    logger.info(f"检测到语音开始... (RMS: {rms:.2f})")
        elif self.is_speaking:
            # 如果正在说话但当前无语音
            self.speech_frames.append(audio_data)  # 还是添加进来，可能是短暂停顿
            self.silence_frames += 1
            
            # 如果静音持续一定时间，结束此次语音
            if self.silence_frames >= self.MAX_SILENCE_FRAMES:
                # 直接合并音频数据，减少处理时间
                complete_audio = b''.join(self.speech_frames)
                
                # 添加调试信息
                audio_np = np.frombuffer(complete_audio, dtype=np.int16)
                duration = len(audio_np) / RATE
                rms = np.sqrt(np.mean(np.square(audio_np.astype(np.float32))))
                logger.info(f"检测到语音结束 - 时长: {duration:.2f}秒, RMS音量: {rms:.2f}, 帧数: {len(self.speech_frames)}")
                
                # 简化有效性检查
                if len(self.speech_frames) < 5 or rms < self.audio_rms_threshold * 0.7:  # 降低有效性要求
                    logger.warning(f"丢弃无效语音: 长度过短或音量过低")
                    self.speech_frames = []
                    self.is_speaking = False
                    self.silence_frames = 0
                    return None
                    
                self.speech_frames = []
                self.is_speaking = False
                self.silence_frames = 0
                
                # 简化最终检查，直接返回音频
                return complete_audio
        
        return None

class WSClient:
    """WebSocket客户端类"""
    
    def __init__(self, server_url, api_key):
        self.server_url = server_url
        self.api_key = api_key
        self.websocket = None
        self.request_queue = {}
        
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info(f"已连接到服务器: {self.server_url}")
            
            # 等待认证请求
            auth_msg = await self.websocket.recv()
            auth_data = json.loads(auth_msg)
            
            if auth_data.get("type") == "event" and auth_data.get("event") == "auth_required":
                # 发送认证信息
                await self.websocket.send(json.dumps({
                    "type": "auth",
                    "api_key": self.api_key
                }))
                
                # 等待认证结果
                auth_result = await self.websocket.recv()
                result_data = json.loads(auth_result)
                
                if result_data.get("type") == "event" and result_data.get("event") == "connection_established":
                    logger.info("认证成功")
                    return True
                else:
                    logger.error(f"认证失败: {result_data}")
                    return False
            else:
                logger.error(f"未收到认证请求: {auth_data}")
                return False
                
        except Exception as e:
            logger.error(f"连接服务器失败: {e}")
            return False
    
    async def send_audio(self, audio_data):
        """发送音频数据进行识别"""
        if not self.websocket:
            logger.error("未连接到服务器")
            return None
            
        try:
            # 生成请求ID
            request_id = str(uuid.uuid4())
            
            # 简化音频质量检查
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            duration = len(audio_np) / RATE
            logger.info(f"发送音频数据 - 时长: {duration:.2f}秒")
            
            # Base64编码
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 构建请求
            request = {
                "type": "request",
                "request_id": request_id,
                "command": "recognize_audio",
                "data": {
                    "audio_data": audio_base64,
                    "check_voiceprint": True,
                    "only_register_user": True  # 只识别注册用户
                }
            }
            
            # 发送请求
            await self.websocket.send(json.dumps(request))
            logger.info(f"已发送音频识别请求: {request_id}, 大小: {len(audio_data)} bytes")
            
            # 等待响应，减少超时时间从10秒到5秒
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
            except asyncio.TimeoutError:
                logger.error("等待服务器响应超时")
                return {"success": False, "error": "服务器响应超时"}
            except json.JSONDecodeError:
                logger.error("服务器返回的响应不是有效的JSON格式")
                return {"success": False, "error": "服务器返回格式错误", "code": "JSON_DECODE_ERROR"}
            
            # 处理响应
            if (response_data.get("type") == "response" and 
                response_data.get("request_id") == request_id):
                if response_data.get("success"):
                    return {
                        "success": True,
                        "text": response_data.get("text", ""),
                        "user": response_data.get("user", "Unknown")
                    }
                else:
                    error_code = response_data.get("code", "UNKNOWN_ERROR")
                    error_msg = response_data.get("error", "未知错误")
                    
                    # 针对不同错误提供更友好的处理
                    if error_code == "ASR_FAILED" and "No speech detected" in error_msg:
                        logger.warning(f"服务器VAD未检测到语音，可能是音量过低")
                    elif error_code == "UNREGISTERED_USER":
                        logger.info(f"未注册用户的语音，不进行识别")
                    elif error_code == "PROCESS_ERROR" and "JSON serializable" in error_msg:
                        error_detail = error_msg.split("JSON serializable", 1)[1] if "JSON serializable" in error_msg else ""
                        logger.warning(f"服务器JSON序列化错误: {error_detail}")
                    else:
                        logger.warning(f"识别失败: {error_code}: {error_msg}")
                    
                    return {
                        "success": False,
                        "error": f"{error_code}: {error_msg}",
                        "user": response_data.get("user", "Unknown"),
                        "code": error_code
                    }
            else:
                logger.error(f"收到非预期响应: {response_data}")
                return {"success": False, "error": "非预期响应"}
                
        except Exception as e:
            logger.error(f"发送音频异常: {e}")
            return {"success": False, "error": str(e)}
    
    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            logger.info("已关闭WebSocket连接")

async def main():
    """主函数"""
    # 初始化音频流
    audio_stream = AudioStream()
    
    # 初始化WebSocket客户端
    ws_client = WSClient(WS_SERVER, API_KEY)
    
    try:
        # 连接到服务器
        connected = await ws_client.connect()
        if not connected:
            logger.error("无法连接到服务器，程序退出")
            return
        
        # 开始录音
        audio_stream.start_stream()
        
        print("\n===== 实时语音识别已启动 =====")
        print("只有注册过声纹的用户语音才会被识别")
        print("按Ctrl+C退出程序\n")
        
        # 用于跟踪连续错误次数
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        # 主循环
        while True:
            # 处理音频帧
            complete_audio = audio_stream.process_frame()
            
            # 如果有完整的语音段
            if complete_audio:
                # 发送音频进行识别
                result = await ws_client.send_audio(complete_audio)
                
                # 处理识别结果
                if result and result.get("success"):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    user = result.get("user", "Unknown")
                    text = result.get("text", "")
                    print(f"[{timestamp}] {user}: {text}")
                    consecutive_errors = 0  # 重置错误计数
                elif result:
                    error = result.get("error", "未知错误")
                    user = result.get("user", "Unknown")
                    error_code = result.get("code", "")
                    
                    if error_code == "UNREGISTERED_USER":
                        logger.info(f"未注册用户的语音，不进行识别")
                        consecutive_errors = 0  # 这不算作错误
                    elif error_code == "PROCESS_ERROR" and "JSON serializable" in error:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 服务器处理错误: JSON序列化问题")
                        print(f"  详情: {error}")
                        consecutive_errors += 1
                    elif error_code == "ASR_FAILED" and "No speech detected" in error:
                        logger.info(f"服务器未检测到语音，可能是背景噪音")
                        consecutive_errors = 0  # 这不算作错误
                    else:
                        logger.warning(f"识别失败: {error}, 用户: {user}")
                        consecutive_errors += 1
                
                # 如果连续错误次数过多，提示可能需要重启服务器
                if consecutive_errors >= max_consecutive_errors:
                    print("\n⚠️ 检测到多次连续错误，服务器可能需要重启")
                    print("  您可以尝试重新启动服务器或客户端\n")
                    consecutive_errors = 0  # 重置计数，避免重复提示
            
            # 短暂等待，减少CPU使用
            await asyncio.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"运行异常: {e}")
    finally:
        # 清理资源
        audio_stream.stop_stream()
        await ws_client.close()
        print("\n===== 实时语音识别已停止 =====")

if __name__ == "__main__":
    # 启动主函数
    asyncio.run(main())
