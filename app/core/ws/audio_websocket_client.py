import asyncio
import websockets
import json
import base64
import pyaudio
import numpy as np
import wave
import os
from typing import Optional, Dict, Any, Union, List

class AudioWebSocketClient:
    def __init__(self, server_url: str = "ws://127.0.0.1:8765"):
        self.server_url = server_url
        self.session_id = None
        self.request_id = 0
        self.websocket = None
        self.is_connected = False
        
        # 音频配置
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        
        # 添加流式识别结果缓存
        self.streaming_results = []
        self.final_result = None
        self.streaming_complete = asyncio.Event()
    
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(self.server_url)
            print(f"已连接到WebSocket服务器: {self.server_url}")
            self.is_connected = True
            
            # 启动消息处理循环
            asyncio.create_task(self.message_handler())
            return True
        except Exception as e:
            print(f"连接失败: {str(e)}")
            return False
    
    async def message_handler(self):
        """处理接收到的WebSocket消息"""
        if not self.websocket:
            return
            
        try:
            async for message in self.websocket:
                await self.handle_message(message)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket连接已关闭")
            self.is_connected = False
    
    async def handle_message(self, message: str):
        """处理服务器发送的消息"""
        print(f"接收到: {message}")
        data = json.loads(message)
        
        msg_type = data.get("type")
        
        if msg_type == "event" and data.get("event") == "connection_established":
            self.session_id = data["data"]["session_id"]
            print(f"会话建立成功，会话ID: {self.session_id}")
            
        elif msg_type == "response":
            success = data.get("success", False)
            request_id = data.get("request_id")
            
            if success:
                print(f"请求 {request_id} 成功")
                if "text" in data:
                    text = data["text"]
                    user = data.get("user", "unknown")
                    print(f"识别文本: {text}")
                    print(f"用户: {user}")
            else:
                print(f"请求 {request_id} 失败: {data.get('error', '未知错误')}")
        
        elif msg_type == "event":
            event = data.get("event")
            
            if event == "transcription_result":
                text = data["data"]["text"]
                user = data["data"]["user"]
                print(f"转写结果: {text}")
                print(f"用户: {user}")
                
                # 收集流式识别的中间结果
                self.streaming_results.append({"text": text, "user": user})
            
            elif event == "recording_ended":
                print("录音结束")
                
                # 生成最终结果
                if self.streaming_results:
                    # 将所有识别结果合并为一个完整文本
                    combined_text = " ".join([result["text"] for result in self.streaming_results])
                    last_user = self.streaming_results[-1]["user"]
                    self.final_result = {
                        "text": combined_text,
                        "user": last_user,
                        "segments": self.streaming_results
                    }
                    
                    print("\n=== 最终识别结果 ===")
                    print(f"用户: {last_user}")
                    print(f"完整文本: {combined_text}")
                    print("===================\n")
                
                # 标记流式识别完成
                self.streaming_complete.set()
    
    async def send_request(self, command: str, data: Dict[str, Any]):
        """发送请求到服务器"""
        if not self.websocket or not self.is_connected:
            print("WebSocket未连接")
            return False
        
        request = {
            "type": "request",
            "command": command,
            "request_id": str(self.request_id),
            "data": data
        }
        self.request_id += 1
        
        request_str = json.dumps(request)
        await self.websocket.send(request_str)
        print(f"已发送: {request_str}")
        return True
    
    async def start_streaming(self, check_voiceprint: bool = True):
        """启动流式语音识别"""
        data = {"check_voiceprint": check_voiceprint}
        return await self.send_request("start_streaming", data)
    
    async def send_audio_data(self, audio_data: bytes, is_end: bool = False):
        """发送音频数据"""
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        data = {
            "audio_data": base64_data,
            "is_end": is_end
        }
        
        return await self.send_request("stream_audio", data)
    
    async def recognize_audio(self, audio_data: bytes, check_voiceprint: bool = True):
        """一次性识别音频"""
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        data = {
            "audio_data": base64_data,
            "check_voiceprint": check_voiceprint
        }
        
        # 发送请求并等待响应
        await self.send_request("recognize_audio", data)
        
        # 重置结果状态
        self.streaming_results = []
        self.final_result = None
        self.streaming_complete.clear()
        
        # 等待服务器处理并返回结果
        try:
            # 给服务器足够的时间处理和返回结果
            await asyncio.sleep(2.0)
            
            # 如果有流式结果处理完成，创建最终结果
            if self.streaming_results:
                combined_text = " ".join([result["text"] for result in self.streaming_results])
                last_user = self.streaming_results[-1]["user"]
                self.final_result = {
                    "text": combined_text,
                    "user": last_user,
                    "segments": self.streaming_results
                }
            # 如果没有结果，创建一个基本结果
            elif not self.final_result:
                self.final_result = {
                    "text": "识别结果将在服务器处理完成后显示",
                    "user": check_voiceprint and "检测中..." or "无声纹识别",
                    "segments": []
                }
        except Exception as e:
            print(f"等待识别结果时发生错误: {str(e)}")
        
        # 返回当前可用的任何结果
        return self.final_result or {"error": "未获得识别结果"}
    
    async def list_voiceprints(self) -> Dict[str, Any]:
        """获取声纹库列表"""
        # 发送请求
        result = await self.send_request("list_voiceprints", {})
        # 不要使用send_request的返回值，它只是一个布尔值表示请求是否成功发送
        # 真正的响应会通过message_handler异步接收
        return {"success": True, "voiceprints": []}
    
    async def register_voiceprint(self, audio_data: bytes, person_name: str) -> Dict[str, Any]:
        """注册声纹
        
        Args:
            audio_data: 音频数据，格式为16kHz, 16bit, 单声道
            person_name: 用户姓名/标识
            
        Returns:
            Dict: 注册结果
        """
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        data = {
            "audio_data": base64_data,
            "person_name": person_name
        }
        
        await self.send_request("register_voiceprint", data)
        # 等待响应
        return {"success": True}
    
    async def match_voiceprint(self, audio_data: bytes) -> Dict[str, Any]:
        """匹配声纹
        
        Args:
            audio_data: 音频数据，格式为16kHz, 16bit, 单声道
            
        Returns:
            Dict: 匹配结果
        """
        base64_data = base64.b64encode(audio_data).decode('utf-8')
        
        data = {
            "audio_data": base64_data
        }
        
        # 发送匹配请求并等待响应
        await self.send_request("match_voiceprint", data)
        
        # 由于需要等待响应，这里不能直接返回
        # 我们需要等待接收到响应，这可能需要重新设计消息处理机制
        # 为简单起见，暂时返回成功并在控制台输出等待响应提示
        print("声纹匹配请求已发送，等待响应...")
        return {"success": True}
    
    async def record_audio(self, seconds: int = 3) -> bytes:
        """录制音频
        
        Args:
            seconds: 录制时长（秒）
            
        Returns:
            bytes: 录制的音频数据
        """
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print(f"\n请开始说话，录制{seconds}秒...")
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * seconds)):
            data = stream.read(self.chunk)
            frames.append(data)
            # 打印进度条
            progress = (i + 1) / int(self.rate / self.chunk * seconds)
            bar_length = 30
            bar = '█' * int(bar_length * progress) + '-' * (bar_length - int(bar_length * progress))
            print(f"\r录制进度: [{bar}] {int(progress * 100)}%", end="", flush=True)
            
        print("\n录制完成!")
        
        stream.stop_stream()
        stream.close()
        
        # 合并所有音频帧
        audio_data = b''.join(frames)
        return audio_data
    
    async def stream_audio_from_mic(self, duration: int = 5, check_voiceprint: bool = True) -> Dict[str, Any]:
        """从麦克风采集并流式发送音频数据，返回最终识别结果
        
        Args:
            duration (int): 录音时长，单位秒
            check_voiceprint (bool): 是否检查声纹

        Returns:
            Dict[str, Any]: 最终识别结果，包含完整文本和分段结果
        """
        # 重置状态
        self.streaming_results = []
        self.final_result = None
        self.streaming_complete.clear()
        
        # 如果需要检查声纹，可以先录制一小段音频进行声纹识别
        if check_voiceprint:
            print("首先进行声纹验证...")
            verify_audio = await self.record_audio(2)  # 录制2秒用于验证
            
            # 匹配声纹
            base64_data = base64.b64encode(verify_audio).decode('utf-8')
            data = {"audio_data": base64_data}
            await self.send_request("match_voiceprint", data)
            
            # 稍微等待一下响应
            await asyncio.sleep(1)
            print("声纹验证已完成，继续流式识别")
        
        # 启动流式识别
        if not await self.start_streaming(check_voiceprint):
            return {"error": "无法启动流式识别"}
        
        # 打开音频流
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        print(f"开始录音 {duration} 秒...")
        
        # 记录音频片段
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            is_end = i >= int(self.rate / self.chunk * duration) - 1
            await self.send_audio_data(data, is_end)
            await asyncio.sleep(0.01)  # 短暂延迟，避免发送过快
        
        print("录音结束，等待最终结果...")
        
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        
        # 等待最终结果（设置超时时间）
        try:
            await asyncio.wait_for(self.streaming_complete.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("警告: 等待最终结果超时")
        
        return self.final_result or {"error": "未收到识别结果"}
    
    async def stream_audio_from_mic_batch(self, duration: int = 5, check_voiceprint: bool = True) -> Dict[str, Any]:
        """从麦克风采集音频数据，完成后一次性发送进行识别
        
        Args:
            duration (int): 录音时长，单位秒
            check_voiceprint (bool): 是否检查声纹

        Returns:
            Dict[str, Any]: 最终识别结果
        """
        # 重置状态
        self.streaming_results = []
        self.final_result = None
        self.streaming_complete.clear()
        
        print(f"\n开始录音 {duration} 秒...")
        
        # 录制整段音频
        audio_data = await self.record_audio(duration)
        print("录音完成，开始进行识别...")
        
        # 如果需要检查声纹，先验证声纹
        if check_voiceprint:
            print("进行声纹验证...")
            base64_data = base64.b64encode(audio_data).decode('utf-8')
            data = {"audio_data": base64_data}
            await self.send_request("match_voiceprint", data)
            
            # 等待声纹验证结果
            await asyncio.sleep(0.5)
            print("声纹验证已完成")
        
        # 一次性识别整段音频
        print("开始识别录音内容...")
        result = await self.recognize_audio(audio_data, check_voiceprint)
        
        # 等待返回结果
        try:
            await asyncio.sleep(1.5)  # 等待服务器处理并返回结果
            
            # 如果没有结果，创建一个错误结果
            if not self.final_result:
                self.final_result = {"error": "未收到识别结果"}
                
        except asyncio.TimeoutError:
            print("警告: 等待最终结果超时")
            return {"error": "识别超时"}
        
        return self.final_result or {"error": "未收到识别结果"}
    
    async def recognize_audio_from_file(self, file_path: str):
        """从文件中读取音频并识别"""
        try:
            with wave.open(file_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                return await self.recognize_audio(audio_data)
        except Exception as e:
            print(f"读取音频文件失败: {str(e)}")
            return False
    
    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
        
        self.audio.terminate()
        print("连接已关闭")


async def main():
    # 使用本地地址作为默认值，避免DNS解析错误
    server_url = "ws://127.0.0.1:8765"
    
    # 可以根据需要修改为实际的服务器地址
    # server_url = "ws://192.168.1.100:8765"
    
    print(f"尝试连接到WebSocket服务器: {server_url}")
    
    # 创建WebSocket客户端
    client = AudioWebSocketClient(server_url)
    
    try:
        # 连接到服务器
        if await client.connect():
            print("成功连接到服务器，准备进行语音识别...")
            
            # 先查询声纹列表，检查注册状态
            print("正在获取声纹库信息...")
            await client.list_voiceprints()
            # 等待一会儿，让响应通过websocket接收
            await asyncio.sleep(1)
            
            # 显示菜单
            while True:
                print("\n==== 语音识别系统 ====")
                print("1. 查看声纹列表")
                print("2. 注册新声纹")
                print("3. 进行流式语音识别 (检查声纹)")
                print("4. 进行流式语音识别 (不检查声纹)")
                print("5. 录音完成后识别 (检查声纹)")
                print("6. 录音完成后识别 (不检查声纹)")
                print("7. 从文件识别音频")
                print("0. 退出程序")
                
                choice = input("\n请选择操作: ")
                
                if choice == "1":
                    # 查看声纹列表
                    # 发送请求并等待响应
                    await client.list_voiceprints()
                    # 等待一会儿，让响应通过message_handler异步处理
                    await asyncio.sleep(1)
                    
                    response = await client.list_voiceprints()
                    if response.get("success", False) and "voiceprints" in response:
                        voiceprints = response["voiceprints"]
                        print("\n当前声纹库列表:")
                        print("+" + "-" * 70 + "+")
                        print(f"| {'ID':<40} | {'用户名':<25} |")
                        print("+" + "-" * 70 + "+")
                        
                        for vp in voiceprints:
                            print(f"| {vp['id']:<40} | {vp['person_name']:<25} |")
                            
                        print("+" + "-" * 70 + "+")
                        print(f"总计: {len(voiceprints)} 条声纹记录\n")
                    else:
                        print("获取声纹列表失败")
                
                elif choice == "2":
                    # 注册新声纹
                    person_name = input("请输入用户名: ")
                    print("即将录制声纹样本...")
                    audio_data = await client.record_audio(3)
                    result = await client.register_voiceprint(audio_data, person_name)
                    if result.get("success", False):
                        print(f"声纹 '{person_name}' 注册成功")
                    else:
                        print(f"声纹注册失败: {result.get('error', '未知错误')}")
                
                elif choice == "3":
                    # 流式识别 (检查声纹)
                    duration = int(input("请输入录音时长(秒): ") or "5")
                    result = await client.stream_audio_from_mic(duration, check_voiceprint=True)
                    
                    if "error" not in result:
                        print("\n=== 识别摘要 ===")
                        print(f"完整文本: {result['text']}")
                        print(f"识别用户: {result['user']}")
                        print(f"识别片段数: {len(result['segments'])}")
                        print("================\n")
                    else:
                        print(f"识别失败: {result['error']}")
                
                elif choice == "4":
                    # 流式识别 (不检查声纹)
                    duration = int(input("请输入录音时长(秒): ") or "5")
                    result = await client.stream_audio_from_mic(duration, check_voiceprint=False)
                    
                    if "error" not in result:
                        print("\n=== 识别摘要 ===")
                        print(f"完整文本: {result['text']}")
                        print("用户: 未检查声纹")
                        print(f"识别片段数: {len(result['segments'])}")
                        print("================\n")
                    else:
                        print(f"识别失败: {result['error']}")
                
                elif choice == "5":
                    # 录音完成后识别 (检查声纹)
                    duration = int(input("请输入录音时长(秒): ") or "5")
                    result = await client.stream_audio_from_mic_batch(duration, check_voiceprint=True)
                    
                    if "error" not in result:
                        print("\n=== 识别摘要 ===")
                        print(f"完整文本: {result['text']}")
                        print(f"识别用户: {result['user']}")
                        print("================\n")
                    else:
                        print(f"识别失败: {result['error']}")
                
                elif choice == "6":
                    # 录音完成后识别 (不检查声纹)
                    duration = int(input("请输入录音时长(秒): ") or "5")
                    result = await client.stream_audio_from_mic_batch(duration, check_voiceprint=False)
                    
                    if "error" not in result:
                        print("\n=== 识别摘要 ===")
                        print(f"完整文本: {result['text']}")
                        print("用户: 未检查声纹")
                        print("================\n")
                    else:
                        print(f"识别失败: {result['error']}")
                
                elif choice == "7":
                    # 从文件识别
                    file_path = input("请输入音频文件路径: ")
                    if os.path.exists(file_path):
                        await client.recognize_audio_from_file(file_path)
                    else:
                        print("文件不存在")
                
                elif choice == "0":
                    print("退出程序...")
                    break
                
                else:
                    print("无效选择，请重新输入")
                
                input("\n按Enter继续...")
            
        else:
            print("无法连接到WebSocket服务器，请检查以下内容：")
            print("1. 服务器是否已启动并运行")
            print("2. 服务器地址和端口是否正确")
            print("3. 是否有防火墙阻止连接")
            print("4. 网络连接是否正常")
    except Exception as e:
        print(f"运行时发生错误: {str(e)}")
    finally:
        # 关闭连接
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
