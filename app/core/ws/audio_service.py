"""
audio_service.py
处理音频录制和识别的服务模块
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Callable

from app import logger
from app.core.ws.audio_websocket_client import AudioWebSocketClient

class AudioService:
    """音频服务类，处理录音和识别逻辑"""
    
    @staticmethod
    async def send_progress_updates(websocket_sender, duration: int, is_recording_active_func):
        """发送进度更新消息"""
        start_time = asyncio.get_event_loop().time()
        last_update = 0
        
        while True:
            # 检查是否还在录音
            if not is_recording_active_func():
                break
                
            # 计算当前经过的时间
            current_time = asyncio.get_event_loop().time() - start_time
            current_second = int(current_time)
            
            # 只在秒数变化时更新
            if current_second > last_update and current_second <= duration:
                last_update = current_second
                percentage = round(current_second / duration * 100)
                
                # 发送进度更新
                await websocket_sender({
                    "type": "recording",
                    "status": "progress",
                    "current": current_second,
                    "total": duration,
                    "percentage": percentage
                })
                
            # 如果达到或超过总时长，退出循环
            if current_second >= duration:
                break
                
            # 短暂等待，以减少CPU使用
            await asyncio.sleep(0.1)
    
    @staticmethod
    async def perform_recording(
        session_data: Dict[str, Any], 
        websocket_sender,
        duration: int, 
        check_voiceprint: bool
    ):
        """执行录音和识别任务"""
        asr_client = session_data["asr_client"]
        recording_complete_event = asyncio.Event()
        recognition_complete_event = asyncio.Event()
        valid_result_event = asyncio.Event()  # 新增：有效结果事件
        
        try:
            # 先报告正在录音
            await websocket_sender({
                "type": "recording",
                "status": "info",
                "message": "开始录音..."
            })
            
            # 启动进度更新任务
            progress_task = asyncio.create_task(
                AudioService.send_progress_updates(
                    websocket_sender, 
                    duration,
                    lambda: session_data["recording_active"]
                )
            )
            
            # 录制音频（以异步方式）
            audio_data_future = asyncio.create_task(asr_client.record_audio(duration))
            
            # 监听音频录制完成
            def on_audio_complete(task):
                if not task.cancelled():
                    recording_complete_event.set()
                
            audio_data_future.add_done_callback(on_audio_complete)
            
            # 等待录音完成或被取消
            try:
                await recording_complete_event.wait()
                audio_data = await audio_data_future
                
                # 确保进度更新任务完成
                if not progress_task.done():
                    progress_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(progress_task), timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                
                # 报告录音完成
                await websocket_sender({
                    "type": "recording",
                    "status": "info",
                    "message": "录音完成，处理中..."
                })
                
                # 如果需要验证声纹
                if check_voiceprint:
                    await websocket_sender({
                        "type": "recording",
                        "status": "info",
                        "message": "验证声纹中..."
                    })
                    
                    # 发送声纹验证请求（异步）
                    voiceprint_task = asyncio.create_task(asr_client.match_voiceprint(audio_data))
                    
                    # 等待声纹验证完成
                    await asyncio.wait([voiceprint_task], timeout=3.0)
                
                # 开始识别音频
                await websocket_sender({
                    "type": "recording",
                    "status": "info",
                    "message": "识别录音内容中..."
                })
                
                logger.info("开始识别录制的音频")
                
                # 清除旧的结果
                asr_client.final_result = None
                asr_client.streaming_results = []
                asr_client.last_error = None
                
                # 重要修改：设置超时时间和重试次数
                max_attempts = 3
                attempt = 0
                result = None
                
                # 添加一个异步处理函数来重试识别
                async def process_recognition_with_retry():
                    nonlocal result, attempt
                    
                    while attempt < max_attempts:
                        attempt += 1
                        logger.info(f"开始第 {attempt} 次识别尝试")
                        
                        # 创建一个future来存储识别结果
                        recognition_future = asyncio.create_task(
                            asr_client.recognize_audio(audio_data, check_voiceprint)
                        )
                        
                        # 监听识别完成
                        def on_recognition_complete(task):
                            if not task.cancelled():
                                recognition_complete_event.set()
                        
                        recognition_future.add_done_callback(on_recognition_complete)
                        
                        # 等待识别完成（有超时限制）
                        try:
                            await asyncio.wait_for(recognition_complete_event.wait(), timeout=5.0)
                            recognition_complete_event.clear()  # 重置事件以便重试
                            result = await recognition_future
                            
                            # 检查识别结果是否有效，如果有效则立即返回
                            if (result and 
                                "text" in result and 
                                isinstance(result["text"], str) and  # 确保是字符串类型
                                result["text"].strip() and  # 确保不是空字符串
                                result["text"] not in ("等待识别结果...", "识别结果将在服务器处理完成后显示")):
                                # 处理文本，去除可能的特殊字符
                                result["text"] = ''.join(char for char in result["text"] if ord(char) < 65536)
                                result["text"] = result["text"].strip()
                                
                                if result["text"]:  # 确保处理后的文本不为空
                                    logger.info(f"获取到有效识别结果: {result.get('text', '')[:30]}...")
                                    valid_result_event.set()
                                    return True  # 有效结果，立即返回
                            
                            logger.warning(f"第 {attempt} 次识别未得到有效结果，准备重试...")
                            await asyncio.sleep(0.5)
                        except asyncio.TimeoutError:
                            logger.warning(f"第 {attempt} 次识别超时")
                            await asyncio.sleep(0.5)
                            continue  # 超时后直接继续下一次尝试
                    
                    # 所有尝试都失败了才尝试流式处理
                    logger.info("主识别未返回有效结果，尝试流式处理方案...")
                    
                    streaming_task = asyncio.create_task(
                        AudioService.process_streaming_recognition(asr_client, audio_data, check_voiceprint)
                    )
                    
                    try:
                        await asyncio.wait_for(streaming_task, timeout=5.0)
                        
                        if asr_client.final_result and "text" in asr_client.final_result and asr_client.final_result["text"] not in ("等待识别结果...", "识别结果将在服务器处理完成后显示"):
                            result = asr_client.final_result
                            logger.info(f"流式处理成功: {result.get('text', '')[:30]}...")
                            valid_result_event.set()
                            return True
                        else:
                            # 如果仍然没有有效结果，检查是否有部分结果可用
                            if asr_client.streaming_results:
                                combined_text = " ".join([r.get("text", "") for r in asr_client.streaming_results])
                                if combined_text.strip():
                                    result = {
                                        "text": combined_text,
                                        "user": "Unknown",
                                        "segments": asr_client.streaming_results
                                    }
                                    logger.info(f"使用合并的流式结果: {result.get('text', '')[:30]}...")
                                    valid_result_event.set()
                                    return True
                    except asyncio.TimeoutError:
                        logger.warning("流式识别处理超时")
                    
                    # 所有尝试都失败
                    return False
                
                # 执行识别尝试
                recognition_success = await process_recognition_with_retry()
                
                # 处理最终结果
                if recognition_success and result:
                    # 保存最终结果
                    session_data["final_result"] = result
                    
                    # 确保segments字段存在
                    if "segments" not in result or not isinstance(result["segments"], list):
                        result["segments"] = [{"text": result["text"], "user": result.get("user", "Unknown")}]
                    
                    # 发送完成消息
                    await websocket_sender({
                        "type": "recording",
                        "status": "completed",
                        "result": result
                    })
                    logger.info(f"成功完成识别并返回结果: {result['text'][:30]}...")
                else:
                    # 创建默认错误消息
                    error_result = {
                        "text": "无法识别语音内容，请重试",
                        "user": "Unknown",
                        "error": "未能获取识别结果",
                        "segments": []
                    }
                    
                    # 使用声纹错误信息（如果有）
                    if asr_client.last_error:
                        error_result = {
                            "text": "语音已识别，但出现声纹验证错误",
                            "user": asr_client.last_error.get("user", "Unknown"),
                            "error": asr_client.last_error.get("error"),
                            "similarity": asr_client.last_error.get("similarity", 0),
                            "segments": []
                        }
                    
                    # 发送错误结果
                    await websocket_sender({
                        "type": "recording",
                        "status": "completed",
                        "result": error_result,
                        "warning": True
                    })
                    logger.warning(f"返回错误结果: {error_result['text']}")
                
            except asyncio.CancelledError:
                # 录音任务被取消
                if not progress_task.done():
                    progress_task.cancel()
                if not audio_data_future.done():
                    audio_data_future.cancel()
                logger.warning("录音任务被取消")
        
        except Exception as e:
            logger.error(f"录音过程异常: {e}")
            await websocket_sender({
                "type": "error",
                "message": f"录音过程异常: {str(e)}"
            })
        
        finally:
            # 清理资源
            session_data["recording_active"] = False
            if asr_client:
                await asr_client.close()
                session_data["asr_client"] = None
            # logger.info("识别会话已完成并清理")
    
    @staticmethod
    async def process_streaming_recognition(asr_client, audio_data, check_voiceprint):
        """处理流式音频识别"""
        # 启动流式处理
        await asr_client.start_streaming(check_voiceprint)
        
        # 分段发送音频数据
        chunk_size = 8192  # 8KB
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            is_end = i + chunk_size >= len(audio_data)
            await asr_client.send_audio_data(chunk, is_end)
            # 短暂暂停，以避免过载服务器
            await asyncio.sleep(0.01)
        
        # 等待最终结果（设置合理的超时）
        completion_event = asyncio.Event()
        
        # 检查结果是否已经可用
        def check_result():
            if asr_client.final_result:
                completion_event.set()
        
        # 创建一个定时检查机制
        check_interval = 0.5  # 每0.5秒检查一次
        check_count = 10      # 最多检查10次，即5秒
        
        for _ in range(check_count):
            check_result()
            if completion_event.is_set():
                break
            await asyncio.sleep(check_interval)
    
    @staticmethod
    async def perform_batch_recording(
        session_data: Dict[str, Any],
        websocket_sender,
        duration: int,
        check_voiceprint: bool
    ):
        """执行批量录音和识别任务"""
        asr_client = session_data["asr_client"]
        recording_complete_event = asyncio.Event()
        recognition_complete_event = asyncio.Event()
        
        try:
            # 通知客户端开始录音
            await websocket_sender({
                "type": "recording",
                "status": "info",
                "message": "录音中，完成后将进行整体识别..."
            })
            
            # 启动进度更新任务
            progress_task = asyncio.create_task(
                AudioService.send_progress_updates(
                    websocket_sender, 
                    duration,
                    lambda: session_data["recording_active"]
                )
            )
            
            # 录制整段音频（异步）
            audio_task = asyncio.create_task(asr_client.record_audio(duration))
            
            # 设置录音完成回调
            def on_recording_complete(task):
                if not task.cancelled():
                    recording_complete_event.set()
            
            audio_task.add_done_callback(on_recording_complete)
            
            # 等待录音完成
            try:
                await asyncio.wait_for(recording_complete_event.wait(), timeout=duration+2.0)
                audio_data = await audio_task
                logger.info("录音完成，开始一次性识别")
                
                # 取消并等待进度更新任务完成
                if not progress_task.done():
                    progress_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(progress_task), timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                
                # 通知客户端录音完成，开始识别
                await websocket_sender({
                    "type": "recording",
                    "status": "info",
                    "message": "录音完成，开始识别..."
                })
                
                # 如果需要检查声纹
                if check_voiceprint:
                    await websocket_sender({
                        "type": "recording",
                        "status": "info",
                        "message": "进行声纹验证..."
                    })
                    
                    # 发送声纹匹配请求（异步）
                    voiceprint_task = asyncio.create_task(asr_client.match_voiceprint(audio_data))
                    
                    # 等待声纹验证完成，但设置超时
                    await asyncio.wait([voiceprint_task], timeout=3.0)
                    
                    await websocket_sender({
                        "type": "recording",
                        "status": "info",
                        "message": "声纹验证完成，开始识别语音内容..."
                    })
                
                # 一次性识别音频
                logger.info("开始识别录音内容")
                
                # 创建一个future来存储识别结果
                recognition_future = asyncio.create_task(
                    asr_client.recognize_audio(audio_data, check_voiceprint)
                )
                
                # 监听识别完成
                def on_recognition_complete(task):
                    if not task.cancelled():
                        recognition_complete_event.set()
                
                recognition_future.add_done_callback(on_recognition_complete)
                
                # 等待识别完成（有超时限制）
                try:
                    await asyncio.wait_for(recognition_complete_event.wait(), timeout=5.0)
                    result = await recognition_future
                    
                    # 保存最终结果
                    session_data["final_result"] = result
                    
                    # 确保结果中有实际文本内容
                    if result.get("text") == "等待识别结果...":
                        logger.warning("结果中没有实际文本，尝试替换为默认文本")
                        result["text"] = "未能获取有效的识别文本，但语音已处理"
                    
                    # 为了保持一致性，制作一个segments格式，但只包含一个段落
                    if "segments" not in result:
                        result["segments"] = [{"text": result["text"], "user": result.get("user", "unknown")}]
                    
                    # 发送完成消息
                    await websocket_sender({
                        "type": "recording",
                        "status": "completed",
                        "result": result
                    })
                    
                except asyncio.TimeoutError:
                    error = "识别超时"
                    await websocket_sender({
                        "type": "error",
                        "message": f"语音识别失败: {error}"
                    })
            
            except asyncio.TimeoutError:
                logger.warning("录音过程超时")
                await websocket_sender({
                    "type": "error",
                    "message": "录音过程超时"
                })
            
            except asyncio.CancelledError:
                logger.warning("录音任务被取消")
                if not progress_task.done():
                    progress_task.cancel()
        
        except Exception as e:
            logger.error(f"批量录音识别异常: {e}")
            await websocket_sender({
                "type": "error",
                "message": f"批量录音识别异常: {str(e)}"
            })
        
        finally:
            # 清理资源
            session_data["recording_active"] = False
            if asr_client:
                await asr_client.close()
                session_data["asr_client"] = None
    
    @staticmethod
    async def get_voiceprints():
        """获取声纹库列表"""
        asr_client = None
        try:
            # 创建临时客户端获取声纹列表
            asr_client = AudioWebSocketClient("ws://127.0.0.1:8765")
            connection_event = asyncio.Event()
            
            # 异步连接并等待结果
            connection_task = asyncio.create_task(asr_client.connect())
            
            def on_connection_complete(task):
                connection_event.set()
                
            connection_task.add_done_callback(on_connection_complete)
            
            # 等待连接完成，设置超时
            try:
                await asyncio.wait_for(connection_event.wait(), timeout=5.0)
                connection_result = await connection_task
                
                if connection_result:
                    # 获取声纹列表并等待
                    voiceprints_task = asyncio.create_task(asr_client.list_voiceprints())
                    
                    try:
                        voiceprints_response = await asyncio.wait_for(voiceprints_task, timeout=5.0)
                        return {"success": True, "voiceprints": voiceprints_response.get("voiceprints", [])}
                    except asyncio.TimeoutError:
                        return {"success": False, "message": "获取声纹列表超时"}
                else:
                    return {"success": False, "message": "无法连接到语音服务器"}
            except asyncio.TimeoutError:
                return {"success": False, "message": "连接到语音服务器超时"}
                
        except Exception as e:
            logger.error(f"获取声纹列表异常: {e}")
            return {"success": False, "message": f"获取声纹列表异常: {str(e)}"}
        finally:
            # 确保关闭客户端连接
            if asr_client:
                await asr_client.close()
