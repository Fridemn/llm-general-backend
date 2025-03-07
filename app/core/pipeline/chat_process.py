import os
import json
import logging
import traceback

from typing import Optional, Dict, Any, Union
from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app import app_config, logger
from app.core.pipeline.text_process import text_process
from app.core.pipeline.voice_process import voice_process
from app.core.db.db_history import db_message_history
from app.core.llm.message import LLMMessage, LLMResponse, MessageRole, MessageComponent, MessageSender, MessageType

DEFAULT_MODEL = app_config.llm_config['default_model']

class ChatProcess:
    """
    聊天处理流水线，整合文本和语音处理流程
    """
    
    def __init__(self):
        pass
    
    async def handle_request(self, 
                           model: Optional[str], 
                           message: Optional[str],
                           history_id: Optional[str],
                           role: MessageRole,
                           stream: bool,
                           stt: bool,
                           tts: bool,
                           audio_file: Optional[UploadFile],
                           user_id: Optional[str]) -> Union[StreamingResponse, LLMResponse, Dict[str, Any]]:
        """
        处理统一聊天请求
        
        Args:
            model: LLM模型名称
            message: 文本消息
            history_id: 对话历史ID
            role: 消息角色
            stream: 是否流式响应
            stt: 是否需要语音转文本
            tts: 是否需要文本转语音
            audio_file: 上传的音频文件
            user_id: 用户ID
            
        Returns:
            StreamingResponse或LLMResponse
        """
        # 初始化变量
        input_message = None
        transcribed_text = None
        
        try:
            # 确保参数合法
            model = model or DEFAULT_MODEL
            history_id = history_id if history_id and history_id.strip() else None
            
            # 准备输入消息
            input_message = await self._prepare_input_message(
                message, 
                history_id, 
                role, 
                stt, 
                audio_file,
            )
            
            # 如果是语音输入，获取语音转写文本
            if stt and audio_file and hasattr(input_message, 'message_str'):
                transcribed_text = input_message.message_str
            
            # 根据流式处理需求选择处理方式
            if stream:
                return await self._handle_stream_response(
                    model, input_message, history_id, user_id,
                    stt, tts, transcribed_text
                )
            else:
                return await self._handle_normal_response(
                    model, input_message, history_id, user_id, tts
                )
                
        except HTTPException:
            raise
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"聊天处理流水线异常: {str(e)}\n{error_trace}")
            raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")
    
    async def _prepare_input_message(self, 
                                  message: Optional[str],
                                  history_id: Optional[str],
                                  role: MessageRole,
                                  stt: bool,
                                  audio_file: Optional[UploadFile]) -> LLMMessage:
        """
        根据请求参数准备输入消息
        
        Returns:
            LLMMessage消息对象
        """
        temp_dir = None
        audio_info = None
        
        try:
            # 处理语音输入
            if stt and audio_file:
                # 使用voice_process保存和处理音频文件
                audio_info = await voice_process.save_audio_file(audio_file)
                temp_dir = audio_info["temp_dir"]
                audio_file_path = audio_info["temp_path"]
                permanent_audio_path = audio_info["permanent_path"]
                permanent_audio_url = audio_info["permanent_url"]
                
                # 使用voice_process进行语音转文本
                transcribed_text = await voice_process.process_stt(
                    audio_file_path
                )
                
                # 音频处理成功后，保存到永久存储
                voice_process.save_to_permanent_storage(audio_file_path, permanent_audio_path)
                
                # 构造输入消息 - 使用URL路径而非本地路径
                audio_component = voice_process.create_audio_component(
                    permanent_audio_url,
                    extra_info={
                        "transcript": transcribed_text,
                        "original_filename": audio_file.filename,
                        "file_path": permanent_audio_path  # 保存原始文件路径用于文件存在性检查
                    }
                )
                
                text_component = MessageComponent(
                    type=MessageType.TEXT,
                    content=transcribed_text
                )
                
                return LLMMessage(
                    history_id=history_id or "",
                    sender=MessageSender(role=role),
                    components=[audio_component, text_component],
                    message_str=transcribed_text
                )
            else:
                # 处理纯文本输入
                if not message or not message.strip():
                    raise HTTPException(status_code=400, detail="消息内容不能为空")
                
                # 构造输入消息
                return LLMMessage.from_text(
                    text=message,
                    history_id=history_id or "",
                    role=role
                )
                
        except Exception as e:
            # 清理临时文件
            if temp_dir:
                voice_process.cleanup_temp_files(temp_dir)
            logger.error(f"准备输入消息失败: {str(e)}")
            raise
    
    async def _handle_stream_response(self, 
                                    model: str,
                                    input_message: LLMMessage,
                                    history_id: Optional[str],
                                    user_id: Optional[str],
                                    stt: bool,
                                    tts: bool,
                                    transcribed_text: Optional[str]) -> StreamingResponse:
        """
        处理流式响应
        
        Returns:
            StreamingResponse对象
        """
        async def generate():
            temp_dir = None
            try:
                count = 0
                full_response_text = ""  # 收集完整响应用于TTS
                token_info = None        # 保存token信息
                message_id = None        # 保存原始消息ID
                current_history_id = history_id or input_message.history_id  # 使用有效的历史ID
                
                # 如果是语音输入，先返回识别结果
                if (stt and transcribed_text):
                    yield f"data: {json.dumps({'transcription': transcribed_text})}\n\n"
                
                # 处理消息流
                async for chunk in text_process.process_message_stream(
                    model,
                    input_message,
                    current_history_id,  # 使用确定的历史ID
                    user_id
                ):
                    count += 1
                    
                    # 检查是否是token信息特殊标记
                    if chunk.startswith("__TOKEN_INFO__"):
                        try:
                            token_data = json.loads(chunk[14:])  # 去掉特殊前缀
                            token_info = token_data
                            message_id = token_data.get("message_id")
                            # 如果token信息中包含了history_id，使用它更新当前历史ID
                            if "history_id" in token_data and token_data["history_id"]:
                                current_history_id = token_data["history_id"]
                            token_response = f"data: {json.dumps({'token_info': token_data})}\n\n"
                            yield token_response
                        except Exception as e:
                            logger.error(f"处理token信息失败: {str(e)}")
                    else:
                        # 收集完整响应文本用于TTS
                        full_response_text += chunk
                        # 将普通文本块包装为SSE格式
                        response_text = f"data: {json.dumps({'text': chunk})}\n\n"
                        yield response_text
                
                # 如果需要TTS且有响应文本且有message_id
                if tts and full_response_text.strip() and message_id and current_history_id:
                    logger.debug(f"尝试处理TTS，历史ID: {current_history_id}, 消息ID: {message_id}")
                    audio_url = await voice_process.process_tts(full_response_text)
                    if audio_url:
                        # 创建音频组件
                        audio_component = voice_process.create_audio_component(
                            audio_url, 
                            full_response_text,
                            {"tts_model": "default"}
                        )
                        
                        # 创建并保存新的音频消息
                        audio_message = LLMMessage(
                            history_id=current_history_id,
                            sender=MessageSender(role=MessageRole.ASSISTANT, nickname=model),
                            components=[audio_component],
                            message_str=full_response_text,
                            output_tokens=token_info.get("output_tokens") if token_info else None,
                            source_model=model
                        )
                        
                        # 保存音频消息到数据库
                        logger.debug(f"流式响应：保存音频消息，历史ID: {current_history_id}")
                        success = await db_message_history.add_message(current_history_id, audio_message)
                        
                        if success:
                            # 删除原始文本消息
                            await db_message_history.delete_message(current_history_id, message_id)
                            logger.info(f"流式响应：已成功将AI回复转换为音频消息")
                            # 发送音频URL
                            yield f"data: {json.dumps({'audio': audio_url, 'new_message_id': audio_message.message_id})}\n\n"
                        else:
                            logger.error(f"流式响应：保存音频消息失败，历史ID={current_history_id}")
                            yield f"data: {json.dumps({'audio': audio_url})}\n\n"
                    else:
                        yield f"data: {json.dumps({'tts_error': '无法生成语音'})}\n\n"
                
                # 如果没有生成任何内容
                if count == 0:
                    yield f"data: {json.dumps({'text': '未能生成响应'})}\n\n"
                    
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"流式处理失败: {str(e)}\n{error_trace}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                # 标记流结束
                yield "data: [DONE]\n\n"
                # 清理临时文件
                if temp_dir and os.path.exists(temp_dir):
                    voice_process.cleanup_temp_files(temp_dir)
        
        # 确保设置正确的 SSE 响应头
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no", 
                "Content-Type": "text/event-stream"
            }
        )
    
    async def _handle_normal_response(self, 
                                   model: str,
                                   input_message: LLMMessage,
                                   history_id: Optional[str],
                                   user_id: Optional[str],
                                   tts: bool) -> Union[LLMResponse, Dict[str, Any]]:
        """
        处理普通响应
        
        Returns:
            LLMResponse或字典对象
        """
        try:
            # 处理消息并获取响应
            response = await text_process.process_message(
                model,
                input_message,
                history_id,
                user_id
            )
            
            # 如果需要TTS处理
            if tts and response.response_message.message_str.strip():
                # 优先使用响应消息中的历史ID，如果没有，再使用输入参数中的历史ID
                current_history_id = response.response_message.history_id or history_id
                if not current_history_id:
                    logger.warning("缺少有效的历史记录ID，无法保存音频消息")
                    # 添加音频URL到结果字典中，但不进行数据库操作
                    audio_url = await voice_process.process_tts(response.response_message.message_str)
                    if audio_url:
                        result_dict = response.dict()
                        result_dict["audio_url"] = audio_url
                        return result_dict
                    return response

                original_message_id = response.response_message.message_id
                audio_url = await voice_process.process_tts(response.response_message.message_str)
                
                if audio_url:
                    # 创建音频组件并添加到响应
                    audio_component = voice_process.create_audio_component(
                        audio_url, 
                        response.response_message.message_str,
                        {"tts_model": "default"}
                    )
                    
                    # 创建并保存新的音频消息
                    audio_message = LLMMessage(
                        history_id=current_history_id,  # 使用确认有效的历史ID
                        sender=MessageSender(role=MessageRole.ASSISTANT, nickname=model),
                        components=[audio_component],
                        message_str=response.response_message.message_str,
                        output_tokens=response.response_message.output_tokens,
                        source_model=model
                    )
                    
                    # 保存音频消息到数据库
                    logger.debug(f"保存音频消息，历史ID: {current_history_id}")
                    success = await db_message_history.add_message(current_history_id, audio_message)
                    
                    if success:
                        # 删除原始文本消息
                        await db_message_history.delete_message(current_history_id, original_message_id)
                        logger.info(f"已成功将AI回复转换为音频消息: {audio_message.message_id}")
                        
                        # 更新响应为新的音频消息
                        response.response_message = audio_message
                    else:
                        logger.error(f"保存音频消息到数据库失败: history_id={current_history_id}")
                    
                    # 添加音频URL到结果字典中
                    result_dict = response.dict()
                    result_dict["audio_url"] = audio_url
                    
                    return result_dict
            
            return response
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"处理消息失败: {str(e)}\n{error_trace}")
            raise

# 全局聊天处理流水线实例
chat_process = ChatProcess()
