from typing import List, Dict, Optional, AsyncGenerator, Any
import traceback
import uuid
from fastapi import HTTPException
import json

from app import logger
from app.core.llm import BaseLLM, ChatMessage, LLMResponse as RawLLMResponse, OpenAILLM, AnthropicLLM, LLMConfig, OllamaLLM
from app.core.llm.config import llm_config, MODEL_TO_ENDPOINT, DEFAULT_MODEL
from app.core.llm.db_history import db_message_history
from app.core.llm.message import LLMMessage, LLMResponse, MessageSender, MessageRole, MessageComponent, MessageType
from app.utils.token_counter import TokenCounter


class ChatPipeline:
    """
    聊天流水线处理器，用于处理用户消息并返回AI回复
    """

    def __init__(self):
        self.llm_instances: Dict[str, BaseLLM] = {}
        self._initialize_llms()
    
    def _initialize_llms(self):
        for endpoint, config in llm_config.items():
            for model in config["available_models"]:
                llm_config_obj = LLMConfig(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    model_name=model
                )
                
                # 根据endpoint类型选择对应的LLM实现
                if endpoint == "anthropic":
                    self.llm_instances[model] = AnthropicLLM(llm_config_obj)
                elif endpoint == "ollama":
                    self.llm_instances[model] = OllamaLLM(llm_config_obj)
                else:
                    self.llm_instances[model] = OpenAILLM(llm_config_obj)
    
    def _get_endpoint_for_model(self, model: str) -> str:
        model = model or DEFAULT_MODEL
        if model not in MODEL_TO_ENDPOINT:
            raise ValueError(f"未知的模型: {model}")
        return MODEL_TO_ENDPOINT[model]
    
    async def process_chat(self, model: str, message: str) -> RawLLMResponse:
        model = model or DEFAULT_MODEL
        if model not in self.llm_instances:
            raise ValueError(f"未知的模型: {model}")
            
        messages = [ChatMessage(role="user", content=message)]
        llm = self.llm_instances[model]
        return await llm.chat_completion(messages)

    def _extract_text_from_message(self, llm_message: LLMMessage) -> str:
        """从LLMMessage中提取纯文本内容，用于发送给LLM"""
        # 首先尝试使用message_str
        if llm_message.message_str and llm_message.message_str.strip():
            return llm_message.message_str
        
        # 否则从组件中提取文本
        text_parts = []
        for component in llm_message.components:
            if component.type == MessageType.TEXT:
                text_parts.append(component.content)
            elif component.type == MessageType.AUDIO and component.extra and "transcript" in component.extra:
                # 如果是音频组件且有转写文本
                text_parts.append(component.extra["transcript"])
        
        return " ".join(text_parts) if text_parts else ""

    async def process_message(self, model: str, message: LLMMessage, history_id: Optional[str] = None, user_id: Optional[str] = None) -> LLMResponse:
        model = model or DEFAULT_MODEL
        if model not in self.llm_instances:
            raise ValueError(f"未知的模型: {model}")
        
        try:
            # 处理历史ID
            current_history_id = history_id or message.history_id
            if not current_history_id:
                try:
                    # 创建新的历史记录并关联用户ID
                    current_history_id = await db_message_history.create_history(user_id)
                    message.history_id = current_history_id
                except Exception as e:
                    error_trace = traceback.format_exc()
                    # 创建临时ID继续聊天
                    current_history_id = str(uuid.uuid4())
                    message.history_id = current_history_id
            
            # 为用户消息设置目标模型
            message.target_model = model
            
            # 从消息中提取文本内容用于LLM处理
            message_text = self._extract_text_from_message(message)
            
            # 预先估算用户消息的token数量
            try:
                estimated_tokens = TokenCounter.count_tokens(message_text, model)
                message.input_tokens = estimated_tokens
            except Exception as e:
                logger.error(f"计算用户消息token失败: {e}")
            
            # 尝试保存用户消息到历史记录
            try:
                await db_message_history.add_message(current_history_id, message)
            except Exception as e:
                logger.error(f"保存用户消息到历史记录失败，但继续处理: {e}")
            
            # 获取历史消息并转换为LLM消息格式
            chat_messages = []
            history = []
            try:
                history = await db_message_history.get_history(current_history_id)
                
                # 消息简化部分
                # 只取最近的10条消息，避免tokens过多
                for hist_msg in history[-10:]:
                    role = "user"
                    if hist_msg.sender.role == MessageRole.ASSISTANT:
                        role = "assistant"
                    elif hist_msg.sender.role == MessageRole.SYSTEM:
                        role = "system"
                    
                    # 从消息中提取文本内容
                    msg_text = self._extract_text_from_message(hist_msg)
                        
                    chat_messages.append(ChatMessage(
                        role=role,
                        content=msg_text
                    ))

            except Exception as e:
                logger.error(f"获取历史记录失败，只使用当前消息: {e}")
            
            # 如果没有历史消息，则只添加当前消息
            if not chat_messages:
                chat_messages = [ChatMessage(role="user", content=message_text)]
            
            # 在调用API之前估算所有消息的token
            try:
                openai_messages = []
                for msg in chat_messages:
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                estimated_input_tokens = TokenCounter.estimate_openai_tokens(openai_messages, model)
            except Exception as e:
                logger.error(f"预估输入tokens失败: {e}")
                estimated_input_tokens = None
            
            # 调用LLM进行回复
            llm = self.llm_instances[model]
            raw_response = await llm.chat_completion(chat_messages)
            
            # 获取实际的token使用情况
            actual_input_tokens = raw_response.input_tokens
            actual_output_tokens = raw_response.output_tokens
            
            # 如果API没有返回token计数，则使用我们的预估值
            if actual_input_tokens is None:
                actual_input_tokens = estimated_input_tokens
            
            # 构造响应消息，包含token信息
            response_message = LLMMessage(
                history_id=current_history_id,
                sender=MessageSender(role=MessageRole.ASSISTANT, nickname=model),
                components=[MessageComponent(
                    type=MessageType.TEXT,
                    content=raw_response.text
                )],
                message_str=raw_response.text,
                output_tokens=actual_output_tokens,
                source_model=model
            )
            
            # 更新用户消息的输入tokens
            if actual_input_tokens is not None:
                try:
                    # 直接更新消息对象
                    message.input_tokens = actual_input_tokens
                    
                    # 更新历史记录中的消息token数据
                    await db_message_history.update_message(
                        current_history_id,
                        message.message_id,
                        {"input_tokens": actual_input_tokens}
                    )
                except Exception as e:
                    logger.error(f"更新用户消息token计数失败: {e}")
            
            # 尝试保存AI回复到历史记录
            try:
                await db_message_history.add_message(current_history_id, response_message)
            except Exception as e:
                logger.error(f"保存AI回复到历史记录失败: {e}")
            
            return LLMResponse(
                response_message=response_message,
                raw_response=raw_response.raw_response,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens
            )
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"处理消息时发生错误: {e}\n{error_trace}")
            raise

    async def process_message_stream(self, model: str, message: LLMMessage, history_id: Optional[str] = None, user_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """流式处理消息并返回生成器"""
        model = model or DEFAULT_MODEL
        if model not in self.llm_instances:
            raise ValueError(f"未知的模型: {model}")
        
        try:
            # 处理历史ID
            current_history_id = history_id or message.history_id
            if not current_history_id:
                try:
                    # 创建新的历史记录并关联用户ID
                    current_history_id = await db_message_history.create_history(user_id)
                    message.history_id = current_history_id
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"创建历史记录失败: {e}\n{error_trace}")
                    # 创建临时ID继续聊天
                    current_history_id = str(uuid.uuid4())
                    message.history_id = current_history_id
            
            # 为用户消息设置目标模型
            message.target_model = model
            
            # 从消息中提取文本内容用于LLM处理
            message_text = self._extract_text_from_message(message)
            
            # 预先估算用户消息的token数量
            try:
                estimated_tokens = TokenCounter.count_tokens(message_text, model)
                message.input_tokens = estimated_tokens
            except Exception as e:
                logger.error(f"计算用户消息token失败: {e}")
            
            # 尝试保存用户消息到历史记录
            try:
                await db_message_history.add_message(current_history_id, message)
            except Exception as e:
                logger.error(f"保存用户消息到历史记录失败，但继续处理: {e}")
            
            # 获取历史消息并转换为LLM消息格式
            chat_messages = []
            history = []
            try:
                history = await db_message_history.get_history(current_history_id)
                
                # 只取最近的10条消息，避免tokens过多
                for hist_msg in history[-10:]:
                    role = "user"
                    if hist_msg.sender.role == MessageRole.ASSISTANT:
                        role = "assistant"
                    elif hist_msg.sender.role == MessageRole.SYSTEM:
                        role = "system"
                    
                    # 从消息中提取文本内容
                    msg_text = self._extract_text_from_message(hist_msg)
                        
                    chat_messages.append(ChatMessage(
                        role=role,
                        content=msg_text
                    ))
            except Exception as e:
                logger.error(f"获取历史记录失败，只使用当前消息: {e}")
            
            # 如果没有历史消息，则只添加当前消息
            if not chat_messages:
                chat_messages = [ChatMessage(role="user", content=message_text)]
            
            # 在调用API之前估算所有消息的token
            try:
                openai_messages = []
                for msg in chat_messages:
                    openai_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                estimated_input_tokens = TokenCounter.estimate_openai_tokens(openai_messages, model)
            except Exception as e:
                logger.error(f"预估输入tokens失败: {e}")
                estimated_input_tokens = None
            
            # 流式调用LLM
            llm = self.llm_instances[model]
            
            # 准备存储完整响应内容
            full_response = ""
            
            # 创建AI响应消息对象
            response_message = LLMMessage(
                history_id=current_history_id,
                sender=MessageSender(role=MessageRole.ASSISTANT, nickname=model),
                components=[MessageComponent(
                    type=MessageType.TEXT,
                    content=""  # 初始为空，稍后填充
                )],
                message_str="",  # 初始为空，稍后填充
                source_model=model
            )
            
            # 流式返回结果
            async for chunk in llm.chat_completion_stream(chat_messages):
                full_response += chunk
                response_message.message_str = full_response
                response_message.components[0].content = full_response
                yield chunk
            
            # 在流式响应结束后，计算token使用情况
            try:
                # 计算输出token
                output_tokens = TokenCounter.count_tokens(full_response, model)
                response_message.output_tokens = output_tokens
                logger.debug(f"响应输出tokens: {output_tokens}")
                
                # 更新用户消息的输入token
                if estimated_input_tokens is not None:
                    message.input_tokens = estimated_input_tokens
                    await db_message_history.update_message(
                        current_history_id,
                        message.message_id,
                        {"input_tokens": estimated_input_tokens}
                    )
            except Exception as e:
                logger.error(f"计算token失败: {e}")
                
            # 将完整响应消息保存到历史记录
            try:
                await db_message_history.add_message(current_history_id, response_message)
                
                # 将token信息作为特殊消息返回
                token_info = json.dumps({
                    'input_tokens': estimated_input_tokens,
                    'output_tokens': output_tokens,
                    'message_id': response_message.message_id
                })
                yield f"__TOKEN_INFO__{token_info}"
            except Exception as e:
                logger.error(f"保存AI回复到历史记录失败: {e}")
                
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"处理消息时发生错误: {e}\n{error_trace}")
            yield f"错误: {str(e)}"

# 全局聊天管道实例
chat_pipeline = ChatPipeline()
