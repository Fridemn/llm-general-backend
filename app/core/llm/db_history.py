import time
import uuid
import logging
from typing import List, Optional
import json
import traceback
from fastapi import HTTPException
from tortoise.exceptions import OperationalError, ConfigurationError
from tortoise import Tortoise
from tortoise.functions import Count

from app.models.chat import ChatHistory, ChatMessage, MessageRole
from app.core.llm.message import LLMMessage, MessageComponent, MessageType

logger = logging.getLogger("app")

class DBMessageHistory:
    """基于数据库的消息历史记录管理"""
    
    async def _ensure_connection(self):
        """确保数据库连接已初始化"""
        try:
            # 尝试一个简单的查询来测试连接
            await ChatHistory.all().limit(1)
            return True
        except ConfigurationError:
            return False
        except Exception as e:
            return False
    
    async def create_history(self, user_id: Optional[str] = None, title: str = "新对话") -> str:
        """创建新的历史记录"""
        # 检查连接
        if not await self._ensure_connection():
            raise HTTPException(status_code=500, detail="数据库连接未初始化")
        
        try:
            user_uuid = None
            if user_id and user_id.strip():
                try:
                    user_uuid = uuid.UUID(user_id)
                except ValueError:
                    logger.warning(f"无效的用户ID格式: {user_id}")
                    
            # 创建新的历史记录
            history = await ChatHistory.create(
                user_id=user_uuid,
                title=title
            )
            return str(history.history_id)
        except Exception as e:
            logger.error(f"创建历史记录失败: {e}")
            return str(uuid.uuid4())
    
    async def add_message(self, history_id: str, message: LLMMessage) -> bool:
        """添加消息到历史记录"""
        # 确保连接已初始化
        await self._ensure_connection()
        
        try:
            # 验证history_id
            if not history_id or not history_id.strip():
                logger.error("无效的历史记录ID")
                return False
            
            # 确保history_id存在
            try:
                history_uuid = uuid.UUID(history_id)
                history = await ChatHistory.filter(history_id=history_uuid).first()
                if not history:
                    # 如果历史记录不存在，则创建一个
                    history = await ChatHistory.create(
                        history_id=history_uuid
                    )
                
                # 格式化消息组件
                components = []
                if isinstance(message.components, list):
                    components = [
                        comp.dict() if hasattr(comp, 'dict') else comp 
                        for comp in message.components
                    ]
                
                # 处理角色值，确保只存储实际角色值（如"user"、"assistant"、"system"）
                role_value = message.sender.role
                # 如果角色是字符串形式的枚举（如"MessageRole.ASSISTANT"），提取实际值
                if isinstance(role_value, str) and role_value.startswith("MessageRole."):
                    role_value = role_value.split(".", 1)[1].lower()
                # 如果角色是枚举对象，获取其值
                elif hasattr(role_value, "value"):
                    role_value = role_value.value
                
                # 准备消息数据
                message_data = {
                    "message_id": uuid.UUID(message.message_id) if message.message_id else uuid.uuid4(),
                    "history": history,
                    "role": role_value,
                    "content": message.message_str,
                    "components": components,
                    "model": message.sender.nickname,
                    # 用户消息使用传入的时间戳，AI消息使用当前时间戳
                    "timestamp": (
                        message.timestamp 
                        if role_value in [MessageRole.USER, MessageRole.SYSTEM] and message.timestamp
                        else int(time.time())
                    ),
                }
                
                # 添加token相关信息
                if role_value == MessageRole.USER or role_value == MessageRole.SYSTEM:
                    if message.input_tokens is not None:
                        message_data['input_tokens'] = message.input_tokens
                    if message.target_model:
                        message_data['target_model'] = message.target_model
                elif role_value == MessageRole.ASSISTANT:
                    if message.output_tokens is not None:
                        message_data['output_tokens'] = message.output_tokens
                    if message.source_model:
                        message_data['source_model'] = message.source_model
                
                # 创建新消息记录
                await ChatMessage.create(**message_data)
                
                # 更新历史记录的更新时间
                history.update_time = time.time()
                await history.save()
                
                return True
            except ValueError as e:
                logger.error(f"无效的UUID格式: {e}")
                return False
                
        except OperationalError as e:
            logger.error(f"数据库操作错误: {e}")
            return False
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            traceback.print_exc()
            return False
    
    async def get_history(self, history_id: str) -> List[LLMMessage]:
        """获取历史记录中的消息"""
        # 确保连接已初始化
        await self._ensure_connection()
        
        try:
            # 验证history_id格式
            try:
                history_uuid = uuid.UUID(history_id)
            except ValueError:
                logger.error(f"无效的历史记录ID格式: {history_id}")
                return []
                
            # 检查历史记录是否存在
            history = await ChatHistory.filter(history_id=history_uuid).first()
            if not history:
                logger.warning(f"历史记录不存在: {history_id}")
                return []
                
            # 获取该历史记录的所有消息
            messages = await ChatMessage.filter(history_id=history_uuid).order_by('timestamp').all()
            
            result = []
            for msg in messages:
                try:
                    # 解析组件数据
                    components_data = msg.components
                        
                    # 构造组件对象
                    components = []
                    for comp_data in components_data:
                        if not isinstance(comp_data, dict):
                            continue
                            
                        comp_type = comp_data.get('type', MessageType.TEXT)
                        components.append(MessageComponent(
                            type=comp_type,
                            content=comp_data.get('content', ''),
                            extra=comp_data.get('extra')
                        ))
                    
                    # 准备token相关数据
                    token_data = {}
                    role = msg.role
                    
                    # 检查并修复角色格式
                    if role.startswith("MessageRole."):
                        actual_role = role.split(".", 1)[1].lower()
                        if actual_role in ["user", "assistant", "system"]:
                            role = actual_role
                        else:
                            role = "user"
                    
                    if role == MessageRole.USER or role == MessageRole.SYSTEM:
                        if msg.input_tokens is not None:
                            token_data['input_tokens'] = msg.input_tokens
                        if msg.target_model:
                            token_data['target_model'] = msg.target_model
                    elif role == MessageRole.ASSISTANT:
                        if msg.output_tokens is not None:
                            token_data['output_tokens'] = msg.output_tokens
                        if msg.source_model:
                            token_data['source_model'] = msg.source_model
                    
                    # 创建消息对象
                    message = LLMMessage(
                        message_id=str(msg.message_id),
                        history_id=history_id,
                        sender={
                            "role": role,
                            "nickname": msg.model
                        },
                        components=components,
                        message_str=msg.content,
                        timestamp=msg.timestamp,
                        **token_data
                    )
                    result.append(message)
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"处理消息时出错: {str(e)}\n{error_trace}")
            
            return result
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"获取历史记录失败: {str(e)}\n{error_trace}")
            return []
    
    async def delete_history(self, history_id: str) -> bool:
        """删除历史记录"""
        # 确保连接已初始化
        await self._ensure_connection()
        
        try:
            # 验证history_id格式
            try:
                history_uuid = uuid.UUID(history_id)
            except ValueError:
                logger.error(f"无效的历史记录ID格式: {history_id}")
                return False
            
            # 首先删除关联的消息
            await ChatMessage.filter(history_id=history_uuid).delete()
            
            # 然后删除历史记录
            deleted_count = await ChatHistory.filter(history_id=history_uuid).delete()
            if deleted_count > 0:
                return True
            else:
                logger.warning(f"要删除的历史记录不存在: {history_id}")
                return False
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"删除历史记录失败: {e}\n{error_trace}")
            return False

    async def get_user_histories(self, user_id: str) -> List[dict]:
        """获取用户的所有对话历史"""
        # 确保连接已初始化
        await self._ensure_connection()
        
        try:
            # 获取所有历史记录
            histories = await ChatHistory.filter(user_id=user_id).order_by('-update_time').all()
            
            result = []
            for h in histories:
                # 获取每个历史记录的消息数量
                message_count = await ChatMessage.filter(history_id=h.history_id).count()
                
                result.append({
                    "history_id": str(h.history_id),
                    "title": h.title,
                    "create_time": h.create_time.isoformat(),
                    "update_time": h.update_time.isoformat(),
                    "message_count": message_count
                })
            
            return result
        except Exception as e:
            logger.error(f"获取用户历史记录失败: {e}")
            return []
    
    async def update_history_title(self, history_id: str, title: str) -> bool:
        """更新历史记录的标题"""
        # 确保连接已初始化
        await self._ensure_connection()
        
        try:
            history = await ChatHistory.filter(history_id=history_id).first()
            if history:
                history.title = title
                await history.save()
                return True
            return False
        except Exception as e:
            logger.error(f"更新历史记录标题失败: {e}")
            return False

    async def update_message(self, history_id: str, message_id: str, updates: dict) -> bool:
        """更新历史记录中的特定消息"""
        try:
            # 直接更新消息表中的记录
            message = await ChatMessage.filter(
                history_id=uuid.UUID(history_id),
                message_id=uuid.UUID(message_id)
            ).first()
            
            if not message:
                return False
                
            # 更新消息字段
            for key, value in updates.items():
                setattr(message, key, value)
                
            await message.save()
            return True
        except Exception as e:
            logger.error(f"更新消息失败: {e}")
            return False


# 创建全局实例
db_message_history = DBMessageHistory()
