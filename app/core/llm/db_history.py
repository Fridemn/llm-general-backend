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

from app.models.chat import ChatHistory, MessageRole
from app.core.llm.message import LLMMessage, MessageComponent, MessageType

logger = logging.getLogger("app")

class DBMessageHistory:
    """基于数据库的消息历史记录管理"""
    
    async def _ensure_connection(self):
        """确保数据库连接已初始化"""
        try:
            # 尝试一个简单的查询来测试连接
            await ChatHistory.all().limit(1)
            # logger.debug("数据库连接已经初始化")
            return True
        except ConfigurationError:
            # logger.error("数据库连接未初始化，模型可能未正确注册")
            return False
        except Exception as e:
            # logger.error(f"数据库连接测试失败: {e}")
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
                    
            # 创建新的历史记录，使用空列表初始化messages字段
            history = await ChatHistory.create(
                user_id=user_uuid,
                title=title,
                messages=[]  # 初始化为空列表
            )
            # logger.info(f"成功创建历史记录: {history.history_id}")
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
                        history_id=history_uuid,
                        messages=[]
                    )
                
                # 格式化消息组件
                components = []
                if isinstance(message.components, list):
                    components = [
                        comp.dict() if hasattr(comp, 'dict') else comp 
                        for comp in message.components
                    ]
                
                # 准备token相关数据
                token_data = {}
                if message.sender.role == MessageRole.USER or message.sender.role == MessageRole.SYSTEM:
                    if message.input_tokens is not None:
                        token_data['input_tokens'] = message.input_tokens
                    if message.target_model:
                        token_data['target_model'] = message.target_model
                elif message.sender.role == MessageRole.ASSISTANT:
                    if message.output_tokens is not None:
                        token_data['output_tokens'] = message.output_tokens
                    if message.source_model:
                        token_data['source_model'] = message.source_model
                
                # 构造消息字典
                message_dict = {
                    "message_id": message.message_id,
                    "role": str(message.sender.role),  # 确保存储的是枚举值字符串，而不是枚举名称
                    "content": message.message_str,
                    "components": components,
                    "model": message.sender.nickname,
                    "timestamp": message.timestamp or int(time.time()),
                    **token_data
                }
                
                # 获取现有消息列表
                messages = history.get_messages_list()
                
                # 添加新消息
                messages.append(message_dict)
                
                # 更新历史记录
                history.messages = messages
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
                
            # 获取消息列表
            messages_data = history.get_messages_list()
            
            # 对消息按时间戳排序
            try:
                messages_data.sort(key=lambda x: x.get('timestamp', 0))
            except Exception as e:
                logger.error(f"排序消息失败: {e}")
            
            # logger.info(f"从历史记录 {history_id} 获取了 {len(messages_data)} 条消息")
            
            result = []
            for msg_data in messages_data:
                try:
                    # 解析组件数据
                    components_data = msg_data.get("components", [])
                        
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
                    role = msg_data.get("role")
                    
                    # 处理消息角色，确保是有效的枚举值
                    role = msg_data.get("role", "user")
                    
                    # 检查并修复角色格式
                    if role.startswith("MessageRole."):
                        # 如果是 "MessageRole.USER" 这种格式，提取实际的角色
                        actual_role = role.split(".", 1)[1].lower()
                        if actual_role in ["user", "assistant", "system"]:
                            role = actual_role
                        else:
                            role = "user"  # 默认为用户角色
                    
                    if role == MessageRole.USER or role == MessageRole.SYSTEM:
                        if "input_tokens" in msg_data:
                            token_data['input_tokens'] = msg_data.get("input_tokens")
                        if "target_model" in msg_data:
                            token_data['target_model'] = msg_data.get("target_model")
                    elif role == MessageRole.ASSISTANT:
                        if "output_tokens" in msg_data:
                            token_data['output_tokens'] = msg_data.get("output_tokens")
                        if "source_model" in msg_data:
                            token_data['source_model'] = msg_data.get("source_model")
                    
                    # 创建消息对象
                    message = LLMMessage(
                        message_id=msg_data.get("message_id", str(uuid.uuid4())),
                        history_id=history_id,
                        sender={
                            "role": role,  # 使用修复后的角色
                            "nickname": msg_data.get("model")
                        },
                        components=components,
                        message_str=msg_data.get("content", ""),
                        timestamp=msg_data.get("timestamp", int(time.time())),
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
                
            # 直接删除历史记录
            deleted_count = await ChatHistory.filter(history_id=history_uuid).delete()
            if deleted_count > 0:
                # logger.info(f"成功删除历史记录: {history_id}")
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
            histories = await ChatHistory.filter(user_id=user_id).order_by('-update_time').all()
            return [
                {
                    "history_id": str(h.history_id),
                    "title": h.title,
                    "create_time": h.create_time.isoformat(),
                    "update_time": h.update_time.isoformat(),
                    "message_count": len(h.get_messages_list())  # 添加消息数量统计
                }
                for h in histories
            ]
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
            history_uuid = uuid.UUID(history_id)
            history = await ChatHistory.filter(history_id=history_uuid).first()
            if not history:
                return False
                
            messages = history.get_messages_list()
            updated = False
            
            for i, msg in enumerate(messages):
                if msg.get("message_id") == message_id:
                    # 更新消息字段
                    messages[i].update(updates)
                    updated = True
                    break
            
            if updated:
                # 保存更新后的消息列表
                history.messages = messages
                await history.save()
                return True
            
            return False
        except Exception as e:
            logger.error(f"更新消息失败: {e}")
            return False


# 创建全局实例
db_message_history = DBMessageHistory()
