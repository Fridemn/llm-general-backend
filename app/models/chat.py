import uuid
from tortoise.models import Model
from tortoise import fields
from enum import Enum
import json
import logging

logger = logging.getLogger("app")

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatHistory(Model):
    """聊天历史记录表，包含所有消息"""
    history_id = fields.UUIDField(pk=True, default=uuid.uuid4, index=True)
    user_id = fields.UUIDField(null=True, description="关联的用户ID")
    title = fields.CharField(max_length=100, default="新对话", description="对话标题")
    create_time = fields.DatetimeField(auto_now_add=True, description="创建时间")
    update_time = fields.DatetimeField(auto_now=True, description="最后更新时间")
    messages = fields.JSONField(default=list, description="所有消息的JSON列表")
    
    class Meta:
        table = "chat_history"
        app = "models"
    
    def get_messages_list(self):
        """获取消息列表，如果为空则返回空列表"""
        if not self.messages:
            return []
            
        if isinstance(self.messages, str):
            try:
                return json.loads(self.messages)
            except:
                logger.error(f"解析消息JSON失败，历史记录ID: {self.history_id}")
                return []
        
        return self.messages
