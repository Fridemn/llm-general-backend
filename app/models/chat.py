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
    """聊天历史记录表，仅包含对话的基本信息"""
    history_id = fields.UUIDField(pk=True, default=uuid.uuid4, index=True)
    user_id = fields.UUIDField(null=True, description="关联的用户ID")
    title = fields.CharField(max_length=100, default="新对话", description="对话标题")
    create_time = fields.DatetimeField(auto_now_add=True, description="创建时间")
    update_time = fields.DatetimeField(auto_now=True, description="最后更新时间")
    
    class Meta:
        table = "chat_history"
        app = "models"

class ChatMessage(Model):
    """聊天消息表，存储具体消息内容"""
    message_id = fields.UUIDField(pk=True, default=uuid.uuid4, index=True)
    history = fields.ForeignKeyField('models.ChatHistory', related_name='messages')
    role = fields.CharField(max_length=20, description="消息角色")
    content = fields.TextField(description="消息内容")
    components = fields.JSONField(default=list, description="消息组件JSON")
    model = fields.CharField(max_length=100, null=True, description="模型名称")
    timestamp = fields.BigIntField(description="消息时间戳")
    input_tokens = fields.IntField(null=True, description="输入token数")
    output_tokens = fields.IntField(null=True, description="输出token数")
    source_model = fields.CharField(max_length=100, null=True, description="源模型")
    target_model = fields.CharField(max_length=100, null=True, description="目标模型")
    
    class Meta:
        table = "chat_message"
        app = "models"
