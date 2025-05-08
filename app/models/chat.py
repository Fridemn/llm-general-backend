import uuid
from tortoise.models import Model
from tortoise import fields
from enum import Enum
from typing import List, Dict, Any


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
    history = fields.ForeignKeyField("models.ChatHistory", related_name="messages")
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

    @property
    def message_components(self) -> List[Dict[str, Any]]:
        """获取消息组件列表"""
        if not self.components:
            # 如果没有组件，尝试将content作为文本组件
            return [{"type": "text", "content": self.content, "extra": None}]
        return self.components

    def get_display_content(self) -> str:
        """获取可显示的消息内容"""
        if not self.components:
            return self.content

        text_parts = []
        for comp in self.components:
            if comp["type"] == "text":
                text_parts.append(comp["content"])
            elif comp["type"] == "audio":
                text_parts.append(f"[音频: {comp['content']}]")
            elif comp["type"] == "image":
                text_parts.append(f"[图片: {comp['content']}]")
            elif comp["type"] == "file":
                text_parts.append(f"[文件: {comp['content']}]")
            elif comp["type"] == "video":
                text_parts.append(f"[视频: {comp['content']}]")
            else:
                text_parts.append(f"[{comp['type']}: {comp['content']}]")

        return " ".join(text_parts)

    def has_component_type(self, component_type: str) -> bool:
        """检查消息是否包含特定类型的组件"""
        return any(comp["type"] == component_type for comp in self.message_components)

    def get_components_by_type(self, component_type: str) -> List[Dict[str, Any]]:
        """获取特定类型的所有组件"""
        return [comp for comp in self.message_components if comp["type"] == component_type]

    @classmethod
    async def from_llm_message(cls, llm_message, history_id):
        """从LLMMessage创建ChatMessage"""
        return await cls.create(
            message_id=uuid.UUID(llm_message.message_id),
            history_id=history_id,
            role=llm_message.sender.role,
            content=llm_message.message_str,
            components=[
                {"type": comp.type, "content": comp.content, "extra": comp.extra} for comp in llm_message.components
            ],
            timestamp=llm_message.timestamp,
            input_tokens=llm_message.input_tokens,
            output_tokens=llm_message.output_tokens,
            source_model=llm_message.source_model,
            target_model=llm_message.target_model,
        )
