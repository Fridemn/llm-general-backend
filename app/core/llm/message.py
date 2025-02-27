from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field
import time
import uuid

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

class MessageComponent(BaseModel):
    type: MessageType
    content: str
    extra: Optional[dict] = None

class MessageSender(BaseModel):
    role: MessageRole
    nickname: Optional[str] = None

class LLMMessage(BaseModel):
    """统一的消息格式"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    history_id: str
    sender: MessageSender
    components: List[MessageComponent]
    message_str: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    raw_message: Optional[Any] = None
    
    # 添加token统计相关字段
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    target_model: Optional[str] = None
    source_model: Optional[str] = None

    @classmethod
    def from_text(cls, text: str, history_id: str, role: MessageRole = MessageRole.USER):
        return cls(
            history_id=history_id,
            sender=MessageSender(role=role),
            components=[MessageComponent(type=MessageType.TEXT, content=text)],
            message_str=text
        )

class LLMResponse(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_message: LLMMessage
    raw_response: Optional[dict] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
