import time
import uuid
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Any
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
    
    @classmethod
    def create_text(cls, text: str) -> "MessageComponent":
        """创建文本类型组件"""
        return cls(type=MessageType.TEXT, content=text)
    
    @classmethod
    def create_audio(cls, audio_url: str, duration: Optional[float] = None, 
                     format: Optional[str] = None) -> "MessageComponent":
        """创建音频类型组件"""
        extra = {"duration": duration, "format": format}
        return cls(type=MessageType.AUDIO, content=audio_url, extra={k: v for k, v in extra.items() if v is not None})
    
    def to_display_text(self) -> str:
        """返回组件的可显示文本"""
        if self.type == MessageType.TEXT:
            return self.content
        elif self.type == MessageType.AUDIO:
            return f"[音频: {self.content}]"
        elif self.type == MessageType.IMAGE:
            return f"[图片: {self.content}]"
        elif self.type == MessageType.FILE:
            return f"[文件: {self.content}]"
        elif self.type == MessageType.VIDEO:
            return f"[视频: {self.content}]"
        return f"[{self.type}: {self.content}]"

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
            components=[MessageComponent.create_text(text)],
            message_str=text
        )
        
    @classmethod
    def from_audio(cls, audio_url: str, history_id: str, 
                   role: MessageRole = MessageRole.USER, 
                   duration: Optional[float] = None, 
                   format: Optional[str] = None):
        component = MessageComponent.create_audio(audio_url, duration, format)
        return cls(
            history_id=history_id,
            sender=MessageSender(role=role),
            components=[component],
            message_str=component.to_display_text()
        )
    
    @classmethod
    def from_components(cls, components: List[MessageComponent], history_id: str, 
                       role: MessageRole = MessageRole.USER):
        """从多个组件创建混合类型消息"""
        message_str = " ".join(comp.to_display_text() for comp in components)
        return cls(
            history_id=history_id,
            sender=MessageSender(role=role),
            components=components,
            message_str=message_str
        )
    
    def add_component(self, component: MessageComponent) -> "LLMMessage":
        """添加新的组件到现有消息"""
        self.components.append(component)
        self.message_str = " ".join(comp.to_display_text() for comp in self.components)
        return self

class LLMResponse(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    response_message: LLMMessage
    raw_response: Optional[dict] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
