from typing import Dict, Any, Optional
import logging
import traceback

from fastapi import HTTPException

from app.core.pipeline.text_process import text_process
from app.core.llm.message import LLMMessage, MessageRole
from app.core.llm.db_history import db_message_history

logger = logging.getLogger("app")

class SummarizeProcess:
    """
    摘要处理流水线，负责自动总结对话内容等功能
    """
    
    def __init__(self):
        pass
    
    async def generate_history_title(self, history_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        自动总结历史记录内容并生成标题
        
        Args:
            history_id: 历史记录ID
            user_id: 用户ID，可选
            
        Returns:
            包含生成标题信息的字典
        """
        try:
            # 获取历史记录
            messages = await db_message_history.get_history(history_id)
            if not messages or len(messages) == 0:
                raise HTTPException(status_code=400, detail="历史记录为空，无法总结")
            
            # 准备提示信息
            prompt = "请根据以下对话内容，生成一个10个字以内的简短标题，只返回标题文本，不要有任何解释或额外文字：\n\n"
            
            # 添加最多5条消息以避免过长
            message_count = min(len(messages), 5)
            for i in range(message_count):
                msg = messages[-(message_count-i)]  # 获取最近的几条消息
                role_text = "用户" if msg.sender.role == MessageRole.USER else "AI"
                prompt += f"{role_text}: {msg.message_str[:100]}{'...' if len(msg.message_str) > 100 else ''}\n"
            
            # 构造消息请求
            input_message = LLMMessage.from_text(
                text=prompt,
                history_id="",  # 空字符串，让API自动创建新历史记录
                role=MessageRole.USER
            )
            
            # 生成临时历史ID用于后续清理
            temp_history_id = None
            
            # 调用 gpt-3.5-turbo 模型生成标题
            summary_model = "gpt-3.5-turbo"
            response = await text_process.process_message(
                summary_model,
                input_message
            )
            
            # 从响应中获取自动生成的历史ID
            if response and response.response_message and response.response_message.history_id:
                temp_history_id = response.response_message.history_id
                logger.info(f"发现临时历史ID: {temp_history_id}")
            
            # 提取生成的标题并清理
            title = response.response_message.message_str.strip()
            
            # 移除引号和额外的标点符号
            title = title.strip('"\'「」『』【】《》（）()[]').strip()
            
            # 限制长度
            if len(title) > 20:  # 给一点缓冲空间
                title = title[:20]
                
            logger.info(f"生成的标题: {title}")
            
            # 删除临时历史记录
            if temp_history_id:
                try:
                    await db_message_history.delete_history(temp_history_id)
                    logger.info(f"已删除临时历史记录: {temp_history_id}")
                except Exception as e:
                    logger.warning(f"删除临时历史记录失败: {e}")
            
            # 调用更新API
            success = await db_message_history.update_history_title(history_id, title)
            if not success:
                raise HTTPException(status_code=500, detail="更新历史记录标题失败")
                
            return {
                "success": True,
                "history_id": history_id,
                "new_title": title
            }
        
        except HTTPException:
            raise
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"总结历史记录标题失败: {str(e)}\n{error_trace}")
            raise HTTPException(status_code=500, detail=f"总结历史记录标题失败: {str(e)}")
    
    async def summarize_conversation(self, history_id: str, max_length: int = 250) -> str:
        """
        总结整个对话内容
        
        Args:
            history_id: 历史记录ID
            max_length: 摘要最大长度
            
        Returns:
            对话摘要
        """
        # 这是为未来扩展准备的方法，可以用于生成更详细的对话摘要
        # 暂时留空，后续可以实现
        pass

# 全局摘要处理流水线实例
summarize_process = SummarizeProcess()
