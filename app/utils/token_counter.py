import tiktoken
from typing import List, Union, Dict, Optional, Any
import logging

logger = logging.getLogger("app")

class TokenCounter:
    """Token计数工具类"""
    
    # 常用编码器缓存
    _encoders = {}
    
    @classmethod
    def get_encoder(cls, model: str):
        """获取对应模型的编码器"""
        if model in cls._encoders:
            return cls._encoders[model]
        
        # 模型到编码器的映射
        MODEL_ENCODINGS = {
            "gpt-4": "gpt-4",
            "gpt-3.5": "gpt-3.5-turbo",
            "claude": "gpt-4"
        }
        
        try:
            # 根据模型前缀获取对应的编码器名称
            encoding_name = next(
                (enc for prefix, enc in MODEL_ENCODINGS.items() if model.startswith(prefix)),
                "cl100k_base"  # 默认编码器
            )
            
            # 获取编码器实例
            encoding = (tiktoken.encoding_for_model(encoding_name) 
                      if encoding_name != "cl100k_base"
                      else tiktoken.get_encoding(encoding_name))
            
            cls._encoders[model] = encoding
            return encoding
            
        except Exception as e:
            logger.warning(f"获取编码器失败 [{model}]: {str(e)}")
            try:
                # 尝试使用默认编码器
                encoding = tiktoken.get_encoding("cl100k_base")
                cls._encoders[model] = encoding
                return encoding
            except Exception as e:
                logger.error(f"获取默认编码器失败: {str(e)}")
                return None
    
    @classmethod
    def count_tokens(cls, text: str, model: str = "gpt-3.5-turbo") -> int:
        """计算单个文本的token数量"""
        encoder = cls.get_encoder(model)
        if not encoder:
            # 如果无法获取编码器，使用简单的估算（每4个字符约1个token）
            return len(text) // 4 + 1
        
        try:
            tokens = encoder.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"计算文本token失败: {str(e)}")
            # 如果编码失败，使用简单的估算
            return len(text) // 4 + 1
    
    @classmethod
    def estimate_openai_tokens(cls, messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo") -> int:
        """
        使用OpenAI官方推荐的方法估算消息列表的token数量
        参考: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoder = cls.get_encoder(model)
            if not encoder:
                # 回退到简单估算
                return cls._simple_token_estimate(messages)
            
            # 不同模型的令牌计数略有不同
            tokens_per_message = 3  # 每条消息的元数据开销
            tokens_per_name = 1     # 如果有name字段则额外+1
            
            # 根据模型调整令牌计数规则
            if model.startswith("gpt-3.5-turbo"):
                tokens_per_message = 4  # 对于3.5模型，每条消息开销为4 tokens
                # 添加回复格式的令牌开销
                num_tokens = 2  # 对于3.5模型，回复格式开销为2 tokens
            elif model.startswith("gpt-4"):
                tokens_per_message = 3  # 对于GPT-4，每条消息开销为3 tokens
                # 添加回复格式的令牌开销
                num_tokens = 3  # 对于GPT-4，回复格式开销为3 tokens
            else:
                # 其他模型，使用保守估计
                num_tokens = 3
            
            # 计算每条消息的tokens
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if isinstance(value, str):
                        # 编码内容并统计tokens
                        if key == "name":
                            num_tokens += tokens_per_name
                        tokens = encoder.encode(value)
                        num_tokens += len(tokens)
                    elif isinstance(value, list):
                        # 处理可能的嵌套内容
                        flat_text = " ".join(str(item) for item in value)
                        tokens = encoder.encode(flat_text)
                        num_tokens += len(tokens)
            
            return num_tokens
            
        except Exception as e:
            logger.error(f"OpenAI token估算失败: {str(e)}")
            # 回退到简单估算
            return cls._simple_token_estimate(messages)
    
    @classmethod
    def _simple_token_estimate(cls, messages: List[Dict[str, Any]]) -> int:
        """简单token估算，当编码器不可用时使用"""
        total_tokens = 0
        # 每条消息基础开销
        base_tokens_per_message = 4
        
        for message in messages:
            total_tokens += base_tokens_per_message
            
            # 计算内容tokens
            if "content" in message and isinstance(message["content"], str):
                # 粗略估计：平均4个字符约为1个token
                content_tokens = len(message["content"]) // 4 + 1
                total_tokens += content_tokens
        
        # 额外开销
        total_tokens += 3
        
        return total_tokens
    
    @classmethod
    def estimate_tokens_for_conversation(cls, messages: List[Dict], model: str = "gpt-3.5-turbo") -> Dict[str, int]:
        """
        估算对话的token数量，返回详细的token统计
        返回: {"total": 总token数, "per_message": [每条消息的token数列表]}
        """
        try:
            # 使用OpenAI方式估算总tokens
            total_tokens = cls.estimate_openai_tokens(messages, model)
            
            # 估算每条消息的tokens
            per_message_tokens = []
            for message in messages:
                if "content" in message and isinstance(message["content"], str):
                    # 为每条消息添加基本开销(大约3-4个token)
                    if model.startswith("gpt-3.5"):
                        base_tokens = 4
                    else:
                        base_tokens = 3
                        
                    # 计算文本内容的tokens
                    content_tokens = cls.count_tokens(message["content"], model)
                    message_tokens = base_tokens + content_tokens
                    
                    # 如果有name字段，额外添加1个token
                    if "name" in message and message["name"]:
                        message_tokens += 1
                        
                    per_message_tokens.append(message_tokens)
                else:
                    # 没有文本内容的消息，只计基本开销
                    per_message_tokens.append(3)
            
            return {
                "total": total_tokens,
                "per_message": per_message_tokens
            }
                
        except Exception as e:
            logger.error(f"对话token计算失败: {str(e)}")
            # 返回一个基本估计
            return {
                "total": cls._simple_token_estimate(messages),
                "per_message": [5] * len(messages)  # 每条消息默认5个token
            }
