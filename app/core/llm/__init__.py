import json
import aiohttp
import traceback
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, AsyncGenerator

from app import logger

class ChatMessage(BaseModel):
    role: str
    content: str

class LLMResponse(BaseModel):
    text: str
    raw_response: Optional[Dict[str, Any]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str

class BaseLLM:
    def __init__(self):
        self.model_name = ""
        
    async def chat_completion(self, messages: List[ChatMessage]) -> LLMResponse:
        raise NotImplementedError
        
    async def chat_completion_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """流式返回LLM响应，需要在子类实现"""
        raise NotImplementedError

def handle_init_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            return None
    return wrapper
        
class OpenAILLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model_name = config.model_name
        
    async def chat_completion(self, messages: List[ChatMessage]) -> LLMResponse:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [m.dict() for m in messages]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    if "error" in result:
                        raise Exception(f"API错误: {result['error']}")
                    
                    # 提取token使用情况
                    input_tokens = None
                    output_tokens = None
                    
                    if "usage" in result:
                        if "prompt_tokens" in result["usage"]:
                            input_tokens = result["usage"]["prompt_tokens"]
                        if "completion_tokens" in result["usage"]:
                            output_tokens = result["usage"]["completion_tokens"]
                        
                    return LLMResponse(
                        text=result["choices"][0]["message"]["content"],
                        raw_response=result,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )
        except Exception as e:
            return LLMResponse(
                text=f"抱歉，服务出现了问题: {str(e)}",
                raw_response={"error": str(e)}
            )

    async def chat_completion_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [m.dict() for m in messages],
                "stream": True  # 开启流式响应
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"流式响应错误: API返回状态码 {response.status}"
                        return
                    
                    
                    # 调试第一块原始响应
                    first_chunk = await response.content.readany()
                    first_text = first_chunk.decode('utf-8', errors='replace')
                    
                    # 处理第一块
                    lines = first_text.split('\n')
                    for line in lines:
                        if not line.strip():
                            continue
                        
                        if line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        yield delta["content"]
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {e}, 原始文本: {line}")
                                
                    
                    # 继续处理剩余流
                    async for line in response.content:
                        line = line.decode('utf-8', errors='replace').strip()
                        if not line:
                            continue
                        
                        if line.startswith("data: "):
                            line = line[6:]
                            if line == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta and delta["content"]:
                                        yield delta["content"]
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON解析错误: {e}, 原始文本: {line}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"流式响应错误: {str(e)}"

class AnthropicLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model_name = config.model_name
        
    async def chat_completion(self, messages: List[ChatMessage]) -> LLMResponse:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # 将 OpenAI 消息格式转换为 Anthropic 格式
        claude_messages = []
        for msg in messages:
            claude_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        payload = {
            "model": self.model_name,
            "messages": claude_messages,
            "max_tokens": 1000
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                if "error" in result:
                    raise Exception(result["error"])
                
                # 提取token使用情况（Anthropic API的响应结构可能需要调整）
                input_tokens = None
                output_tokens = None
                
                if "usage" in result:
                    if "input_tokens" in result["usage"]:
                        input_tokens = result["usage"]["input_tokens"]
                    if "output_tokens" in result["usage"]:
                        output_tokens = result["usage"]["output_tokens"]
                    
                return LLMResponse(
                    text=result["content"][0]["text"],
                    raw_response=result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )
    
    async def chat_completion_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        # Anthropic流式响应实现可以后续添加
        yield "Anthropic流式响应尚未实现"

class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.base_url = config.base_url
        self.model_name = config.model_name
        # Ollama 通常不需要 API key，但我们保留这个字段以保持接口一致性
        self.api_key = config.api_key
        
    async def chat_completion(self, messages: List[ChatMessage]) -> LLMResponse:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            ollama_messages = [m.dict() for m in messages]
            
            payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    if "error" in result:
                        raise Exception(f"Ollama API错误: {result['error']}")
                    
                    # Ollama API 通常会在 response 包含 message 字段
                    return LLMResponse(
                        text=result.get("message", {}).get("content", ""),
                        raw_response=result,
                        # Ollama 可能提供的 token 统计信息
                        input_tokens=result.get("prompt_eval_count", None),
                        output_tokens=result.get("eval_count", None)
                    )
        except Exception as e:
            logger.error(f"Ollama API调用错误: {str(e)}")
            return LLMResponse(
                text=f"抱歉，Ollama服务出现了问题: {str(e)}",
                raw_response={"error": str(e)}
            )
            
    async def chat_completion_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            ollama_messages = [m.dict() for m in messages]
            
            payload = {
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"Ollama流式响应错误: API返回状态码 {response.status}"
                        return
                    
                    async for chunk in response.content:
                        if not chunk:
                            continue
                            
                        try:
                            data = json.loads(chunk)
                            # Ollama 的流式响应通常会包含 message 字段
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                            # 处理另一种可能的响应格式，直接包含 'response' 字段
                            elif "response" in data:
                                content = data["response"]
                                if content:
                                    yield content
                        except json.JSONDecodeError as e:
                            logger.error(f"Ollama JSON解析错误: {e}, 原始文本: {chunk}")
        except Exception as e:
            logger.error(f"Ollama流式响应错误: {str(e)}")
            yield f"Ollama流式响应错误: {str(e)}"
