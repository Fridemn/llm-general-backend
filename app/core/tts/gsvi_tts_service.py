"""
gsvi_tts_service.py
GSVI TTS服务实现
"""

import aiohttp
import os
import asyncio
import urllib.parse
from typing import Optional, Dict, Any

from app import logger

class GSVITTSService:
    """GSVI TTS服务实现"""
    
    def __init__(self, api_base: str = None):
        """初始化GSVI TTS服务
        
        Args:
            api_base: API基础URL，默认为http://127.0.0.1:5000
        """
        self.api_base = api_base or "http://127.0.0.1:5000"
    
    async def synthesize(self, text: str, output_file: str, 
                         character: Optional[str] = None, 
                         emotion: Optional[str] = None) -> str:
        """
        将文本转换为语音并保存到文件
        
        Args:
            text: 要转换的文本
            output_file: 输出文件路径
            character: 角色名称（可选）
            emotion: 情感（可选）
            
        Returns:
            输出文件路径
        """
        try:
            params = {"text": text}
            
            # 添加可选参数
            if character:
                params["character"] = character
            if emotion:
                params["emotion"] = emotion
                
            # 构建查询字符串
            query_parts = []
            for key, value in params.items():
                encoded_value = urllib.parse.quote(str(value))
                query_parts.append(f"{key}={encoded_value}")
                
            # 构建完整URL
            api_base = self.api_base.rstrip("/")  # 移除尾部斜杠
            url = f"{api_base}/tts?{'&'.join(query_parts)}"
            
            # logger.info(f"调用GSVI TTS服务: {url[:100]}...")
            
            # 发送HTTP请求
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        # 创建目录（如果不存在）
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        
                        # 写入响应内容到文件
                        with open(output_file, "wb") as f:
                            f.write(await response.read())
                            
                        return output_file
                    else:
                        error_text = await response.text()
                        logger.error(f"GSVI TTS API调用失败: 状态码={response.status}, 错误={error_text}")
                        raise Exception(f"GSVI TTS API调用失败: 状态码={response.status}")
        
        except Exception as e:
            logger.error(f"GSVI TTS合成失败: {str(e)}", exc_info=True)
            raise
