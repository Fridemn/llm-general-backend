"""
TTS服务基类和TTS处理逻辑
"""
import os
import uuid
import logging
import aiohttp
from typing import Optional, Dict, Any
import asyncio

from app.core.config.voice_config import get_voice_config, get_voice_config_section

# 获取TTS配置
tts_config = get_voice_config_section("tts_service")
TTS_API_BASE = tts_config.get("api_base", "http://127.0.0.1:5000")
TTS_OUTPUT_DIR = tts_config.get("output_dir", "static/audio")

logger = logging.getLogger(__name__)

class TTSService:
    """TTS服务基类"""
    
    def __init__(self, api_base: Optional[str] = None):
        self.api_base = api_base or TTS_API_BASE
        
    async def synthesize(self, text: str, output_file: Optional[str] = None) -> str:
        """合成语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径，如果为None则自动生成
            
        Returns:
            str: 生成的音频文件路径
        """
        raise NotImplementedError("子类必须实现此方法")
        
    def get_output_path(self, filename: Optional[str] = None) -> str:
        """获取输出文件路径
        
        Args:
            filename: 文件名，如果为None则生成随机文件名
            
        Returns:
            str: 完整的输出文件路径
        """
        if filename is None:
            filename = f"tts_{uuid.uuid4()}.mp3"
            
        # 确保目录存在
        os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
        
        return os.path.join(TTS_OUTPUT_DIR, filename)
        
    def get_audio_url(self, filepath: str) -> str:
        """根据文件路径获取可访问的URL
        
        Args:
            filepath: 文件路径
            
        Returns:
            str: 可访问的URL
        """
        # 移除前置路径，只保留相对于静态目录的部分
        # 例如： static/audio/file.mp3 -> /static/audio/file.mp3
        if filepath.startswith(TTS_OUTPUT_DIR):
            relative_path = filepath[len(TTS_OUTPUT_DIR)-len("audio"):]
            return f"/static/{relative_path}"
        return f"/{filepath}"

class GSVITTSService(TTSService):
    """GSVI TTS服务实现"""
    
    async def synthesize(self, text: str, output_file: Optional[str] = None) -> str:
        """合成语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径，如果为None则自动生成
            
        Returns:
            str: 生成的音频文件路径
        """
        if not text:
            raise ValueError("文本不能为空")
            
        # 如果未指定输出文件，生成一个随机文件名
        if output_file is None:
            output_file = self.get_output_path()
            
        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        try:
            # 修正: 使用正确的API调用方式 - GET请求，/tts端点，URL参数
            import urllib.parse
            
            # 准备URL参数
            params = {
                "text": text,
                "character": "zh-CN-XiaoxiaoNeural",  # 可以从配置中读取
            }
            
            # 构建查询字符串
            query_parts = []
            for key, value in params.items():
                encoded_value = urllib.parse.quote(str(value))
                query_parts.append(f"{key}={encoded_value}")
                
            # 构建完整URL
            url = f"{self.api_base}/tts?{'&'.join(query_parts)}"
            logger.info(f"请求TTS服务: {url}")
            
            # 发送GET请求到TTS服务
            async with aiohttp.ClientSession() as session:
                # 添加重试机制
                max_retries = 2
                for attempt in range(max_retries + 1):
                    try:
                        async with session.get(url, timeout=30) as response:
                            if response.status == 200:
                                # 直接将响应内容写入文件
                                with open(output_file, "wb") as f:
                                    f.write(await response.read())
                                    
                                logger.info(f"TTS合成成功: {output_file}")
                                return output_file
                            else:
                                error_text = await response.text()
                                logger.error(f"TTS服务返回错误状态码: {response.status}, 错误: {error_text}")
                                
                                if attempt < max_retries:
                                    logger.info(f"尝试重试 ({attempt+1}/{max_retries})...")
                                    await asyncio.sleep(1)
                                    continue
                                    
                                raise Exception(f"TTS服务请求失败: {response.status}, 错误: {error_text}")
                    except aiohttp.ClientConnectorError as e:
                        logger.error(f"无法连接到TTS服务: {e}")
                        if attempt < max_retries:
                            logger.info(f"连接失败，尝试重试 ({attempt+1}/{max_retries})...")
                            await asyncio.sleep(2)
                            continue
                        raise Exception(f"无法连接到TTS服务: {e}")
                        
        except asyncio.TimeoutError:
            logger.error("TTS服务请求超时")
            # 超时时生成一个占位音频文件并返回
            return self._generate_fallback_audio(output_file)
        except Exception as e:
            logger.error(f"TTS合成异常: {str(e)}")
            # 在错误时也生成一个占位音频文件
            return self._generate_fallback_audio(output_file)
        
    def _generate_fallback_audio(self, output_file: str) -> str:
        """生成一个空的占位音频文件，当TTS失败时使用
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            str: 生成的文件路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 检查我们是否已经有一个静态的空音频文件可以复制
            fallback_audio = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         "static", "audio", "tts_error.mp3")
            
            if os.path.exists(fallback_audio):
                # 复制现有的静音音频文件
                import shutil
                shutil.copy(fallback_audio, output_file)
                logger.info(f"已生成TTS占位音频: {output_file}")
                return output_file
            else:
                # 如果没有静态文件，创建一个空文件
                with open(output_file, 'wb') as f:
                    # 写入一个最小的有效MP3文件头
                    f.write(b'\xFF\xFB\x90\x44\x00\x00\x00\x00')
                logger.info(f"已创建空的TTS占位音频: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"创建占位音频文件失败: {e}")
            return output_file

# 导出便捷函数
async def synthesize_text(text: str) -> Dict[str, Any]:
    """合成文本为语音
    
    Args:
        text: 要合成的文本
        
    Returns:
        Dict[str, Any]: 包含结果的字典 {"success": bool, "file_path": str, "audio_url": str}
    """
    try:
        service = GSVITTSService()
        file_path = await service.synthesize(text)
        audio_url = service.get_audio_url(file_path)
        
        return {
            "success": True,
            "file_path": file_path,
            "audio_url": audio_url
        }
    except Exception as e:
        logger.error(f"语音合成失败: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
