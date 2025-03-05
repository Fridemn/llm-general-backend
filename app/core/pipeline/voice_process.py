from typing import Optional, Dict, Any
import os
import uuid
import shutil
import tempfile
import logging
from datetime import datetime

from app.core.stt.stt_factory import STTFactory
from app.core.tts.tts_factory import TTSFactory
from app.core.llm.message import MessageComponent, MessageType

logger = logging.getLogger("app")

# 创建音频文件存储目录
AUDIO_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'static', 'audio')
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)

class VoiceProcess:
    """
    语音处理流水线，负责STT和TTS功能
    """
    
    def __init__(self):
        pass
        
    async def process_stt(self, audio_file_path: str, stt_model: Optional[str] = None, translate_to_english: bool = False) -> str:
        """
        语音转文本处理
        
        Args:
            audio_file_path: 音频文件路径
            stt_model: 使用的STT模型
            translate_to_english: 是否翻译为英语
            
        Returns:
            转写的文本内容
        """
        # 使用STT工厂获取STT提供者
        stt_provider = STTFactory.create_provider()
        
        if stt_provider is None:
            logger.error("无法创建STT提供者")
            raise ValueError("无法创建STT提供者")
        
        # 设置额外参数（如翻译到英文）
        extra_params = {"translate_to_english": translate_to_english}
        if stt_model:
            extra_params["model"] = stt_model
        
        # 使用提供者处理音频
        transcribed_text = await stt_provider.transcribe(audio_file_path, **extra_params)
        logger.info(f"语音转录结果: {transcribed_text}")
        
        # 检查转录结果是否为空
        if not transcribed_text.strip():
            raise ValueError("语音转录结果为空")
            
        return transcribed_text
    
    async def save_audio_file(self, audio_file, temp_dir: Optional[str] = None) -> Dict[str, str]:
        """
        保存音频文件到临时目录和永久存储
        
        Args:
            audio_file: FastAPI的UploadFile对象
            temp_dir: 临时目录路径
            
        Returns:
            包含临时和永久文件路径的字典
        """
        # 如果没有提供临时目录，创建一个
        should_delete_temp = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            should_delete_temp = True
            
        file_extension = os.path.splitext(audio_file.filename)[1]
        audio_file_path = os.path.join(temp_dir, f"audio_{uuid.uuid4()}{file_extension}")
        
        # 保存上传的文件到临时位置
        with open(audio_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # 为永久存储创建文件名（使用时间戳和UUID确保唯一性）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uuid.uuid4()}{file_extension}"
        permanent_audio_path = os.path.join(AUDIO_STORAGE_DIR, filename)
        
        logger.info(f"音频文件临时保存至: {audio_file_path}")
        logger.info(f"音频文件将永久保存至: {permanent_audio_path}")
        
        return {
            "temp_path": audio_file_path,
            "permanent_path": permanent_audio_path,
            "filename": filename,
            "temp_dir": temp_dir,
            "should_delete_temp": should_delete_temp
        }
    
    def save_to_permanent_storage(self, temp_path: str, permanent_path: str) -> bool:
        """
        将临时音频文件复制到永久存储位置
        
        Args:
            temp_path: 临时文件路径
            permanent_path: 永久存储文件路径
            
        Returns:
            是否成功保存
        """
        try:
            shutil.copy2(temp_path, permanent_path)
            logger.info(f"音频文件已复制到永久存储位置: {permanent_path}")
            return True
        except Exception as e:
            logger.error(f"将音频文件复制到永久位置失败: {str(e)}")
            return False
    
    def cleanup_temp_files(self, temp_dir: str) -> None:
        """
        清理临时文件和目录
        
        Args:
            temp_dir: 临时目录路径
        """
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"临时目录已清理: {temp_dir}")
            except Exception as e:
                logger.error(f"清理临时目录失败: {str(e)}")
    
    async def process_tts(self, text: str) -> Optional[str]:
        """
        文本转语音处理
        
        Args:
            text: 需要转为语音的文本
            
        Returns:
            生成的音频文件路径，如果失败则为None
        """
        if not text.strip():
            logger.warning("TTS输入文本为空")
            return None
            
        try:
            # 使用TTS工厂获取TTS提供者
            tts_provider = TTSFactory.create_provider()
            
            if tts_provider is None:
                logger.error("无法创建TTS提供者")
                return None
                
            # 生成音频文件
            audio_path = await tts_provider.get_audio(text)
            
            if audio_path:
                logger.info(f"TTS音频生成成功: {audio_path}")
                # 返回web访问路径
                return f"/static/audio/{os.path.basename(audio_path)}"
            else:
                logger.error("TTS音频生成失败，路径为空")
                return None
                
        except Exception as e:
            logger.error(f"TTS处理失败: {str(e)}")
            return None
    
    def create_audio_component(self, audio_path: str, text: str = "", 
                              extra_info: Optional[Dict[str, Any]] = None) -> MessageComponent:
        """
        创建音频消息组件
        
        Args:
            audio_path: 音频文件路径
            text: 相关文本
            extra_info: 额外信息
            
        Returns:
            创建的音频消息组件
        """
        extra = extra_info or {}
        
        if text:
            # 如果有文本，则添加到extra信息中
            extra["original_text"] = text[:100] + "..." if len(text) > 100 else text
            
        return MessageComponent(
            type=MessageType.AUDIO,
            content=audio_path,
            extra=extra
        )

# 全局语音处理实例
voice_process = VoiceProcess()
