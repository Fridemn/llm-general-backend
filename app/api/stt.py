from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
import logging
import os
from typing import Optional, Dict, Any

from app.utils.stt import STTProcessor, STTStrategy
from app import app_config

logger = logging.getLogger("app")
api_stt = APIRouter()


class AudioPathRequest(BaseModel):
    """音频文件路径请求模型"""
    file_path: str
    model: Optional[str] = None
    translate_to_english: bool = False
    


@api_stt.post("/transcribe")
async def transcribe_audio(request: AudioPathRequest):
    """
    接收音频文件路径，返回识别后的文本
    
    参数:
    - file_path: 音频文件的路径
    - model: 可选，使用的模型名称
    - translate_to_english: 是否翻译为英语
    
    返回:
    - 识别后的文本或翻译后的文本
    """
    try:
        # 验证文件是否存在
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"文件未找到: {request.file_path}")
            
        # 从配置中加载STT设置
        stt_config = app_config.stt_config
        api_key = stt_config['whisper']['api_key']
        
        # 如果用户没有指定模型，则使用配置中的第一个模型
        model = request.model or stt_config['whisper']['available_models'][0]
        api_base = request.model or stt_config['whisper']['base_url']
        
        # 根据是否需要翻译选择策略
        strategy = STTStrategy.TRANSLATE if request.translate_to_english else STTStrategy.STANDARD
        
        # 初始化处理器
        processor = STTProcessor(api_key=api_key, model=model, strategy=strategy)
        
        # 处理音频文件
        result = processor.process(request.file_path)
        
        return {"success": True, "text": result}
        
    except Exception as e:
        logger.error(f"转录过程中发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")