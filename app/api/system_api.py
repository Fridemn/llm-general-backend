from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio

from app import logger
from app.core.tts.tts_service import GSVITTSService
from app.core.config.voice_config import get_voice_config_section

api_system = APIRouter()

@api_system.get("/health")
async def health_check():
    """系统健康检查"""
    return {"status": "ok", "version": "1.0.0"}

@api_system.get("/service/status")
async def service_status() -> Dict[str, Any]:
    """获取各服务的状态"""
    result = {
        "status": "ok",
        "services": {}
    }
    
    # 检查TTS服务
    try:
        tts_config = get_voice_config_section("tts_service")
        api_base = tts_config.get("api_base")
        api_endpoint = tts_config.get("api_endpoint")
        
        tts_service = GSVITTSService(api_base=api_base, api_endpoint=api_endpoint)
        tts_health = await tts_service.check_service_health()
        
        result["services"]["tts"] = {
            "status": "online" if tts_health else "offline",
            "api_base": api_base,
            "api_endpoint": api_endpoint
        }
    except Exception as e:
        logger.error(f"TTS服务状态检查异常: {str(e)}")
        result["services"]["tts"] = {
            "status": "error",
            "error": str(e)
        }
    
    # 这里可以添加其他服务的健康检查
    
    # 如果有任何服务不可用，返回部分可用状态
    if any(service["status"] != "online" for service in result["services"].values()):
        result["status"] = "partially_available"
    
    return result
