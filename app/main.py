from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any

api_system = APIRouter()

class SystemStatus(BaseModel):
    status: str
    details: Dict[str, Any]

@api_system.get("/status", response_model=SystemStatus)
async def get_system_status():
    """获取系统状态"""
    try:
        # 模拟系统状态检查
        status = "ok"
        details = {
            "version": "1.0.0",
            "uptime": "24 hours",
            "services": {
                "database": "connected",
                "cache": "connected",
                "message_queue": "connected"
            }
        }
        return SystemStatus(status=status, details=details)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )