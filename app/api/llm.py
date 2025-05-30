"""
llm.py
存放与语言模型相关的接口，如对话、历史记录等
"""

import os
import logging
import traceback
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form

from app import app_config, logger
from app.api.user import get_current_user
from app.core.llm.message import MessageRole
from app.core.db.db_history import db_message_history
from app.core.pipeline.chat_process import chat_process
from app.core.pipeline.function_process import function_process
from app.core.pipeline.summarize_process import summarize_process
from app.core.funcall.function_handler import function_handler

MODEL_TO_ENDPOINT = app_config.llm_config["model_to_endpoint"]

api_llm = APIRouter()

# 创建音频文件存储目录
AUDIO_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static", "audio")
os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    model: Optional[str] = None
    message: str
    history_id: Optional[str] = None
    role: MessageRole = MessageRole.USER


class HistoryUpdateRequest(BaseModel):
    title: str


class AudioChatRequest(BaseModel):
    file_path: str
    model: Optional[str] = None
    history_id: Optional[str] = None
    role: MessageRole = MessageRole.USER
    stt_model: Optional[str] = None


class UnifiedChatRequest(BaseModel):
    """统一的聊天请求模型"""

    model: Optional[str] = None
    message: Optional[str] = ""  # 文本消息，当stt=False时使用
    history_id: Optional[str] = None
    role: MessageRole = MessageRole.USER
    stream: bool = False  # 是否流式输出
    stt: bool = False  # 是否需要语音识别
    tts: bool = False  # 是否需要文本转语音(预留)
    stt_model: Optional[str] = None  # STT模型选择


# 统一的聊天接口
@api_llm.post("/chat")
async def unified_chat(
    model: Optional[str] = Form(None),
    message: Optional[str] = Form(""),
    history_id: Optional[str] = Form(None),
    role: MessageRole = Form(MessageRole.USER),
    stream: bool = Form(False),
    stt: bool = Form(False),
    tts: bool = Form(False),
    audio_file: Optional[UploadFile] = File(None),
    current_user=Depends(get_current_user),
):
    """
    统一的聊天接口，支持文本聊天、语音聊天、流式输出等多种功能
    """
    try:
        # 获取当前用户ID
        user_id = str(current_user["user_id"]) if current_user else None

        # 检查是否是函数调用命令
        if message:
            function_call = function_handler.detect_function_call_intent(message)
            if function_call:
                function_name, result, need_llm = function_handler.handle_function_call(function_call)

                function_message = function_handler.create_function_message(history_id or "", function_name, result)

                if history_id:
                    await db_message_history.add_message(history_id, function_message)

                need_llm = function_call.get("need_llm", need_llm)

                if need_llm:
                    return await function_process.handle_function_result(
                        model=model,
                        function_name=function_name,
                        result=result,
                        history_id=history_id,
                        stream=stream,
                        tts=tts,
                        user_id=user_id,
                        function_message=function_message,
                    )
                else:
                    if stream:
                        return function_process.create_function_stream_response(function_name, result)
                    else:
                        return {
                            "success": True,
                            "function_call": {"name": function_name, "result": result},
                            "message_id": function_message.message_id,
                        }

        # 使用聊天流水线处理请求
        return await chat_process.handle_request(
            model=model,
            message=message,
            history_id=history_id,
            role=role,
            stream=stream,
            stt=stt,
            tts=tts,
            audio_file=audio_file,
            user_id=user_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"统一聊天接口异常: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


# ------------------------------
# 历史记录部分，包括创建、获取、删除历史记录
# ------------------------------


@api_llm.post("/history/new")
async def create_new_history(current_user=Depends(get_current_user)):
    """创建新的历史记录"""
    try:
        user_id = str(current_user["user_id"]) if current_user else None
        history_id = await db_message_history.create_history(user_id)
        return {"history_id": history_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建历史记录失败: {str(e)}")


@api_llm.get("/history/{history_id}")
async def get_history(history_id: str, current_user=Depends(get_current_user)):
    """获取历史消息"""
    try:
        logger.info(f"请求历史记录: {history_id}")

        # 使用db_message_history获取格式化消息
        messages = await db_message_history.get_history(history_id)
        # logger.info(f"成功获取 {len(messages)} 条消息记录")

        # 返回消息列表，处理可能的序列化错误
        message_dicts = []
        for msg in messages:
            try:
                message_dict = msg.dict()
                # 确保history_id设置正确
                if not message_dict.get("history_id"):
                    message_dict["history_id"] = history_id
                message_dicts.append(message_dict)
            except Exception as e:
                logger.error(f"消息序列化失败: {str(e)}")
                # 创建简化版消息
                simple_msg = {
                    "message_id": msg.message_id,
                    "history_id": history_id,
                    "message_str": msg.message_str,
                    "timestamp": msg.timestamp,
                    "sender": {"role": "unknown", "nickname": None},
                    "components": [],
                }

                # 尝试添加角色信息
                if hasattr(msg, "sender"):
                    if hasattr(msg.sender, "role"):
                        simple_msg["sender"]["role"] = str(msg.sender.role)
                    if hasattr(msg.sender, "nickname"):
                        simple_msg["sender"]["nickname"] = msg.sender.nickname

                message_dicts.append(simple_msg)

        return {"history_id": history_id, "messages": message_dicts, "count": len(message_dicts)}
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"获取历史消息失败: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"获取历史消息失败: {str(e)}")


@api_llm.delete("/history/{history_id}")
async def delete_history(history_id: str, current_user=Depends(get_current_user)):
    """删除历史记录"""
    success = await db_message_history.delete_history(history_id)
    return {"success": success}


@api_llm.get("/user/histories")
async def get_user_histories(current_user=Depends(get_current_user)):
    """获取当前用户的所有聊天历史"""
    if not current_user:
        raise HTTPException(status_code=401, detail="未登录")

    histories = await db_message_history.get_user_histories(str(current_user["user_id"]))
    return {"histories": histories}


# ------------------------------
# 历史记录更新部分，包括更新标题、自动总结标题
# ------------------------------


@api_llm.put("/history/{history_id}")
async def update_history_title(history_id: str, request: HistoryUpdateRequest, current_user=Depends(get_current_user)):
    """更新历史记录标题"""
    success = await db_message_history.update_history_title(history_id, request.title)
    if not success:
        raise HTTPException(status_code=404, detail="历史记录不存在")
    return {"success": True}


@api_llm.post("/history/{history_id}/summarize")
async def summarize_history_title(history_id: str, current_user=Depends(get_current_user)):
    """自动总结历史记录内容并更新标题"""
    try:
        user_id = str(current_user["user_id"]) if current_user else None

        # 使用摘要流水线处理请求
        return await summarize_process.generate_history_title(history_id=history_id, user_id=user_id)

    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"总结历史记录标题失败: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"总结历史记录标题失败: {str(e)}")


# ------------------------------
# 模型部分，包括获取可用模型列表
# ------------------------------


@api_llm.get("/available-models")
async def get_available_models():
    return {"available-models": list(MODEL_TO_ENDPOINT.keys())}


@api_llm.get("/models")
async def get_models_by_provider():
    """获取按提供商分组的模型列表"""
    providers = {}

    for model, endpoint in MODEL_TO_ENDPOINT.items():
        if endpoint not in providers:
            providers[endpoint] = []
        providers[endpoint].append(model)

    return {"providers": providers, "all_models": list(MODEL_TO_ENDPOINT.keys())}
