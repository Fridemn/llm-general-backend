#------------------------------
#历史记录部分，包括创建、获取、删除历史记录
#------------------------------

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
        
        # # 验证历史记录存在
        # try:
        #     history = await ChatHistory.get(history_id=history_id)
        #     logger.info(f"找到历史记录: {history_id}, 标题: {history.title}")
        # except DoesNotExist:
        #     logger.warning(f"历史记录不存在: {history_id}")
        #     return {"history_id": history_id, "messages": []}
        
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
                    "sender": {
                        "role": "unknown",
                        "nickname": None
                    },
                    "components": []
                }
                
                # 尝试添加角色信息
                if hasattr(msg, "sender"):
                    if hasattr(msg.sender, "role"):
                        simple_msg["sender"]["role"] = str(msg.sender.role)
                    if hasattr(msg.sender, "nickname"):
                        simple_msg["sender"]["nickname"] = msg.sender.nickname
                        
                message_dicts.append(simple_msg)
        
        return {
            "history_id": history_id,
            "messages": message_dicts,
            "count": len(message_dicts)
        }
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

#------------------------------
#历史记录更新部分，包括更新标题、自动总结标题
#------------------------------

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
        # 获取历史记录
        try:
            messages = await db_message_history.get_history(history_id)
            if not messages or len(messages) == 0:
                raise HTTPException(status_code=400, detail="历史记录为空，无法总结")
        except Exception as e:
            logger.error(f"获取历史记录失败: {e}")
            raise HTTPException(status_code=404, detail=f"获取历史记录失败: {str(e)}")
        
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

#------------------------------
#模型部分，包括获取可用模型列表
#------------------------------

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
    
    return {
        "providers": providers,
        "all_models": list(MODEL_TO_ENDPOINT.keys())
    }