"""
ws_tts_config.py
TTS WebSocket 服务配置
"""

# TTS 服务配置
TTS_SERVER_CONFIG = {
    # TTS 服务 URL，可以通过环境变量 TTS_SERVER_URL 覆盖
    "server_url": "http://localhost:50000/",
    
    # 可用音色列表
    "available_voices": [
        {"id": "", "name": "默认音色"},
        {"id": "女声", "name": "女声"},
        {"id": "男声", "name": "男声"},
        {"id": "少年音", "name": "少年音"},
        {"id": "萝莉音", "name": "萝莉音"}
    ],
    
    # 最大并发请求数
    "max_concurrent_requests": 5,
    
    # 连接超时时间（秒）
    "connection_timeout": 5.0,
    
    # 请求超时时间（秒）
    "request_timeout": 30.0
}
