""" 
存储默认的配置文件
"""
from .secret import *
from .constant import *

DEFAULT_CONFIG={
    "user_config":{
        "user_points":{
            "init_points": 0,
            "invite_points": 1000,
        },
    },
    "mysql_config": {
        "connections": {
            "default": {
                "engine": "tortoise.backends.mysql",
                "credentials": {
                    "host": mysql_server,
                    "port": mysql_port,
                    "user": mysql_user,
                    "password": mysql_password,
                    "database": mysql_database,
                },
            },
        }, 
        "apps": {
            "models": {
                "models": [
                    "app.models.user",
                    "app.models.chat",
                ],
                "default_connection": "default",
            },
        },
        'use_tz': False,
        'time_zone': 'Asia/Shanghai'       
    },
    "redis_config": {
        "host": redis_host,
        "password": redis_password,
        "db": redis_db,
        "port": redis_pord,
        "user_register_code": REDIS_USER_REGISTER_CODE,
        "user_login_code": REDIS_USER_LOGIN_CODE,
        "user_reset_code": REDIS_USER_RESET_CODE
    },
    "jwt_config": {
        "jwt_secret_key": jwt_secrect_config,
    },
    "verification_code_config": {
        "alibaba_cloud_accesskey_id": alibaba_cloud_accesskey_iD,
        "alibaba_cloud_accesskey_secret": alibaba_cloud_accesskey_secret,
        "sign_name": sign_name,
    },
    "stt_config": {
        "whisper": {
            "api_key": api_key,
            "base_url": base_url,
            "available_models": ["whisper-1"]
        },
    },
    "llm_config": {
    "endpoints": {
        "openai": {
            "api_key": api_key,
            "base_url": base_url,
            "available_models": ["gpt-3.5-turbo", "gpt-4"]
        },
        "anthropic": {
            "api_key": "your-anthropic-key",
            "base_url": "https://api.anthropic.com/v1",
            "available_models": ["claude-3-7-sonnet"]
        },
        "ollama": {
            "api_key": "",
            "base_url": "http://localhost:11434",
            "available_models": ["huihui_ai/deepseek-r1-abliterated:7b"]
        },
        "custom_endpoint": {
            "api_key": "your-custom-key",
            "base_url": "http://your-custom-endpoint",
            "available_models": ["custom-model-1"]
        }
    },
    "model_to_endpoint": {
        "gpt-3.5-turbo": "openai",
        "gpt-4": "openai",
        "claude-3-7-sonnet": "anthropic",
        "custom-model-1": "custom_endpoint"
    },
    "default_model": "gpt-3.5-turbo"
},
}

