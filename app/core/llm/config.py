import json
import os
from typing import Dict, Any

# 读取配置文件
def load_config() -> Dict[str, Any]:
    config_path = os.path.join('data', 'cmd_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 使用 utf-8-sig 编码来处理带有 BOM 的 UTF-8 文件
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)
    
    return config

# 获取LLM配置
def get_llm_config() -> Dict[str, Any]:
    config = load_config()
    if 'llm_config' not in config:
        raise ValueError("配置文件中找不到LLM配置")
    
    return config['llm_config']

# 导出LLM配置
llm_config = get_llm_config()['endpoints']
MODEL_TO_ENDPOINT = get_llm_config()['model_to_endpoint']
DEFAULT_MODEL = get_llm_config().get('default_model', 'gpt-3.5-turbo')
