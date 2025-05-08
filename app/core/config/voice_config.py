"""
配置管理模块，用于加载和管理语音服务相关的配置
"""

import os
import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VoiceServiceConfig:
    """语音服务配置管理类"""

    _instance = None
    _config_data = {}

    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(VoiceServiceConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """加载配置文件"""
        try:
            # 配置文件路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_file = os.path.join(base_dir, "config", "voice_service_config.json")

            # 检查配置文件是否存在
            if not os.path.exists(config_file):
                logger.warning(f"配置文件不存在: {config_file}，将使用默认配置")
                self._config_data = self._get_default_config()
                return

            # 加载配置文件
            with open(config_file, "r", encoding="utf-8") as f:
                self._config_data = json.load(f)
                logger.info(f"成功加载配置文件: {config_file}")

        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}，将使用默认配置")
            self._config_data = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "asr_service": {
                "server_url": "ws://127.0.0.1:8765",
                "api_key": "dd91538f5918826f2bdf881e88fe9956",
                "timeout": 10.0,
            },
            "tts_service": {"api_base": "http://127.0.0.1:5000", "output_dir": "static/audio"},
            "audio_config": {"rate": 16000, "channels": 1, "chunk": 1600, "format": "paInt16"},
            "vad_config": {
                "window": 30,
                "threshold": 0.3,
                "min_speech_frames": 5,
                "max_silence_frames": 15,
                "audio_rms_threshold": 200,
            },
            "websocket": {"heartbeat_interval": 15, "connection_timeout": 60},
        }

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置项

        Args:
            section: 配置节名称
            key: 配置项名称
            default: 默认值

        Returns:
            配置值或默认值
        """
        if section in self._config_data and key in self._config_data[section]:
            return self._config_data[section][key]
        return default

    def get_section(self, section: str) -> Optional[Dict[str, Any]]:
        """获取整个配置节

        Args:
            section: 配置节名称

        Returns:
            配置节数据或None
        """
        return self._config_data.get(section)


# 全局配置实例
voice_config = VoiceServiceConfig()


# 导出便捷函数
def get_voice_config(section: str, key: str, default: Any = None) -> Any:
    """获取语音服务配置项"""
    return voice_config.get(section, key, default)


def get_voice_config_section(section: str) -> Optional[Dict[str, Any]]:
    """获取语音服务配置节"""
    return voice_config.get_section(section)
