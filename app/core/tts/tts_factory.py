import logging
from typing import Optional, Dict, Type

from app import app_config, logger
from app.core.tts import TTSProvider
from app.core.tts.edge_strategy import ProviderEdgeTTS
from app.core.strategy_selector import StrategySelector

class TTSFactory:
    """TTS 工厂类，负责创建 TTS 提供者实例"""
    
    # TTS 策略名称到类的映射
    PROVIDER_MAP: Dict[str, Type[TTSProvider]] = {
        "edge": ProviderEdgeTTS
    }
    
    @staticmethod
    def create_provider() -> Optional[TTSProvider]:
        """
        创建 TTS 提供者实例
        
        Returns:
            TTSProvider 实例，如果没有找到活跃策略则返回None
        """
        try:
            # 使用app_config直接获取TTS配置
            # logger.info(f"使用app_config中的TTS配置")
            
            # 创建选择器，传入活跃策略名称和完整配置
            selector = StrategySelector[TTSProvider](app_config.tts_config["active"], app_config.tts_config)
            
            # 获取活跃策略
            active_strategy = selector.get_active_strategy()
            if not active_strategy:
                logger.error("未找到活跃的TTS策略")
                return None
                
            # 创建提供者
            provider_config = {
                "id": active_strategy,
                "type": active_strategy,
            }
            
            return selector.create_provider(TTSFactory.PROVIDER_MAP, provider_config)
        except Exception as e:
            logger.error(f"创建TTS提供者实例失败: {str(e)}")
            # 如果配置文件有问题，尝试直接创建默认提供者
            logger.info("尝试创建默认Edge TTS提供者")
            try:
                default_config = {
                    "id": "edge",
                    "type": "edge",
                    "edge-tts-voice": "zh-CN-XiaoxiaoNeural"
                }
                return ProviderEdgeTTS(default_config, {})
            except Exception as e2:
                logger.error(f"创建默认TTS提供者失败: {str(e2)}")
                return None
