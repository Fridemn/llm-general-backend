import logging
from typing import Optional, Dict, Type

from app import app_config, logger
from app.core.stt import STTProvider
from app.core.strategy_selector import StrategySelector
from app.core.stt.openai_strategy import ProviderOpenAISTT

class STTFactory:
    """STT 工厂类，负责创建 STT 提供者实例"""
    
    # STT 策略名称到类的映射
    PROVIDER_MAP: Dict[str, Type[STTProvider]] = {
        "openai": ProviderOpenAISTT
    }
    
    @staticmethod
    def create_provider() -> Optional[STTProvider]:
        """
        创建 STT 提供者实例
        
        Returns:
            STTProvider 实例，如果没有找到活跃策略则返回None
        """
        try:
            # 使用app_config直接获取STT配置
            # logger.info(f"使用app_config中的STT配置")
            
            # 创建选择器，传入活跃策略名称和完整配置
            selector = StrategySelector[STTProvider](app_config.stt_config["active"], app_config.stt_config)
            
            # 获取活跃策略
            active_strategy = selector.get_active_strategy()
            if not active_strategy:
                logger.error("未找到活跃的STT策略")
                return None
                
            # 创建提供者
            provider_config = {
                "id": active_strategy,
                "type": active_strategy,
            }
            
            return selector.create_provider(STTFactory.PROVIDER_MAP, provider_config)
        except Exception as e:
            logger.error(f"创建STT提供者实例失败: {str(e)}")
            # 如果配置文件有问题，尝试直接创建默认提供者
            logger.info("尝试创建默认OpenAI STT提供者")
            try:
                default_config = {
                    "id": "openai",
                    "type": "openai",
                    "model": "whisper-large-v3"
                }
                return ProviderOpenAISTT(default_config, {})
            except Exception as e2:
                logger.error(f"创建默认STT提供者失败: {str(e2)}")
                return None
