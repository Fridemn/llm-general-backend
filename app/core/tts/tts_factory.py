import os
from pathlib import Path
from typing import Optional, Dict, Type
import logging

from app.core.strategy_selector import StrategySelector
from app.core.tts import TTSProvider
from app.core.tts.edge_strategy import ProviderEdgeTTS

# 配置日志
logger = logging.getLogger(__name__)

class TTSFactory:
    """TTS 工厂类，负责创建 TTS 提供者实例"""
    
    # TTS 策略名称到类的映射
    PROVIDER_MAP: Dict[str, Type[TTSProvider]] = {
        "edge": ProviderEdgeTTS
    }
    
    @staticmethod
    def create_provider(config_path: Optional[str] = None) -> Optional[TTSProvider]:
        """
        创建 TTS 提供者实例
        
        Args:
            config_path: 配置文件路径，默认为data/strategy_config.json
            
        Returns:
            TTSProvider 实例，如果没有找到活跃策略则返回None
        """
        if config_path is None:
            # 尝试确定config_path的位置
            base_dir = Path(__file__).parent.parent.parent.parent  # 项目根目录
            config_path = os.path.join(base_dir, "data", "strategy_config.json")
            
        # 创建选择器
        selector = StrategySelector[TTSProvider](config_path, "tts")
        
        # 创建提供者
        provider_config = {
            "id": selector.get_active_strategy(),
            "type": selector.get_active_strategy(),
        }
        
        return selector.create_provider(TTSFactory.PROVIDER_MAP, provider_config)
