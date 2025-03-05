import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type, TypeVar, Generic

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')

class StrategySelector(Generic[T]):
    """
    通用策略选择器基类，用于从配置中选择和创建相应的策略实例
    """
    
    def __init__(self, config_path: str, provider_type: str):
        """
        初始化策略选择器
        
        Args:
            config_path: 配置文件路径
            provider_type: 提供者类型名称 (如 'tts', 'stt')
        """
        self.config_path = config_path
        self.provider_type = provider_type
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载策略配置
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}")
                return {}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 确保配置包含provider_type部分
            if self.provider_type not in config:
                logger.warning(f"配置中不包含 {self.provider_type} 部分")
                return {}
                
            return config[self.provider_type]
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return {}
            
    def get_active_strategy(self) -> Optional[str]:
        """
        获取活跃的策略名称
        
        Returns:
            活跃的策略名称，如果没有找到则返回None
        """
        for strategy_name, is_active in self.config.items():
            if is_active:
                return strategy_name
        return None

    def create_provider(self, provider_class_map: Dict[str, Type[T]], provider_config: Dict[str, Any] = None) -> Optional[T]:
        """
        创建提供者实例
        
        Args:
            provider_class_map: 策略名称到提供者类的映射
            provider_config: 提供者配置
            
        Returns:
            提供者实例，如果没有找到活跃策略则返回None
        """
        active_strategy = self.get_active_strategy()
        if not active_strategy:
            logger.warning(f"未找到活跃的{self.provider_type}策略")
            return None
            
        if active_strategy not in provider_class_map:
            logger.warning(f"未找到{active_strategy}的实现类")
            return None
            
        try:
            provider_class = provider_class_map[active_strategy]
            config = provider_config or {}
            return provider_class(config, {})
        except Exception as e:
            logger.error(f"创建{active_strategy}提供者实例时出错: {str(e)}")
            return None
