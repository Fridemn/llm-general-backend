import os
import json
from typing import Dict, Any, Optional, TypeVar, Generic, Type

from app import logger


# 定义泛型类型变量
T = TypeVar('T')

class StrategySelector(Generic[T]):
    """
    策略选择器，用于从配置文件中加载和选择策略
    
    泛型参数 T 代表策略提供者的类型
    """
    
    def __init__(self, config_path_or_active: str, config_type_or_config: str = None):
        """
        初始化策略选择器
        
        Args:
            config_path_or_active: 配置文件路径或活跃策略名称
            config_type_or_config: 配置类型(如 'tts', 'stt')或配置对象
        """
        # 根据第二个参数类型判断初始化方式
        if isinstance(config_type_or_config, dict):
            # 直接使用配置对象
            self.config = config_type_or_config
            self.config_type = "direct"
            # logger.info(f"使用直接配置初始化策略选择器，活跃策略: {config_path_or_active}")
        else:
            # 从文件加载配置
            self.config_path = config_path_or_active
            self.config_type = config_type_or_config
            self.config = self._load_config()
            # logger.info(f"已加载{self.config_type}配置: {self.config}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        从配置文件加载配置
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                # logger.error(f"配置文件不存在: {self.config_path}")
                return {}
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 检查配置是否包含指定类型
            if self.config_type not in config:
                # logger.error(f"配置文件中不存在{self.config_type}配置")
                return {}
                
            return config[self.config_type]
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return {}
            
    def get_active_strategy(self) -> Optional[str]:
        """
        获取活跃策略名称
        
        Returns:
            活跃策略名称，如果不存在则返回None
        """
        active = self.config.get("active")
        # logger.info(f"{self.config_type}活跃策略: {active}")
        return active
        
    def get_strategy_config(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取策略配置
        
        Args:
            strategy_name: 策略名称，如果为None则使用活跃策略
            
        Returns:
            策略配置字典
        """
        name = strategy_name or self.get_active_strategy()
        if not name:
            # logger.error(f"未指定{self.config_type}策略名称")
            return {}
            
        if name not in self.config:
            # logger.error(f"{self.config_type}配置中未找到策略: {name}")
            return {}
            
        strategy_config = self.config.get(name, {})
        # logger.info(f"{self.config_type}策略{name}配置: {strategy_config}")
        return strategy_config
        
    def create_provider(self, provider_map: Dict[str, Type[T]], provider_config: Dict[str, Any]) -> Optional[T]:
        """
        创建策略提供者实例
        
        Args:
            provider_map: 策略名称到类的映射
            provider_config: 提供者配置
            
        Returns:
            策略提供者实例，如果创建失败则返回None
        """
        try:
            # 获取活跃策略名称
            strategy_name = provider_config.get("type") or self.get_active_strategy()
            if not strategy_name:
                logger.error(f"未指定{self.config_type}策略名称，无法创建提供者")
                return None
                
            # 获取策略类
            provider_class = provider_map.get(strategy_name)
            if not provider_class:
                logger.error(f"未找到{self.config_type}策略: {strategy_name}")
                return None
                
            # 获取策略配置
            strategy_config = self.get_strategy_config(strategy_name)
            
            # 合并配置
            merged_config = {**strategy_config, **provider_config}
            merged_config["type"] = strategy_name  # 确保type字段存在
            
            # 创建提供者实例
            # logger.info(f"创建{self.config_type}提供者: {strategy_name}, 配置: {merged_config}")
            provider = provider_class(merged_config, {})
            return provider
            
        except Exception as e:
            logger.error(f"创建{self.config_type}提供者失败: {str(e)}")
            return None

