import os
import json
import logging
from typing import Dict, Any, Optional, TypeVar, Generic, Type

# 配置日志
logger = logging.getLogger(__name__)

# 定义泛型类型变量
T = TypeVar('T')

class StrategySelector(Generic[T]):
    """
    策略选择器，用于从配置文件中加载和选择策略
    
    泛型参数 T 代表策略提供者的类型
    """
    
    def __init__(self, config_path: str, config_type: str):
        """
        初始化策略选择器
        
        Args:
            config_path: 配置文件路径
            config_type: 配置类型，如 'tts', 'stt'
        """
        self.config_path = config_path
        self.config_type = config_type
        self.config = self._load_config()
        logger.info(f"已加载{self.config_type}配置: {self.config}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        从配置文件加载配置
        
        Returns:
            配置字典
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}, 使用默认配置")
                return self._get_default_config()
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 检查配置是否包含指定类型
            if self.config_type not in config:
                logger.warning(f"配置文件中不存在{self.config_type}配置, 使用默认配置")
                return self._get_default_config()
                
            return config[self.config_type]
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}, 使用默认配置")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        if self.config_type == "tts":
            return {
                "active": "edge",
                "edge": {
                    "type": "edge",
                    "edge-tts-voice": "zh-CN-XiaoxiaoNeural"
                }
            }
        elif self.config_type == "stt":
            return {
                "active": "openai",
                "openai": {
                    "type": "openai",
                    "model": "whisper-1"
                }
            }
        else:
            return {"active": None}
            
    def get_active_strategy(self) -> Optional[str]:
        """
        获取活跃策略名称
        
        Returns:
            活跃策略名称，如果不存在则返回None
        """
        active = self.config.get("active")
        logger.info(f"{self.config_type}活跃策略: {active}")
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
            logger.error(f"未指定{self.config_type}策略名称")
            return {}
            
        if name not in self.config:
            logger.error(f"{self.config_type}配置中未找到策略: {name}")
            return {}
            
        strategy_config = self.config.get(name, {})
        logger.info(f"{self.config_type}策略{name}配置: {strategy_config}")
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
            logger.info(f"创建{self.config_type}提供者: {strategy_name}, 配置: {merged_config}")
            provider = provider_class(merged_config, {})
            return provider
            
        except Exception as e:
            logger.error(f"创建{self.config_type}提供者失败: {str(e)}")
            return None
