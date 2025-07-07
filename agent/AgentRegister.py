"""
Agent注册机制
用于管理多个Agent实例，支持通过model name选择不同的Agent
"""

from typing import Dict, Optional, Type, List, Any
from .BaseAgent import BaseAgent
from config.config import get_config
import threading


class AgentRegistry:
    """Agent注册器，管理多个Agent实例"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_classes: Dict[str, Type[BaseAgent]] = {}
        self._lock = threading.Lock()
    
    def register_agent_class(self, model_name: str, agent_class: Type[BaseAgent]):
        """
        注册Agent类
        
        Args:
            model_name: 模型名称，用于API中的model参数
            agent_class: Agent类（继承自BaseAgent）
        """
        with self._lock:
            self._agent_classes[model_name] = agent_class
    
    def create_agent(
        self, 
        model_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        context_file: Optional[str] = None,
        **kwargs
    ) -> BaseAgent:
        """
        创建Agent实例
        
        Args:
            model_name: 模型名称
            name: Agent名称
            description: Agent描述
            context_file: 上下文文件路径
            **kwargs: 其他参数
            
        Returns:
            Agent实例
        """
        if model_name not in self._agent_classes:
            raise ValueError(f"Unknown model: {model_name}")
        
        agent_class = self._agent_classes[model_name]
        
        # 获取配置
        config = get_config()
        
        # 使用默认值或传入的参数
        agent_name = name or f"{model_name}-agent"
        agent_description = description or f"Agent instance for {model_name}"
        
        # 创建Agent实例
        agent = agent_class(
            name=agent_name,
            description=agent_description,
            llm_interface=config.BASIC_INTERFACE,
            context_file=context_file,
            **kwargs
        )
        
        return agent
    
    def get_or_create_agent(
        self, 
        model_name: str,
        **kwargs
    ) -> BaseAgent:
        """
        获取或创建Agent实例（单例模式）
        
        Args:
            model_name: 模型名称
            **kwargs: 创建参数
            
        Returns:
            Agent实例
        """
        with self._lock:
            if model_name not in self._agents:
                self._agents[model_name] = self.create_agent(model_name, **kwargs)
            return self._agents[model_name]
    
    def get_agent(self, model_name: str) -> Optional[BaseAgent]:
        """
        获取已创建的Agent实例
        
        Args:
            model_name: 模型名称
            
        Returns:
            Agent实例或None
        """
        return self._agents.get(model_name)
    
    def list_models(self) -> List[str]:
        """
        列出所有已注册的模型名称
        
        Returns:
            模型名称列表
        """
        return list(self._agent_classes.keys())
    
    def list_agents(self) -> List[str]:
        """
        列出所有已创建的Agent实例
        
        Returns:
            Agent实例的模型名称列表
        """
        return list(self._agents.keys())
    
    def clear_agents(self):
        """清空所有Agent实例"""
        with self._lock:
            self._agents.clear()
    
    def remove_agent(self, model_name: str) -> bool:
        """
        移除Agent实例
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if model_name in self._agents:
                del self._agents[model_name]
                return True
            return False
    
    def get_agent_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取Agent信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Agent信息字典或None
        """
        agent = self.get_agent(model_name)
        if agent:
            return {
                "model_name": model_name,
                "name": agent.name,
                "description": agent.description,
                "toolkit_size": len(agent.toolkit),
                "session_info": agent.get_session_info()
            }
        return None


# 全局Agent注册器实例
_global_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """获取全局Agent注册器"""
    return _global_registry


def register_agent(model_name: str, agent_class: Type[BaseAgent]):
    """
    注册Agent类的便捷函数
    
    Args:
        model_name: 模型名称
        agent_class: Agent类
    """
    _global_registry.register_agent_class(model_name, agent_class)


def get_agent(model_name: str, **kwargs) -> BaseAgent:
    """
    获取或创建Agent实例的便捷函数
    
    Args:
        model_name: 模型名称
        **kwargs: 创建参数
        
    Returns:
        Agent实例
    """
    return _global_registry.get_or_create_agent(model_name, **kwargs)


def list_available_models() -> List[str]:
    """
    列出所有可用模型的便捷函数
    
    Returns:
        模型名称列表
    """
    return _global_registry.list_models()
