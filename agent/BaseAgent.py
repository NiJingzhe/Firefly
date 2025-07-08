from abc import ABC, abstractmethod
from SimpleLLMFunc import llm_chat, OpenAICompatible, Tool
from typing import (
    Dict,
    List,
    Optional,
    Callable,
    Generator,
    Tuple,
    Sequence,
    AsyncGenerator,
)
import threading
from context.context import ConversationContext, ensure_global_context
from context.sketch_pad import SmartSketchPad, get_global_sketch_pad
import threading


class BaseAgent(ABC):
    """
    Agent基类，定义了Agent的基本接口和通用功能
    
    所有具体的Agent实现都应该继承此类并实现抽象方法
    """
    
    # 类级别的实例缓存，确保每个Agent子类的单例
    _class_instances: Dict[str, 'BaseAgent'] = {}
    _class_lock = threading.Lock()

    @classmethod
    def get_instance(
        cls, 
        model_name: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        llm_interface: Optional[OpenAICompatible] = None,
        **kwargs
    ) -> 'BaseAgent':
        """
        获取Agent实例的类方法（单例模式）
        
        Args:
            model_name: 模型名称
            name: Agent名称
            description: Agent描述  
            llm_interface: LLM接口
            **kwargs: 其他参数
            
        Returns:
            Agent实例
        """
        with cls._class_lock:
            # 使用类名和model_name作为唯一标识
            instance_key = f"{cls.__name__}:{model_name}"
            
            if instance_key not in cls._class_instances:
                if not llm_interface:
                    # 如果没有提供llm_interface，尝试从配置获取
                    from config.config import get_config
                    config = get_config()
                    llm_interface = config.BASIC_INTERFACE
                
                instance_name = name or f"{model_name}-agent"
                instance_description = description or f"Agent instance for {model_name}"
                
                cls._class_instances[instance_key] = cls(
                    name=instance_name,
                    description=instance_description,
                    llm_interface=llm_interface,
                    model_name=model_name,
                    **kwargs
                )
            
            return cls._class_instances[instance_key]
    
    @classmethod
    def clear_instances(cls):
        """清空所有实例缓存"""
        with cls._class_lock:
            cls._class_instances.clear()
    
    @classmethod
    def get_all_instances(cls) -> Dict[str, 'BaseAgent']:
        """获取所有实例"""
        return cls._class_instances.copy()

    def __init__(
        self,
        name: str,
        description: str,
        llm_interface: Optional[OpenAICompatible] = None,
        max_history_length: int = 5,
        save_context: bool = True,
        context_file: Optional[str] = None,
        model_name: Optional[str] = None,  # 添加model_name参数
        **kwargs  # 额外的参数，子类可以处理
    ):
        self.name = name
        self.description = description
        self.model_name = model_name  # 存储model_name
        self.llm_interface = llm_interface

        if not self.llm_interface:
            raise ValueError("llm_interface must be provided")

        # 子类需要定义自己的工具集
        self.toolkit = self.get_toolkit()

        # 使用全局的ConversationContext实例，确保对话历史的持久性
        # 如果指定了context_file，则使用ensure_global_context来确保使用正确的实例
        if context_file:
            self.context = ensure_global_context(
                llm_interface=self.llm_interface,
                max_history_length=max_history_length,
                save_to_file=save_context,
                context_file=context_file,
            )
        else:
            # 使用默认的全局上下文
            self.context = ensure_global_context(
                llm_interface=self.llm_interface,
                max_history_length=max_history_length,
                save_to_file=save_context,
            )

        # 使用全局的SketchPad实例，确保所有Agent共享同一个存储
        # 这样可以在不同Agent会话之间保持数据的连续性
        self.sketch_pad = get_global_sketch_pad()

        # 初始化chat函数
        self.chat = llm_chat(
            llm_interface=self.llm_interface,
            toolkit=self.toolkit,  # type: ignore[call-arg]
            stream=True,
            max_tool_calls=2000,
            timeout=600,
        )(self.chat_impl)

    @abstractmethod
    def get_toolkit(self) -> List[Callable]:
        """
        获取Agent专用的工具集（抽象方法）
        
        子类必须实现此方法来定义自己的工具集
        
        Returns:
            工具函数列表
        """
        pass

    @abstractmethod
    def chat_impl(
        self, 
        history: List[Dict[str, str]], 
        query: str, 
        time: str, 
        sketch_pad_summary: str
    ) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
        """
        Agent的对话实现逻辑（抽象方法）
        
        子类必须实现此方法来定义具体的对话行为
        
        Args:
            history: 对话历史
            query: 用户查询
            time: 当前时间
            sketch_pad_summary: SketchPad摘要
            
        Returns:
            Generator yielding (response_chunk, updated_history)
        """
        pass

    @abstractmethod
    def run(self, query: str) -> AsyncGenerator[str, None]:
        """
        运行Agent处理用户查询（抽象方法）
        
        Args:
            query: 用户查询
            
        Returns:
            AsyncGenerator yielding response chunks
        """
        pass

    # 通用辅助方法
    def _get_sketch_pad_summary(self) -> str:
        """获取SketchPad的摘要信息，包括所有keys和截断的values"""
        try:
            # 获取所有项目的详细信息
            all_items = self.sketch_pad.list_all(include_details=True)

            if not all_items:
                return "SketchPad为空：无存储内容"

            summary_lines = [f"SketchPad当前状态 (共{len(all_items)}个项目):"]

            for item in all_items[:20]:  # 限制显示前20个项目
                key = item["key"]
                tags = ", ".join(item["tags"]) if item["tags"] else "无标签"
                timestamp = item["timestamp"]
                content_type = item["content_type"]

                # 获取完整内容并截断
                full_item = self.sketch_pad.get_item(key)
                if full_item:
                    value_str = str(full_item.value)
                    # 截断内容到合理长度
                    if len(value_str) > 100:
                        value_preview = value_str[:100] + "..."
                    else:
                        value_preview = value_str

                    # 处理换行符
                    value_preview = value_preview.replace("\n", "\\n")

                    summary_lines.append(
                        f"  • {key}: [{content_type}] {value_preview} "
                        f"(标签: {tags}, 时间: {timestamp[:19]})"
                    )
                else:
                    summary_lines.append(f"  • {key}: [已删除或无法访问]")

            if len(all_items) > 20:
                summary_lines.append(f"  ... 还有 {len(all_items) - 20} 个项目未显示")

            return "\n".join(summary_lines)

        except Exception as e:
            return f"获取SketchPad摘要时出错: {str(e)}"

    # 上下文管理的便捷方法
    def get_conversation_history(self, limit: Optional[int] = None):
        """获取当前会话的对话历史"""
        return self.context.get_history(limit)

    def get_full_saved_history(self, limit: Optional[int] = None):
        """获取完整保存的对话历史"""
        return self.context.get_full_saved_history(limit)

    def search_conversation(self, query: str, limit: int = 5):
        """搜索当前会话的对话历史"""
        return self.context.search_history(query, limit)

    def search_full_history(self, query: str, limit: int = 5):
        """搜索完整保存的对话历史"""
        return self.context.search_full_history(query, limit)

    def clear_conversation(self, keep_summary: bool = True):
        """清空当前会话的对话历史"""
        self.context.clear_history(keep_summary)

    def get_conversation_summary(self):
        """获取当前会话的对话摘要"""
        return self.context.get_context_summary()

    def get_full_saved_summary(self):
        """获取完整保存的对话摘要"""
        return self.context.get_full_saved_summary()

    def export_conversation(self, file_path: str):
        """导出当前会话的对话记录"""
        self.context.export_context(file_path)

    def import_conversation(self, file_path: str, merge: bool = False):
        """导入对话记录"""
        self.context.import_context(file_path, merge)

    # SketchPad 管理的便捷方法
    async def store_in_sketch_pad(
        self,
        value,
        key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ):
        """存储数据到 SketchPad"""
        return await self.sketch_pad.store(value, key, ttl=ttl, tags=tags)

    def get_from_sketch_pad(self, key: str):
        """从 SketchPad 获取数据"""
        return self.sketch_pad.retrieve(key)

    def search_sketch_pad(self, query: str, limit: int = 5):
        """搜索 SketchPad 内容"""
        return self.sketch_pad.search(query, limit)

    def get_sketch_pad_stats(self):
        """获取 SketchPad 统计信息"""
        return self.sketch_pad.get_statistics()

    def clear_sketch_pad(self):
        """清空 SketchPad"""
        self.sketch_pad.clear_all()

    def get_session_info(self):
        """获取会话信息（包括对话历史和 SketchPad 统计）"""
        return {
            "agent_name": self.name,
            "model_name": self.model_name,
            "agent_class": self.__class__.__name__,
            "conversation_count": len(self.get_conversation_history()),
            "sketch_pad_stats": self.get_sketch_pad_stats(),
            "conversation_summary": self.get_conversation_summary(),
        }