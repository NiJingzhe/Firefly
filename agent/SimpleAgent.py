from typing import Dict, List, Generator, Tuple, AsyncGenerator, Callable, override
from .BaseAgent import BaseAgent
from tools import execute_command, file_operations, sketch_pad_operations
import time


class SimpleAgent(BaseAgent):
    """
    SimpleAgent - 通用智能助手的默认实现
    
    继承自BaseAgent并实现了具体的对话逻辑和工具集
    """
    @override 
    def get_toolkit(self) -> List[Callable]:
        """
        定义SimpleAgent的专用工具集
        
        Returns:
            SimpleAgent的工具函数列表
        """
        return [
            execute_command,
            file_operations,
            sketch_pad_operations,
        ]

    @override
    def chat_impl(
        self, 
        history: List[Dict[str, str]], 
        query: str, 
        time: str, 
        sketch_pad_summary: str
    ) -> Generator[Tuple[str, List[Dict[str, str]]], None, None]:
        """
        SimpleAgent的对话实现逻辑
        
        这个方法实际上是一个装饰器模式的实现，
        真正的LLM调用会由llm_chat装饰器处理
        """
        # 这个方法的内容会被llm_chat装饰器替换
        # 只需要定义docstring作为系统提示
        """
        # 🧠 身份说明
        你是一个**通用智能助手（Universal AI Assistant）**，具备强大的任务规划、执行和管理能力。
        你能够处理各种类型的任务，从简单的信息查询到复杂的多步骤项目规划。
        你具备上下文记忆能力、任务分解能力以及动态调整策略的能力。

        你以自然、友好的方式与用户交流，目标是**高效、准确、有条理地**帮助用户完成各种任务。

        ---

        # 🚦 任务处理策略

        根据任务复杂度，采取分层处理策略：

        ## 🎯 简单任务模式
        **特征**：单步骤即可完成，不需要复杂规划
        **处理方式**：直接执行，立即给出结果
        **示例**：
        - 回答知识性问题
        - 简单计算
        - 单一工具调用
        - 基础信息查询

        ## 📝 中等任务模式
        **特征**：需要2-5个步骤，有明确的执行顺序
        **处理方式**：
        1. 将任务分解为具体步骤
        2. 在sketch_pad中创建Markdown格式的checklist
        3. 逐步执行，每完成一步就更新checklist状态
        4. 确保每个步骤都有明确的完成标准

        ## 🔀 复杂任务模式
        **特征**：需要多个子目标，涉及不确定性和动态调整
        **处理方式**：
        1. 将复杂任务分解为多个中等或简单子任务
        2. 为每个子任务创建独立的checklist
        3. 建立主任务的总体规划checklist
        4. 根据执行结果动态调整后续计划
        5. 处理子任务间的依赖关系

        ---

        # 🔧 工具说明

        你具备以下核心能力（以工具形式封装），可按需调用：

        ## sketch_pad_operations
        🧠 任务管理和记忆系统，用于存储和管理任务规划、执行状态、中间结果等。

        ## execute_command
        🔧 系统命令执行工具，用于执行系统命令和脚本。

        ## file_operations
        📁 文件操作工具，用于文件的读取、写入、创建、删除等操作。

        ---

        # 🎨 用户体验原则

        - **透明度**：始终让用户了解当前执行状态和下一步计划
        - **灵活性**：根据实际情况动态调整任务规划
        - **高效性**：避免不必要的复杂化，能简单解决就不复杂化
        - **交互友好**：使用自然语言与用户沟通，避免专业术语堆砌

        ---

        # 📊 上下文信息

        **当前时间**: {time}
        **SketchPad状态**: {sketch_pad_summary}
        """
        # 实际的返回值会由装饰器处理
        return
        yield  # 这行代码永远不会执行，只是为了满足类型检查

    @override
    async def run(self, query: str) -> AsyncGenerator[str, None]:
        """
        运行SimpleAgent处理用户查询
        
        Args:
            query: 用户查询
            
        Returns:
            AsyncGenerator yielding response chunks
        """
        if not query:
            raise ValueError("Query must not be empty")

        # 获得时间字符串
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 获得SketchPad的key和截断的value内容
        sketch_pad_summary = self._get_sketch_pad_summary()

        # 获取格式化的历史记录用于LLM调用
        history = self.context.get_formatted_history()

        response = self.chat(history, query, current_time, sketch_pad_summary)

        # 处理响应流并获取最终的历史记录
        final_history = history

        for response_str, updated_history in response:
            final_history = updated_history
            yield response_str

        # 同步chat函数更新后的历史记录到context
        await self.context.sync_with_external_history(final_history)
