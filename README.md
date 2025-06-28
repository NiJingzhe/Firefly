![repocover](https://github.com/NiJingzhe/SimpleAgent/blob/dev/img/repocover.png?raw=true)
# SimpleAgent

![Github Stars](https://img.shields.io/github/stars/NiJingzhe/SimpleAgent.svg?style=social)
![Github Forks](https://img.shields.io/github/forks/NiJingzhe/SimpleAgent.svg?style=social)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/NiJingzhe/SimpleAgent/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NiJingzhe/SimpleAgent/pulls)


一个基于 SimpleLLMFunc 构建的通用智能体框架，专注于任务管理、记忆系统和工具集成。

## 🎯 项目简介

SimpleAgent 是一个通用的智能体框架，设计理念是提供一个强大且灵活的基础平台，让开发者能够快速构建专业领域的智能助手。框架核心特色是其任务管理能力和智能记忆系统。

**框架优势**：
- **任务智能分解**: 自动识别任务复杂度，采用分层处理策略（简单/中等/复杂任务模式）
- **智能记忆系统**: SketchPad提供LRU缓存、标签管理和AI摘要功能
- **灵活工具集成**: 通过简单修改Prompt和工具，适配任何专业领域
- **完整会话管理**: 自动保存对话历史，支持摘要和检索

**当前默认配置**为通用助手模式，可处理：
- 文件操作和管理
- 系统命令执行
- 数据存储和检索
- 任务规划和跟踪
- 会话历史管理

## ✨ 核心特性

### 🧠 智能任务管理
- **任务复杂度识别**: 自动判断简单/中等/复杂任务，采用不同处理策略
- **Checklist系统**: 中等和复杂任务自动生成Markdown格式的执行清单
- **动态任务调整**: 根据执行结果实时调整后续计划
- **依赖关系管理**: 处理子任务间的依赖关系和执行顺序

### 🤖 智能对话系统
- **多模型支持**: 集成 GPT-4o、Claude、Gemini、DeepSeek 等多种 LLM
- **流式对话**: 实时响应，支持打字机效果的流式输出
- **单例上下文**: 全局统一的对话上下文管理
- **会话持久化**: 自动保存对话历史到文件，支持会话恢复

### 📒 SketchPad 智能存储系统
- **LRU缓存机制**: 基于访问频率的智能内存管理
- **多维标签系统**: 支持标签分类、搜索和管理
- **AI自动摘要**: 智能生成内容摘要，便于快速定位
- **持久化存储**: 数据持久化到文件，支持会话间数据共享
- **内容类型识别**: 自动识别和分类不同类型的内容

### 🛠️ 核心工具集
- **文件操作 (file_operations)**: 支持读取、修改、插入、追加、覆写等精细化文件操作
- **命令执行 (execute_command)**: 安全的系统命令执行，自动记录执行历史
- **SketchPad操作 (sketch_pad_operations)**: 数据存储、检索、搜索、统计等功能
- **丰富的特殊命令**: 支持会话管理、历史查询、数据导出等操作

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 支持的操作系统: macOS, Linux, Windows

### 安装依赖

# 使用 uv 安装依赖（推荐）
```bash
uv sync
```
# 或使用 pip
```bash
pip install -r requirements.txt
```

### 配置设置
  
1. 复制配置模板：
```bash
cp config/provider_template.json config/provider.json
```

2. 编辑 `config/provider.json`，填入您的 API 密钥：
```json
{
  "chatanywhere": {
    "gpt-4o": {
      "api_key": "your-api-key-here",
      "base_url": "https://api.chatanywhere.tech/v1"
    },
    "claude-sonnet-4-20250514": {
      "api_key": "your-claude-key-here",
      "base_url": "https://api.chatanywhere.tech/v1"
    }
  }
}
```

### 启动应用

#### 终端交互模式
```bash
python main.py
```

#### Web API 服务模式
```bash
# 基本启动
python start_web_server.py

# 自定义主机和端口
python start_web_server.py --host 0.0.0.0 --port 8080

# 开发模式（自动重载）
python start_web_server.py --reload
```

访问 `http://localhost:8000/docs` 查看 API 文档。

## 🔄 框架定制化

### 智能任务处理模式

SimpleAgent 具备智能任务识别和分层处理能力，自动根据任务复杂度选择最佳处理策略：

#### 🎯 简单任务模式
- **特征**: 单步骤即可完成，不需要复杂规划
- **处理方式**: 直接执行，立即给出结果
- **示例**: 回答知识性问题、简单计算、单一工具调用

#### 📋 中等任务模式  
- **特征**: 需要2-5个步骤，有明确的执行顺序
- **处理方式**: 创建Markdown格式的checklist，逐步执行并更新状态
- **示例**: 文件批量处理、多步数据分析、项目配置

#### 🔀 复杂任务模式
- **特征**: 多个子目标，涉及不确定性和动态调整
- **处理方式**: 分解为子任务，建立依赖关系，动态调整计划
- **示例**: 项目开发、系统集成、复杂数据处理流水线

### 快速适配其他领域

通过修改Agent配置，可以快速适配到不同专业领域：

#### 1. 修改 Agent Prompt
编辑 `agent/BaseAgent.py` 中的 `chat_impl` 函数：

```python
def chat_impl(history, query, time, sketch_pad_summary):
    """
    # 🎯 身份说明
    你是专业的[领域名称]智能助手，精通[核心技能1]、[核心技能2]、[核心技能3]。
    使用中文与用户交流，提供从[起始阶段]到[结束阶段]的全流程支持。
    
    # 🚦 策略说明  
    根据用户意图选择合适策略：
    [定义你的工作流程和策略]
    """
```

#### 2. 扩展专业工具集
在 `agent/global_agent.py` 中的工具集列表中添加专业工具：

```python
toolkit = [
    # 保留通用工具
    execute_command,
    file_operations,
    sketch_pad_operations,
    # 添加专业工具
    your_domain_tool_1,
    your_domain_tool_2,
    your_domain_tool_3,
]
```

#### 3. 开发专业工具
参考 `tools/` 目录下的现有工具，创建你的专业工具模块：

```python
from SimpleLLMFunc import tool

@tool(name="your_domain_tool", description="专业工具描述")
def your_domain_tool(param1: str, param2: int) -> str:
    """你的专业工具实现"""
    pass
```

### 领域适配示例

#### 数据分析助手
- **工具集**: pandas操作、可视化生成、统计分析、模型训练
- **工作流**: 数据探索 → 清洗处理 → 分析建模 → 结果可视化

#### 代码开发助手  
- **工具集**: 代码生成、测试编写、文档生成、代码审查
- **工作流**: 需求分析 → 架构设计 → 代码实现 → 测试验证

#### 文档写作助手
- **工具集**: 内容研究、结构规划、文档生成、格式美化
- **工作流**: 主题确定 → 大纲设计 → 内容创作 → 审校发布

## 💡 使用指南

### 基本对话

启动后，您可以直接与智能体对话，系统会自动识别任务复杂度并采用合适的处理策略：

```
>>> 帮我分析这个文本文件的内容
>>> 创建一个项目文件结构并生成配置文件
>>> 执行一系列数据处理任务
```

### 特殊命令系统

系统提供丰富的特殊命令来管理会话和数据：

#### 会话管理命令
- `/help` - 显示完整帮助信息
- `/history` - 查看当前会话历史（最近5条）
- `/full_history` - 查看完整保存历史（最近10条）
- `/clear` - 清空当前会话历史（保留完整历史文件）
- `/summary` - 显示当前会话摘要
- `/full_summary` - 显示完整保存摘要
- `/export <filename>` - 导出会话记录到文件
- `/session` - 显示会话统计信息

#### 历史搜索命令
- `/search <query>` - 搜索当前会话历史
- `/search_all <query>` - 搜索完整保存历史

#### SketchPad 管理命令
- `/pad` - 显示 SketchPad 内容概览
- `/pad_stats` - 显示详细统计信息
- `/pad_search <query>` - 内容搜索
- `/pad_store <key> <value>` - 手动存储内容
- `/pad_get <key>` - 获取指定内容
- `/pad_delete <key>` - 删除指定项目
- `/pad_update <key> <new_value>` - 更新现有项目
- `/pad_tag <key> <tag1,tag2>` - 为项目添加标签
- `/pad_clear` - 清空所有 SketchPad 数据

### 智能工作流示例

#### 简单任务：文件操作
```
>>> 读取config.json文件的内容
系统会：
1. 直接使用file_operations工具读取文件
2. 自动存储到SketchPad以便后续使用
3. 返回内容和SketchPad key
```

#### 中等任务：项目初始化
```
>>> 创建一个Python项目的基础结构
系统会：
1. 在SketchPad中创建任务checklist
2. 逐步创建目录结构
3. 生成配置文件
4. 更新checklist状态
5. 确认所有步骤完成
```

#### 复杂任务：数据处理流水线
```
>>> 处理多个数据文件，进行清洗、分析和报告生成
系统会：
1. 分解为多个子任务
2. 建立依赖关系图
3. 为每个子任务创建独立checklist
4. 动态调整执行计划
5. 生成最终报告
```

## 🏗️ 项目结构

```
SimpleAgent_General/
├── main.py                    # 终端交互入口
├── start_web_server.py        # Web服务器启动脚本
├── pyproject.toml            # 项目配置和依赖
├── agent/                    # 智能体核心模块
│   ├── BaseAgent.py          # 基础智能体类
│   ├── global_agent.py       # 全局单例管理器
│   └── __init__.py           # 模块初始化
├── config/                   # 配置管理
│   ├── config.py             # 配置加载器
│   ├── provider.json         # API 配置文件
│   └── provider_template.json # 配置模板
├── context/                  # 上下文管理
│   ├── context.py            # 对话上下文管理
│   └── sketch_pad.py         # 智能存储系统
├── tools/                    # 工具集合
│   ├── __init__.py           # 工具模块入口
│   ├── common.py             # 公共工具函数
│   ├── command_tools.py      # 命令执行工具
│   ├── file_tools.py         # 文件操作工具
│   └── sketch_tools.py       # SketchPad 工具
├── web_interface/            # Web API 模块
│   ├── __init__.py           # 模块初始化
│   ├── server.py             # FastAPI 服务器
│   └── models.py             # OpenAI 兼容数据模型
├── history/                  # 对话历史存储
│   └── conversation_history.json
└── sandbox/                  # 工作沙盒目录
```

## 🔧 核心模块详解

### BaseAgent 类（智能体核心）
**位置**: `agent/BaseAgent.py`

智能体的核心实现，提供：
- **智能任务处理**: 自动识别任务复杂度，采用分层处理策略
- **LLM 接口管理**: 支持多种大语言模型的统一调用
- **工具调用框架**: 动态工具注册和智能调用机制  
- **流式对话控制**: 实时流式输出和上下文管理
- **SketchPad 集成**: 无缝集成智能数据存储和检索
- **会话管理**: 完整的对话历史管理和摘要功能

**核心方法**：
- `chat_impl()`: 智能任务处理的核心逻辑
- `run()`: 异步执行用户查询
- `get_conversation_history()`: 获取对话历史
- `store_in_sketch_pad()`: 存储数据到SketchPad

### ConversationContext 类（上下文管理）
**位置**: `context/context.py`

单例模式的对话上下文管理器，提供：
- **历史记录管理**: 自动存储和检索对话历史
- **智能摘要**: AI驱动的长期记忆和摘要功能
- **会话元数据**: 丰富的会话统计和管理信息
- **持久化存储**: 自动保存对话到JSON文件
- **搜索功能**: 支持历史记录的语义搜索

**核心功能**：
- 自动历史长度管理（超长时触发摘要）
- 会话间数据持久化
- 完整历史记录保存
- 灵活的导入导出功能

### SketchPad 系统（智能存储）
**位置**: `context/sketch_pad.py`

通用智能存储系统，特点：
- **LRU 缓存策略**: 基于访问频率的智能内存管理
- **多维标签系统**: 支持标签分类、搜索和管理
- **AI 自动摘要**: 智能生成内容摘要，便于快速定位
- **内容类型识别**: 自动识别文本、JSON、代码等类型
- **访问统计**: 记录访问次数和最后访问时间
- **TTL 过期机制**: 支持自动过期清理

**数据结构**：
```python
@dataclass
class SketchPadItem:
    value: Any                    # 存储的值
    timestamp: datetime          # 创建时间
    summary: Optional[str]       # AI生成的摘要
    expires_at: Optional[datetime] # 过期时间
    access_count: int           # 访问次数
    tags: Set[str]              # 标签集合
    content_type: str           # 内容类型
```

### 核心工具集
**位置**: `tools/` 目录

#### 1. file_operations（文件操作工具）
- **精细化文件操作**: 支持按行读取、修改、插入、追加、覆写
- **SketchPad集成**: 自动存储读取结果到SketchPad
- **智能路径管理**: 自动创建目录结构
- **编码支持**: 统一UTF-8编码处理

#### 2. execute_command（命令执行工具）
- **安全命令执行**: 35秒超时保护，标准输出捕获
- **执行历史记录**: 自动记录命令、返回码、执行时间
- **SketchPad集成**: 自动存储执行记录和输出结果
- **错误处理**: 完善的异常处理和错误记录

#### 3. sketch_pad_operations（SketchPad工具）
- **数据管理**: 存储、检索、删除、清空操作
- **搜索功能**: 内容搜索和标签搜索
- **统计信息**: 详细的使用统计和性能指标
- **批量操作**: 支持批量数据管理

## 📋 依赖项说明

### 框架核心依赖
- **SimpleLLMFunc (0.2.8)**: LLM 接口和工具调用框架
- **FastAPI (>=0.115.14)**: 现代化的Web API框架
- **Uvicorn (>=0.34.3)**: ASGI服务器，用于运行FastAPI
- **Pydantic (>=2.5.0)**: 数据验证和设置管理

### 开发和用户体验
- **Rich**: 美化控制台输出和交互界面
- **Asyncio**: 异步编程支持
- **Threading**: 多线程安全的单例模式
- **JSON**: 数据序列化和持久化
- **Hashlib**: 内容哈希计算
- **UUID**: 唯一标识符生成

### 支持的LLM模型
- **所有支持OpenAI API调用的模型接口**
- 你可以在 `config/provider.json` 中配置不同的模型, API Key和流量控制参数，参考 `config/provider_template.json` 模板。

> tips: 流量控制通过令牌桶算法实现，那你就明白配置中capacity和refill rate的含义了。

### 系统要求
- **Python**: 3.11w (使用最新的异步特性)

## 🎮 示例用法

### 通用任务处理示例

#### 文件处理任务
```
>>> 读取config.json文件，分析其结构，然后创建一个备份文件

系统处理流程：
1. 🎯 识别为中等任务（多步骤）
2. 📋 创建执行checklist并存储到SketchPad
3. 🔧 使用file_operations读取文件
4. 💾 自动存储文件内容到SketchPad
5. 📊 分析JSON结构
6. 📁 创建备份文件
7. ✅ 更新checklist状态，确认完成
```

#### 系统维护任务
```
>>> 检查系统状态，清理临时文件，生成状态报告

系统处理流程：
1. 🔀 识别为复杂任务（多子目标）
2. 📋 分解为子任务：状态检查、文件清理、报告生成
3. ⚡ 使用execute_command执行系统命令
4. 💾 存储各步骤输出到SketchPad
5. 📊 汇总分析结果
6. 📄 生成综合报告
7. 🔄 动态调整执行计划
```

#### 数据管理任务
```
>>> 整理我的工作文件，按类型分类并创建索引

系统处理流程：
1. 🔀 复杂任务分解
2. 📂 扫描文件目录
3. 🏷️ 按扩展名和内容分类
4. 📁 创建分类目录结构
5. 🗂️ 移动文件到对应目录
6. 📝 生成文件索引
7. 💾 将索引存储到SketchPad
```

### SketchPad 使用示例

#### 自动存储和检索
```python
# 文件操作自动存储
>>> 读取data.txt文件
系统响应：
✅ 文件内容已存储到SketchPad
🔑 SketchPad Key: file_a8b9c7d2
📄 内容长度: 1024 字符

# 后续引用存储的内容
>>> 使用key "file_a8b9c7d2" 的内容创建一个摘要
```

#### 手动数据管理
```
>>> /pad_store project_config {"name": "SimpleAgent", "version": "1.0"}
>>> /pad_tag project_config config,json,metadata
>>> /pad_search config
```

### 领域适配示例

#### 作为数据分析助手
通过修改Prompt，可以变成数据分析专家：
```
>>> 分析这个CSV文件，找出销售趋势和异常值
>>> 生成可视化图表并保存报告
>>> 建立预测模型评估下季度表现
```

#### 作为代码开发助手
```
>>> 设计一个REST API的项目结构
>>> 生成基础代码框架和配置文件
>>> 编写单元测试和文档
```

#### 作为文档写作助手
```
>>> 撰写技术文档，包括架构说明和使用指南
>>> 生成API文档和示例代码
>>> 创建用户手册和FAQ
```

---

**SimpleAgent** - 智能任务管理，灵活工具集成，无限扩展可能！

## 🌐 Web API 服务

SimpleAgent 提供了完全符合 OpenAI API 规范的 Web 服务接口，让您可以通过 HTTP API 调用智能体服务。

### 启动 Web 服务器

```bash
# 基本启动（默认 127.0.0.1:8000）
python start_web_server.py

# 指定主机和端口
python start_web_server.py --host 0.0.0.0 --port 8080

# 开发模式（文件修改后自动重载）
python start_web_server.py --reload
```

### API 端点

#### 基础端点
- `GET /` - 服务器信息
- `GET /health` - 健康检查
- `GET /docs` - Swagger API 文档
- `GET /redoc` - ReDoc API 文档

#### OpenAI 兼容端点
- `GET /v1/models` - 列出可用模型
- `POST /v1/chat/completions` - 聊天完成（支持流式和非流式）

### 使用示例

#### cURL 调用
```bash
# 非流式请求
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple-agent-v1",
    "messages": [
      {"role": "user", "content": "分析项目目录结构并生成报告"}
    ],
    "stream": false
  }'

# 流式请求
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "simple-agent-v1", 
    "messages": [
      {"role": "user", "content": "创建一个Python项目的基础结构"}
    ],
    "stream": true
  }'
```

#### Python 客户端
```python
import requests

# 基础客户端
def chat_with_agent(message: str):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "simple-agent-v1",
            "messages": [{"role": "user", "content": message}],
            "stream": False
        }
    )
    return response.json()

# 示例使用
result = chat_with_agent("帮我整理这个文件夹的内容")
print(result['choices'][0]['message']['content'])
```

#### OpenAI 客户端库
```python
from openai import OpenAI

# 使用OpenAI官方客户端库
client = OpenAI(
    api_key="not-needed",  # SimpleAgent不需要API密钥
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="simple-agent-v1",
    messages=[
        {"role": "user", "content": "执行系统维护任务"}
    ]
)

print(response.choices[0].message.content)
```

### Web API 特性

- **OpenAI 兼容**: 完全兼容 OpenAI Chat Completions API
- **流式输出**: 支持 Server-Sent Events (SSE) 流式响应
- **CORS 支持**: 允许跨域访问，便于前端集成
- **自动文档**: 自动生成 Swagger 和 ReDoc 文档
- **错误处理**: 标准化的错误响应格式
- **健康检查**: 提供服务状态监控端点
- **单例管理**: 使用全局Agent单例确保状态一致性

### 部署建议

#### 开发环境
```bash
python start_web_server.py --reload --host 127.0.0.1
```

#### 生产环境
```bash
# 使用 uvicorn 直接启动
uvicorn web_interface.server:app --host 0.0.0.0 --port 8000 --workers 4

# 或使用 gunicorn (需要安装)
gunicorn web_interface.server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 🤝 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

### 开发规范
- 遵循Python PEP 8编码规范
- 添加适当的类型注解
- 编写清晰的文档字符串
- 为新功能添加测试用例
- 更新README文档

## 📄 许可证

本项目采用 GPL2.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持与反馈

如果您遇到问题或有建议，请：
1. 查看文档和示例代码
2. 搜索已有的 Issues
3. 创建新的 Issue 详细描述问题
4. 参与社区讨论

### 常见问题

**Q: 如何添加新的工具？**
A: 参考 `tools/` 目录下的现有工具，使用 `@tool` 装饰器创建新工具，然后在 `agent/global_agent.py` 中添加到工具集。

**Q: 如何修改智能体的行为？**
A: 编辑 `agent/BaseAgent.py` 中的 `chat_impl` 函数的系统提示词。

**Q: SketchPad数据存储在哪里？**
A: SketchPad数据存储在内存中，支持持久化到文件系统。

**Q: 如何配置不同的LLM模型？**
A: 修改 `config/config.py` 中的模型配置，选择不同的接口。

## Star History

<a href="https://www.star-history.com/#NiJingzhe/SimpleAgent&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleAgent&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NiJingzhe/SimpleAgent&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NiJingzhe/SimpleAgent&type=Date" />
 </picture>
</a>
