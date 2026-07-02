# 🤖 AI 面试官 — 多 Agent 智能面试系统

基于 **LangGraph + FastAPI + DeepSeek** 的多 Agent 协作面试系统。支持简历解析、岗位匹配、模拟面试、实时评估和面试报告生成。

---

## 🎯 功能概览

| 功能 | 说明 |
|------|------|
| 📄 简历解析 | 上传 PDF/Word 简历，自动提取关键信息 |
| 🔍 岗位匹配 | 基于向量检索（ChromaDB）的简历-岗位智能匹配 |
| 💬 模拟面试 | 多 Agent 协作进行 H5/web 前端/数据工程等方向模拟面试 |
| 📊 实时评估 | 面试回答实时评分与反馈 |
| 📝 面试报告 | 自动生成详细面试报告 |
| 🧠 RAG 知识库 | 基于面试知识库的智能问答 |

---

## 🏗️ 技术架构

```
用户 (浏览器) → FastAPI Web 服务 → LangGraph Supervisor
                                      ├── Interview Agent（面试对话）
                                      ├── Evaluator Agent（回答评估）
                                      ├── Resume Match Agent（简历匹配）
                                      ├── Report Agent（报告生成）
                                      └── Direct Reply Agent（直接回复）
                                   ↓
                              DeepSeek LLM API
                              ChromaDB 向量数据库
```

### 技术栈

- **后端**: Python / FastAPI / LangGraph / LangChain
- **前端**: HTML5 / CSS3 / JavaScript（单页应用）
- **LLM**: DeepSeek API（对话/推理）+ 阿里云百炼 DashScope（向量嵌入）
- **向量库**: ChromaDB
- **持久化**: Redis → SQLite → 内存 三层降级策略

---

### 🔒 会话持久化与容错

系统采用 **Redis → SQLite → 内存** 三层会话持久化降级策略，确保长时间面试会话稳定运行：

| 层级 | 存储 | 场景 |
|------|------|------|
| L1 | Redis | 生产环境首选，高性能会话缓存 |
| L2 | SQLite | Redis 不可用时自动降级，本地持久化 |
| L3 | 内存 | 最终兜底，保证服务不中断 |

配合全链路接口重试、LLM JSON 解析容错，即使在 LLM 返回格式异常或网络波动时也能保障体验不崩。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API 密钥：

```env
DEEPSEEK_API_KEY=你的DeepSeek密钥
DASHSCOPE_API_KEY=你的DashScope密钥
```

### 3. 启动服务

```bash
uvicorn web_server:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问

浏览器打开 `http://localhost:8000`

---

## 📁 项目结构

```
├── web_server.py          # FastAPI Web 服务入口
├── agent_core.py          # Agent 核心（LLM 调用、工具定义）
├── agent_graph.py         # LangGraph 多 Agent 状态图
├── agent_main.py          # Agent 主调度逻辑
├── agents/                # Agent 模块
│   ├── supervisor.py      # Supervisor 调度器
│   ├── interview_agent.py # 面试对话 Agent
│   ├── evaluator_agent.py # 评估 Agent
│   ├── resume_match_agent.py # 简历匹配 Agent
│   ├── report_agent.py    # 报告生成 Agent
│   ├── direct_reply_agent.py # 直接回复 Agent
│   ├── redis_client.py    # Redis 客户端
│   ├── state.py           # 状态管理
│   └── utils.py           # 工具函数
├── knowledge_base.py      # RAG 知识库
├── vector_db.py           # ChromaDB 向量数据库
├── resume_parser.py       # 简历解析器
├── sqlite_db.py           # SQLite 数据库操作
├── generate_paper.py      # 论文生成脚本
├── static/
│   └── index.html         # Web 前端界面
├── interview_kb/          # 面试知识库文档
├── requirements.txt       # Python 依赖
└── .env.example           # 环境变量模板
```

---

## 🔧 主要依赖

| 组件 | 用途 |
|------|------|
| langchain + langgraph | 多 Agent 编排框架 |
| chromadb | 向量数据库，简历/岗位相似度检索 |
| fastapi + uvicorn | Web 服务框架 |
| langchain-openai | DeepSeek LLM 调用（OpenAI 兼容接口） |
| python-dotenv | 环境变量管理 |
| python-docx | 面试报告 Word 文档生成 |
| redis | 会话缓存（三层降级策略 L1） |
| Pillow + matplotlib | 图表与可视化 |

---

## 📸 界面预览

> 启动项目后访问 `http://localhost:8000` 即可体验


---

## ⚠️ 注意事项

- 首次运行需要配置 LLM API 密钥
- ChromaDB 向量库首次启动会自动初始化
- 项目默认使用阿里云百炼 DeepSeek API

