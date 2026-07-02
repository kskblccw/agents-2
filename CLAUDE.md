# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

```bash
# 激活 conda 环境
conda activate agent-jl

# 启动开发服务器（热重载）
uvicorn web_server:app --reload --host 0.0.0.0 --port 8000

# 直接运行 CLI 交互模式（不使用 Web 服务）
python agent_main.py
```

## 环境变量

项目需要两个 API Key，配置在 `.env` 文件中：
- `DEEPSEEK_API_KEY` — LLM 对话/推理（DeepSeek API，OpenAI 兼容格式）
- `DASHSCOPE_API_KEY` — 文本向量嵌入（阿里云 DashScope `text-embedding-v2`）


## 架构概览

### 多 Agent 系统（V2，当前版本）

系统基于 **LangGraph Supervisor 星型拓扑**，一个 Supervisor 节点作为中心路由器，调度 5 个专业 Agent 节点：

```
用户消息 → Supervisor（路由决策）→ ResumeMatch / Interview / Evaluate / Report / DirectReply
                ↑                        ↓（每个 Agent 完成后返回）
                └────────────────────────┘
```

**关键文件**：
- [agents/supervisor.py](agents/supervisor.py) — Supervisor 节点 + 规则路由 + LLM 路由 + 图构建（`build_multi_agent_graph()`）
- [agents/state.py](agents/state.py) — 共享状态定义（`MultiAgentState`）、Agent 权限白名单（`AGENT_PERMISSIONS`）、`with_permissions` 装饰器
- [web_server.py](web_server.py) — FastAPI 入口，LangGraph 图以 `graph.invoke()` 方式调用，SSE 流式通过 `graph.stream()` 实现

### Supervisor 路由优先级

Supervisor 使用多阶段决策管道，优先级从高到低：

1. **Agent FINISH 信号** — Agent 设置了 `_agent_signal == "FINISH"` 时立即停止
2. **硬性终止** — 总步数 > 50 强制 END
3. **第一优先级铁律** — `waiting_for_user=True` 或需要简历但用户未提供时 FINISH
4. **Agent 显式信号** — Agent 设置的 `_agent_signal` 指向具体 Agent
5. **规则路由**（0 token 消耗）— 关键词意图检测覆盖 ~80% 场景
6. **LLM 路由** — 规则无法判断时调用 LLM，3 次重试后降级到默认路由
7. **默认路由** — 按阶段确定性推进，永不失败

### 5 个 Agent 节点

| Agent | 文件 | 职责 |
|-------|------|------|
| ResumeMatch | [agents/resume_match_agent.py](agents/resume_match_agent.py) | 简历解析（调用 `ResumeParser`）→ 岗位匹配（调用 `MatcherAgent`）→ 岗位选择 |
| Interview | [agents/interview_agent.py](agents/interview_agent.py) | 生成 5 道面试题（带 RAG 知识库增强）→ 逐题提问 |
| Evaluate | [agents/evaluator_agent.py](agents/evaluator_agent.py) | 对用户回答评分（1-10 分），设置 `_agent_signal: "interview"` 触发下一题 |
| Report | [agents/report_agent.py](agents/report_agent.py) | 加权综合评分 + LLM 生成结构化面试报告（技术/表达/经验/素质 + 录用建议） |
| DirectReply | [agents/direct_reply_agent.py](agents/direct_reply_agent.py) | 通用问答、个人信息查询、问候语，上限 5 轮防无限循环 |

### 共享状态与权限隔离

`MultiAgentState`（[agents/state.py](agents/state.py)）是所有 Agent 共享的 `TypedDict`。每个 Agent 只能写入白名单声明的字段（`AGENT_PERMISSIONS`），越权写入被 `with_permissions` 装饰器静默丢弃。白名单之外只有 `messages` 字段是所有 Agent 可写的（LangChain 的 `add_messages` reducer 自动追加）。

状态关键字段：
- `messages` — 共享消息历史（`add_messages` reducer）
- `next_agent` / `_agent_signal` / `_agent_history` / `_total_steps` — Supervisor 控制
- `resume_data` / `job_matches` / `selected_job` — ResumeMatch 写入
- `interview_questions` / `answers` / `evaluations` — Interview + Evaluate 读写
- `final_report` / `overall_score` — Report 写入

### 面试循环流程

```
用户回答 → Evaluate（评分，设 signal="interview"）→ Interview（出下一题）
  → 等待用户回答 → Evaluate → Interview → ... 5 题全部完成
  → Report（生成报告）→ FINISH
```

### 三层降级策略

- **会话检查点**（LangGraph）：Redis → SQLite → Memory（`create_checkpointer` in supervisor.py）
- **聊天历史存储**（ChatStore）：Redis → SQLite → Memory（web_server.py）
- **LLM 调用**：3 次重试 + 指数退避（`LLMClient` in agents/utils.py）
- **JSON 解析**：LLM 返回格式异常时 fallback 到默认值

### V1 遗留代码

[agent_graph.py](agent_graph.py) 和 [agent_core.py](agent_core.py) 中的 `InterviewerAgent` 类是 V1 单 Agent 架构（一个 LLM 节点绑定 6 个 Tool），当前 Web 服务已不使用，但 `agent_core.py` 中的 `MatcherAgent` 和 `@tool` 函数仍被 V2 系统调用。`agent_main.py` 是 CLI 入口，也仍在使用。

### 关键集成点

- **ChromaDB**（[vector_db.py](vector_db.py)）：存储岗位描述的向量嵌入，`MatcherAgent.match_resume()` 做语义相似度检索
- **RAG 知识库**（[knowledge_base.py](knowledge_base.py)）：面试知识库，Interview Agent 和 Evaluator Agent 用它检索出题和评分的参考上下文
- **Redis 客户端**（[agents/redis_client.py](agents/redis_client.py)）：缓存用户简历信息、岗位匹配结果、面试状态
- **简历解析器**（[resume_parser.py](resume_parser.py)）：支持 PDF/DOCX/TXT，LLM 提取结构化信息
