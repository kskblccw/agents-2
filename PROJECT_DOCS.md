# 🤖 AI 面试官 — 多 Agent 智能面试系统 完整学习文档

> **适用场景**：学习本项目源码、准备 Agent/AI 相关岗位面试、理解多 Agent 系统架构设计
>
> **作者推荐阅读顺序**：架构概览 → 状态管理 → Supervisor 路由 → 各 Agent 节点 → 持久化 → Web 层

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术架构全景](#2-技术架构全景)
3. [核心设计理念](#3-核心设计理念)
4. [模块详解](#4-模块详解)
5. [数据流与状态管理](#5-数据流与状态管理)
6. [路由与调度系统](#6-路由与调度系统)
7. [持久化策略](#7-持久化策略)
8. [Web 服务与前端](#8-web-服务与前端)
9. [RAG 知识库](#9-rag-知识库)
10. [简历解析与岗位匹配](#10-简历解析与岗位匹配)
11. [容错与防御式编程](#11-容错与防御式编程)
12. [面试准备 — 技术问答](#12-面试准备--技术问答)
13. [扩展与改进方向](#13-扩展与改进方向)

---

## 1. 项目概述

### 1.1 这是什么项目？

一个**基于 LangGraph 多 Agent 架构的 AI 模拟面试系统**，核心功能包括：

| 功能 | 说明 | 涉及 Agent |
|------|------|-----------|
| 📄 简历解析 | 上传 PDF/DOCX/TXT，LLM 自动提取结构化信息 | ResumeMatch Agent |
| 🔍 岗位匹配 | ChromaDB 向量语义检索，匹配最合适的岗位 | ResumeMatch Agent |
| 💬 模拟面试 | 根据简历+岗位生成 5 道个性化面试题，逐题提问 | Interview Agent |
| 📊 实时评分 | 每道题回答后 LLM 打分(1-10分) + 改进建议 | Evaluator Agent |
| 📝 面试报告 | 全部答完后生成综合评估报告 + 录用建议 | Report Agent |
| 💬 通用问答 | 技术问题解答、面试技巧咨询 | DirectReply Agent |

### 1.2 技术栈速览

```
语言：      Python 3.10+
Web框架：   FastAPI + Uvicorn (异步)
AI框架：    LangChain + LangGraph (Agent 编排)
LLM：      DeepSeek (deepseek-chat / deepseek-v4-pro)
向量嵌入：   DashScope (阿里云百炼 text-embedding-v2, 1536维)
向量库：    ChromaDB
缓存：     Redis
持久化：    Redis → SQLite → Memory 三层降级
文档处理：  pdfplumber + python-docx
前端：     HTML5/CSS3/JS (零框架单页应用)
```

### 1.3 项目规模

| 指标 | 数值 |
|------|------|
| 总文件数 | ~33 个源码文件 |
| 总代码量 | ~15,000 行 |
| Agent 数量 | 5 个业务 Agent + 1 个 Supervisor |
| API 端点 | 6 个 REST 端点 |
| 知识库文档 | 6 个 Markdown 文件 |

---

## 2. 技术架构全景

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          浏览器 (SPA)                                 │
│                     static/index.html                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP/SSE
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI Web Server                                │
│                     web_server.py                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐  │
│  │ ChatStore   │  │ API Routes   │  │ SSE Streaming              │  │
│  │ Redis→SQLite│  │ /api/chat    │  │ /api/chat/stream           │  │
│  │ →Memory     │  │ /api/upload  │  │                            │  │
│  └─────────────┘  └──────┬───────┘  └────────────────────────────┘  │
│                          │                                            │
│                   asyncio.to_thread()                                 │
│                          │                                            │
└──────────────────────────┼────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LangGraph 多 Agent 图                               │
│                    agents/supervisor.py                               │
│                                                                       │
│                         ┌──────────┐                                  │
│                         │  用户输入  │                                  │
│                         └─────┬────┘                                  │
│                               │                                       │
│                               ▼                                       │
│                        ┌──────────────┐                               │
│                        │  SUPERVISOR   │  ← 规则路由(80%) + LLM路由(20%) │
│                        │  调度中心      │                               │
│                        └───┬───┬───┬───┘                               │
│                            │   │   │   │                               │
│              ┌─────────────┘   │   │   └─────────────┐                │
│              ▼                 │   │                 ▼                │
│     ┌──────────────┐          │   │        ┌──────────────┐          │
│     │ ResumeMatch  │          │   │        │  DirectReply │          │
│     │  简历匹配     │          │   │        │   通用问答    │          │
│     └──────┬───────┘          │   │        └──────┬───────┘          │
│            │                  │   │               │                  │
│            └──────────────────┼───┼───────────────┘                  │
│                               │   │                                  │
│              ┌────────────────┘   └────────────────┐                 │
│              ▼                                      ▼                 │
│     ┌──────────────┐                       ┌──────────────┐          │
│     │  Interview   │                       │  Evaluator   │          │
│     │  面试主持     │◄──────难度调整──────────│   回答评分    │          │
│     └──────┬───────┘                       └──────┬───────┘          │
│            │                                      │                  │
│            └──────────────────┬───────────────────┘                  │
│                               │                                      │
│                               ▼                                      │
│                      ┌──────────────┐                                │
│                      │   Report     │                                │
│                      │  报告生成     │                                │
│                      └──────┬───────┘                                │
│                             │                                        │
│                             ▼                                        │
│                        ┌──────────┐                                  │
│                        │   END    │                                  │
│                        └──────────┘                                  │
│                                                                       │
│   所有 Agent 执行完毕后都回到 Supervisor，由 Supervisor 决定下一步。      │
│   ★ 星型拓扑 (Star Topology)：Supervisor 中心 + Agent 辐条             │
└──────────────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌──────────┐
        │ DeepSeek│  │ChromaDB │  │  Redis   │
        │   LLM   │  │ 向量库   │  │ 会话缓存  │
        └─────────┘  └─────────┘  └──────────┘
```

### 2.2 Agent 职责矩阵

| Agent | 文件 | 核心职责 | 输入 | 输出 |
|-------|------|---------|------|------|
| **Supervisor** | `supervisor.py` | 分析状态、路由决策 | 全局状态 | `next_agent` |
| **ResumeMatch** | `resume_match_agent.py` | 简历解析 + 岗位匹配 | 用户文件路径/文本 | `resume_data`, `job_matches`, `selected_job` |
| **Interview** | `interview_agent.py` | 生成面试题 + 逐题提问 | 简历 + 岗位信息 | `interview_questions`, 面试消息 |
| **Evaluator** | `evaluator_agent.py` | 评分 + 反馈 + 难度调整 | 用户回答 + 评分标准 | `evaluations`, `answers`, `last_answer_score` |
| **Report** | `report_agent.py` | 综合评估报告 | 全部 Q&A 记录 | `final_report`, `overall_score` |
| **DirectReply** | `direct_reply_agent.py` | 通用问答/闲聊 | 用户任意消息 | 对话回复 |

---

## 3. 核心设计理念

### 3.1 星型拓扑 (Star Topology)

所有 Agent 都围绕 Supervisor 枢纽运行，这是本项目最核心的架构决策：

```
为什么用星型而非网状？

星型 (本项目)：
  ✓ Supervisor 统一决策 → 控制流清晰
  ✓ 每个 Agent 职责单一 → 易于维护测试
  ✓ 规则路由 + LLM 路由 → 成本可控 (80% 免 token)
  ✗ Supervisor 是单点 → 需要容错设计 (本项目已做)

网状 (备选，未采用)：
  ✓ 灵活，Agent 直接通信
  ✗ 循环难控，容易死锁
  ✗ 调试困难，状态追踪复杂
```

### 3.2 混合路由策略 (Hybrid Routing)

```
用户输入
    │
    ▼
┌─────────────────────┐
│  规则路由 (Rule-Based) │  ← 覆盖 80%，0 token 消耗
│  - 意图检测 (关键词)    │
│  - 阶段判断 (状态机)    │
│  - 优先级排序          │
└─────────┬───────────┘
          │ 无法确定?
          ▼
┌─────────────────────┐
│  LLM 路由 (LLM-Based)  │  ← 覆盖 20%，DeepSeek 决策
│  - 3次重试 + 降级       │
│  - temperature=0.0     │
└─────────┬───────────┘
          │ 仍然失败?
          ▼
┌─────────────────────┐
│  默认路由 (Default)    │  ← 兜底，永不失败
│  - 确定性规则          │
│  - 最保守的安全路径     │
└─────────────────────┘
```

### 3.3 多层降级策略 (Graceful Degradation)

本项目的核心工程哲学：**每一层都可以坏，系统不能挂**。

```
持久化降级：      Redis → SQLite → Memory
LLM 降级：       DeepSeek v4-pro → deepseek-chat → 规则兜底
路由降级：       规则路由 → LLM路由(3次重试) → 默认路由
Agent 降级：     主逻辑 → 异常捕获 → 错误记录 → 状态不污染
用户信息降级：    Redis → state.resume_data → "无数据"
```

### 3.4 Agent 权限控制

这是一个**独特的创新点**，在面试中可以作为亮点展示：

```python
# 每个 Agent 只能写白名单字段，通过装饰器零侵入实现
@with_permissions("resume_match")
def resume_match_node(state):
    return {
        "resume_data": {...},        # ✅ 白名单内 → 允许写入
        "interview_questions": [...], # ❌ 不属于 resume_match → 静默丢弃 + 警告
    }
```

设计精妙之处：
- **声明式**：权限配置和业务逻辑完全分离
- **零侵入**：Agent 节点无需感知权限系统
- **防御式**：非法写入静默丢弃（而非抛异常打断流程）
- **可观测**：每次拦截都有日志记录

---

## 4. 模块详解

### 4.1 agents/state.py — 状态管理核心

**地位**：整个多 Agent 系统的"宪法"，定义了什么数据存在、谁能写什么。

#### MultiAgentState 字段分类

```python
class MultiAgentState(TypedDict, total=False):
    # ---- 消息层 ----
    messages: Annotated[list, add_messages]  # LangGraph 自动追加(operator.add)

    # ---- 路由控制层 (Supervisor 专用) ----
    next_agent: str              # 下一步路由目标
    waiting_for_user: bool       # 是否等待用户输入
    _agent_signal: str           # Agent 给 Supervisor 的信号

    # ---- 追踪层 (内部字段，_前缀) ----
    _agent_history: List[str]    # Agent 调用历史 ["resume_match", "interview", ...]
    _total_steps: int            # 总步数
    _error_count: Dict[str, int] # 各 Agent 错误计数
    _last_error: str             # 最近错误信息
    _direct_reply_count: int     # DirectReply 轮数(防无限循环)

    # ---- 业务层：简历&匹配 ----
    resume_data: dict            # 结构化简历
    job_matches: list            # 岗位匹配结果
    selected_job: dict           # 用户选择的岗位

    # ---- 业务层：面试 ----
    interview_questions: list    # 5道面试题
    current_question_idx: int    # 当前第几题(0-based)
    answers: list                # 已回答记录
    evaluations: list            # 评分记录
    last_answer_score: float     # 最新评分
    interview_stage: str         # 流程阶段标识
    next_difficulty: str         # 下一题难度(easy/normal/hard)

    # ---- 业务层：结果 ----
    final_report: str            # 最终报告
    overall_score: float         # 综合评分
```

#### 关键设计决策

1. **`_` 前缀字段** — 表示内部追踪字段，不面向用户展示，Supervisor 独占写入
2. **`messages` 使用 LangGraph 的 `add_messages` reducer** — 所有 Agent 都可以追加消息（这是框架级别行为，权限系统不管）
3. **`total=False`** — TypedDict 的所有字段可选，状态渐进式构建

### 4.2 agents/supervisor.py — 调度核心 (1228 行)

**地位**：整个系统的"大脑"，决定了每一步哪个 Agent 上场。

#### supervisor_node 的 9 步处理流程

```
Step 1: 重置过期状态       → 如果 agent_signal 是 END/FINISH，清空它
Step 2: 检查 Agent 信号     → Agent 可以通过 _agent_signal 直接告知下一步
Step 3: 递增步数计数器      → _total_steps += 1
Step 4: 检查硬终止条件      → _total_steps > 50 → 强制 END
Step 5: 最先优先级检查      → 铁规则（不可被规则路由覆盖）
Step 6: 规则路由            → 覆盖 80% 场景，零 token
Step 7: LLM 路由            → 覆盖 20% 模糊场景
Step 8: 默认路由兜底        → 以上都失败时的安全路径
Step 9: 条件边拦截          → 循环检测，防止无限循环
```

#### 路由优先级体系

```
优先级 0：面试进行中 → 用户输入 = 回答 → evaluate
优先级 1：明确结束面试 → report
优先级 2：简历意图 → resume_match
优先级 3：岗位匹配意图 → resume_match
优先级 4：个人信息查询 → direct_reply
优先级 5：开始面试意图 → interview
优先级 6：通用问答意图 → direct_reply
优先级 7：按阶段路由 → 根据 interview_stage 决定
```

#### 关键代码模式

```python
def _rule_based_route(state: dict) -> Optional[str]:
    # 返回 None → 表示"我无法决定，让 LLM 来"
    # 返回 "agent_name" → 确定路由

    # 面试中最关键的保护逻辑：
    if questions and current_idx < len(questions):
        # 面试进行中，用户的任何输入都视为回答
        # 防止用户回答内容中的"简历""结束"等词语触发误路由！
        if _is_explicit_end_interview(last_user_msg):
            return "report"  # 只有明确说"结束面试"才跳转
        return "evaluate"    # 否则全部走评分
```

### 4.3 agents/resume_match_agent.py — 简历解析与匹配 (622 行)

**状态机设计**：该 Agent 被 Supervisor 重复调用，每次检查当前阶段执行不同逻辑。

```
resume_match Agent 内部状态机：

调用 1: 没有 resume_data → 提示用户上传简历
调用 2: 有简历文件路径 → 调用 resume_parser 解析 → 存入 resume_data
调用 3: 有简历但没匹配 → 调用 ChromaDB 语义搜索 → 返回 top-3 岗位
调用 4: 有匹配结果但没选择 → 展示岗位让用户选
调用 5: 用户选了岗位 → 存入 selected_job，发出信号让 Supervisor 转 interview
```

**关键设计**：这个 Agent 是被多次调用的（最多 5 次），每次做一件事。这遵循了 Agent 设计的"单一职责"原则，而不是在一次调用中完成所有步骤。

### 4.4 agents/interview_agent.py — 面试主持 (358 行)

```
interview Agent 内部状态机：

情况 1: 有 selected_job 但没有 questions
    → 调用 LLM + RAG 知识库，生成 5 道面试题
    → 题目覆盖：技术基础、项目经验、系统设计、行为面试、专业方向

情况 2: 有 questions 但还没问完
    → 发送当前题目给用户
    → 递增 current_question_idx

难度调整机制：
    next_difficulty 由 Evaluator Agent 设置
    根据上一题得分动态调整下一题难度：
      得分 >= 8 → next_difficulty = "hard"
      得分 >= 5 → next_difficulty = "normal"
      得分 < 5  → next_difficulty = "easy"
```

### 4.5 agents/evaluator_agent.py — 回答评分 (341 行)

```
评分流程：

1. 获取用户最新回答
2. 从 RAG 知识库检索评分标准 (answer_guide.md)
3. 调用 LLM 进行多维度评分：
   - 技术准确性 (40%)
   - 表达清晰度 (25%)
   - 深度与广度 (20%)
   - 实践经验 (15%)
4. 返回 JSON：{score, commentary, strengths, weaknesses, next_difficulty}
5. 设置 _agent_signal 控制下一步：
   - 还有题 → "interview"
   - 全部答完 → "" (空串，Supervisor 会路由到 report)
```

### 4.6 agents/report_agent.py — 报告生成 (229 行)

```
报告生成流程：

1. 收集全部 Q&A + 评分数据
2. 计算加权综合评分（前面的题权重更高）
3. 调用 LLM 生成完整报告：
   - 综合评分
   - 各维度评分（技术、表达、思维、潜力）
   - 逐题详细点评
   - 优势与不足
   - 录用建议（推荐录用 / 可考虑 / 不推荐）
   - 后续学习建议
4. 清理面试相关状态 (questions, answers 等清空)
5. 设置 interview_stage = "done"
```

### 4.7 agents/direct_reply_agent.py — 通用问答 (579 行)

**多模型支持**：

```python
def _get_direct_reply_llm():
    model = os.getenv("DIRECT_REPLY_MODEL", "deepseek-chat")
    if model.startswith("qwen"):
        # DashScope (阿里云百炼)
        return ChatOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", ...)
    elif model.startswith("glm"):
        # Zhipu AI (智谱)
        return ChatOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4", ...)
    else:
        # DeepSeek (默认)
        return ChatOpenAI(base_url="https://api.deepseek.com/v1", ...)
```

**个人信息查询的隐私保护**：
```
查询链路：Redis → state.resume_data → "未找到相关信息"
故意不走 RAG，防止从知识库中泄露其他候选人的信息
```

### 4.8 agents/utils.py — 工具函数 (478 行)

三大核心组件：

```
┌───────────────────────────────────────────────────┐
│                  safe_llm_call()                   │
│  统一的 LLM 调用入口，组合以下三个组件：             │
│                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │ RateLimiter  │  │ResponseCache │  │LLMClient │ │
│  │ 滑动窗口限流  │  │ TTL-LRU缓存  │  │ 重试+超时 │ │
│  │ 2 req/s     │  │ max=100     │  │ retry=3  │ │
│  │ per thread  │  │ ttl=300s    │  │ time=30s │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
└───────────────────────────────────────────────────┘
```

- **RateLimiter**：滑动窗口限流，按 `thread_id` 维度，默认每秒 2 次
- **ResponseCache**：TTL + LRU 双层淘汰，缓存 LLM 响应减少重复调用
- **LLMClient**：OpenAI 兼容客户端，30s 超时，最多 3 次重试，指数退避

---

## 5. 数据流与状态管理

### 5.1 完整的数据流

```
用户: "我要面试" + 上传简历
    │
    ▼
[web_server.py] 接收请求 → asyncio.to_thread() → graph.ainvoke()
    │
    ▼
[Supervisor] 分析状态 → 路由到 resume_match
    │
    ▼
[ResumeMatch Agent] 解析简历 → 向量匹配岗位 → 用户选择岗位
    │
    ▼  (state 更新：resume_data ✅, job_matches ✅, selected_job ✅)
[Supervisor] 检测到岗位已选 + 无面试题 → 路由到 interview
    │
    ▼
[Interview Agent] RAG + LLM 生成 5 题 → 发送第 1 题
    │
    ▼  (state 更新：interview_questions ✅, current_question_idx=0)
[Supervisor] 检测到 waiting_for_user → 停止调度，等待用户输入
    │
    ▼  用户输入回答
[Supervisor] 检测到面试进行中 → 路由到 evaluate
    │
    ▼
[Evaluator Agent] RAG 查评分标准 → LLM 打分 → 设置 next_difficulty
    │
    ▼  (state 更新：answers 追加, last_answer_score 更新)
[Supervisor] 检测到还有题目 → 路由到 interview
    │
    ▼  ... 循环 5 次 ...
[Supervisor] 检测到全部答完 → 路由到 report
    │
    ▼
[Report Agent] 收集数据 → LLM 生成报告 → 清理面试状态
    │
    ▼
[Supervisor] 检测到 final_report 存在 → FINISH
```

### 5.2 LangGraph 状态持久化

LangGraph 的 **Checkpoint** 机制是理解本项目的关键：

```python
# 每次 graph.ainvoke() 后，状态自动保存到 Checkpointer
# 下次调用时，LangGraph 自动从上次中断处恢复

# 这实现了：
# 1. 会话跨请求恢复（用户关闭浏览器后回来继续面试）
# 2. 精确的断点续传（面试中断在第 3 题，恢复后继续第 3 题）
# 3. 时间旅行调试（可以回溯每个 step 的状态快照）
```

---

## 6. 路由与调度系统

### 6.1 为什么需要混合路由？

```
纯规则路由的问题：无法处理模糊表达
  例："我想看看有什么岗位" → 可以规则匹配
  例："我学Python的，有什么合适的吗" → 规则可能误判

纯 LLM 路由的问题：成本高 + 延迟大
  每次路由都调 LLM：30 个 step → 30 次 LLM 调用 → 约 ¥0.5 + 15s 延迟

混合路由 = 规则(80%) + LLM(20%) = 最优解
  30 个 step → 约 6 次 LLM 调用 → 约 ¥0.1 + 3s 延迟
```

### 6.2 路由的防御式设计

```
每次路由决策都经过 3 层：
  Layer 1: 规则路由 → 能确定就直接用
  Layer 2: LLM 路由 → 规则不确定时，LLM 3次重试
  Layer 3: 默认路由 → LLM 也失败时的确定性兜底

条件边还有额外保护：
  - 同一 Agent 连续 5 次 → 强制 END
  - 总步数 > 50 → 强制 END
  - 提前检查(pre-emptive)：如果本次路由会使某 Agent 连续第 6 次 → 阻止
```

---

## 7. 持久化策略

### 7.1 三层降级体系

```
          ┌─────────────────────────────────────────────────┐
          │              持久化层级                          │
          │                                                  │
          │  L1: Redis (生产)                                │
          │  ├── ChatStore: 聊天历史 + 会话列表               │
          │  ├── Checkpointer: LangGraph 状态快照             │
          │  └── UserInfoStore: 用户信息缓存 (TTL 7天)        │
          │                                                  │
          │  ↓ 不可用时自动降级                                │
          │                                                  │
          │  L2: SQLite (本地持久化)                          │
          │  ├── ChatStore: 本地 DB 持久化聊天                │
          │  ├── Checkpointer: 本地状态快照                   │
          │  └── InterviewDB: 简历/面试/Q&A/报告 CRUD        │
          │                                                  │
          │  ↓ 不可用时自动降级                                │
          │                                                  │
          │  L3: Memory (内存兜底)                            │
          │  ├── ChatStore: dict + list 内存存储              │
          │  └── Checkpointer: MemorySaver                    │
          │  特点：重启数据丢失，但服务不中断                   │
          └──────────────────────────────────────────────────┘
```

### 7.2 初始化代码模式

```python
def create_checkpointer():
    # 尝试 Redis
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.Redis.from_url(redis_url, socket_connect_timeout=3)
        r.ping()
        saver = RedisSaver(r)  # langgraph-checkpoint-redis
        logger.info("[Checkpointer] Using RedisSaver")
        return saver
    except Exception as e:
        logger.warning(f"[Checkpointer] Redis unavailable: {e}")

    # 降级 SQLite
    try:
        saver = SqliteSaver.from_conn_string("checkpoints.db")
        logger.info("[Checkpointer] Using SqliteSaver (fallback)")
        return saver
    except Exception as e:
        logger.warning(f"[Checkpointer] SQLite unavailable: {e}")

    # 兜底 Memory
    logger.info("[Checkpointer] Using MemorySaver (last resort)")
    return MemorySaver()
```

---

## 8. Web 服务与前端

### 8.1 FastAPI Web Server (web_server.py, 941 行)

```
API 端点设计：

GET  /                       → 重定向到 /static/index.html
POST /api/chat               → 同步聊天（等待 Agent 完成后返回）
POST /api/chat/stream        → SSE 流式聊天（实时看到 Agent 输出）
POST /api/upload             → 文件上传（简历 PDF/DOCX）
GET  /api/status/{session_id} → 查询会话状态
POST /api/reset/{session_id}  → 重置会话
GET  /api/sessions            → 获取会话列表
POST /api/sessions            → 创建新会话
DELETE /api/sessions/{id}     → 删除会话
```

### 8.2 线程安全处理

```python
# LangGraph 的同步调用在异步 FastAPI 中的处理方式
async def chat_endpoint():
    # contextvars 传递 session_id 给同步函数
    session_var.set(session_id)

    # asyncio.to_thread() 将同步图调用放入线程池
    result = await asyncio.to_thread(
        graph.ainvoke,  # 注意：这里其实可以是 graph.invoke()
        {"messages": [HumanMessage(content=user_msg)]},
        {"configurable": {"thread_id": session_id}}
    )
```

### 8.3 前端架构 (static/index.html, 1191 行)

```
组件清单：
  - 渐变动画背景
  - 左侧会话栏：会话 CRUD、新建会话按钮
  - 主聊天区：消息气泡、Markdown 渲染
  - 输入区：文本输入 + 拖拽文件上传 + 发送按钮
  - 状态面板：面试阶段、简历状态、岗位、评分、报告

技术选型：
  - 零框架（纯 HTML/CSS/JS），无 React/Vue 依赖
  - marked.js CDN 做 Markdown 渲染
  - EventSource API 做 SSE 流式接收
  - CSS Variables 做主题定制
```

---

## 9. RAG 知识库

### 9.1 知识库结构

```
interview_kb/
├── python_backend.md   ← Python 后端面试题及参考答案
├── fullstack.md        ← 全栈开发面试题
├── data_engineer.md    ← 数据工程面试题
├── behavioral.md       ← 行为面试题 (STAR 框架)
├── answer_guide.md     ← 评分标准 + 评价模板 + 追问策略
└── cxq.md             ← 个人信息 (git-ignored)
```

### 9.2 RAG 检索的三个应用场景

```
场景 1：生成面试题时 → 检索对应方向的知识库，紧贴真实面试
场景 2：评分时 → 检索评分标准 (answer_guide.md)，让评分有据可依
场景 3：追问时 → 检索追问策略，让面试更深入

实现方式：
  ChromaDB 多集合 (multi-collection)
  - 每个知识库文件一个 collection
  - chunk_size=500, chunk_overlap=80
  - 检索 top-k=3 最相关片段
```

### 9.3 RAG 链路

```python
# knowledge_base.py → RAGChain
class RAGChain:
    def get_question_context(self, direction: str, count: int = 5):
        """为生成面试题提供上下文"""
        # 1. 从对应方向的 collection 检索 top-k 片段
        # 2. 拼接成上下文文本
        # 3. 注入 LLM prompt

    def get_evaluation_context(self, question: str, answer: str):
        """为评分提供标准"""
        # 1. 检索 answer_guide.md 中的评分标准
        # 2. 检索对应方向的参考答案
        # 3. 注入 LLM prompt

    def get_follow_up_strategy(self, score: float):
        """根据得分推荐追问策略"""
        # 高分 → 追问更深入的问题
        # 低分 → 给提示，降低难度
```

---

## 10. 简历解析与岗位匹配

### 10.1 简历解析流程

```
上传文件 (PDF/DOCX/TXT)
    │
    ▼
┌──────────────────────────────┐
│ 文件类型检测                   │
│ PDF  → pdfplumber 提取文本     │
│ DOCX → python-docx 提取文本   │
│ TXT  → 直接读取               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ LLM 结构化提取                 │
│ Prompt: "请从以下简历文本中提取 │
│   姓名、学历、技能、工作经历..." │
│                               │
│ 返回 JSON:                    │
│ {                             │
│   "name": "张三",             │
│   "education": "本科/硕士",    │
│   "skills": ["Python","..."], │
│   "experience": [...],        │
│   "summary": "..."            │
│ }                             │
└──────────────────────────────┘
```

### 10.2 岗位匹配

```
ChromaDB 向量匹配原理：

1. 将 3 个岗位 JD 向量化存储
   - 每个岗位存储 2 份文档：(全文, 技能关键词)
   - 使用 DashScope text-embedding-v2 (1536维)

2. 将候选人简历向量化
   - 拼接技能 + 工作经历 + 求职意向

3. 语义相似度计算
   - cosine similarity
   - 综合全文匹配分 + 技能匹配分

4. LLM 生成匹配理由
   - Top-3 岗位 + LLM 分析每个岗位的匹配点和差距
```

---

## 11. 容错与防御式编程

这是本项目**最亮眼的工程实践集合**，在面试中可以重点阐述：

### 11.1 容错机制全景

| 层级 | 策略 | 实现位置 |
|------|------|---------|
| LLM 调用 | 30s 超时 + 3 次重试 + 指数退避 | `utils.py:LLMClient` |
| LLM JSON 解析 | JSON.loads → 正则提取 → 默认值 | 各个 Agent |
| 路由决策 | 规则 → LLM(3重试) → 默认 | `supervisor.py` |
| Agent 执行 | 异常捕获 + 错误计数 + 状态不污染 | `state.py:with_permissions` |
| 持久化 | Redis → SQLite → Memory | `supervisor.py`, `web_server.py` |
| 无限循环 | 连续5次检测 + 总步数50上限 | `state.py:check_and_enforce_loop_limits` |
| 会话过期 | TTL 7天自动清理 | `redis_client.py` |

### 11.2 JSON 解析的容错设计

```python
def _parse_json_response(text: str) -> dict:
    """多层 JSON 解析容错"""

    # 方式 1：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 方式 2：提取 ```json ... ``` 代码块
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # 方式 3：提取 { ... } 最外层 JSON
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # 方式 4：返回默认值，保证流程不中断
    logger.warning(f"Failed to parse JSON, returning default")
    return {"score": 5, "commentary": "评分解析失败，使用默认评分"}
```

### 11.3 Agent 异常隔离

```python
# with_permissions 装饰器的异常处理
@with_permissions("evaluate")
def evaluate_node(state):
    # 即使这里抛出任何异常
    # 装饰器会：
    # 1. 捕获异常
    # 2. 记录到 _last_error
    # 3. 递增 _error_count
    # 4. 返回最小的安全状态
    # 5. 不污染任何业务数据
    ...
```

---

## 12. 面试准备 — 技术问答

> 如果用这个项目找 Agent/AI 方向的工作，下面是最可能被问到的技术问题：

### Q1: 为什么选择 LangGraph 而不是其他 Agent 框架？

**参考答案**：
- LangGraph 提供了精确的状态管理和可控的图执行流程，不像 AutoGPT 那样容易失控
- 内置 Checkpoint 机制支持断点续传和人工介入（human-in-the-loop）
- StateGraph 的显式状态定义让多 Agent 间的数据流可追踪、可调试
- 与 LangChain 生态的深度集成（Tool、RAG 等）

**项目中体现**：我们的 Supervisor + 5 Agents 星型拓扑，每个 Agent 的输入输出完全由 StateGraph 管理，权限通过装饰器自动执行。

### Q2: 如何处理 LLM 返回格式不稳定？

**参考答案**：多层 JSON 解析容错 + 默认值兜底。项目中实现了 4 层解析：直接 parse → 代码块提取 → 正则提取 → 默认值。

### Q3: 多 Agent 系统如何防止死循环？

**参考答案**：双重保护：同一 Agent 连续 5 次触发强制 END + 总步数 50 上限。还有 pre-emptive check：在实际路由前检查是否会触发连续 6 次。

### Q4: 为什么用混合路由而不是全 LLM 路由？

**参考答案**：成本优化和延迟优化。规则路由覆盖 80%，零 token 消耗零延迟。LLM 只处理模糊场景。这是生产级系统的务实做法。

### Q5: 三层降级的具体价值？

**参考答案**：每一层都有明确的存在意义：Redis 高性能但需要运维，SQLite 零运维但性能较低，Memory 重启丢失但保证服务不中断。这个设计让系统可以运行在任何环境（开发机、服务器、容器）。

### Q6: Agent 权限控制的目的是什么？

**参考答案**：防止 Agent 之间的状态污染。在多 Agent 系统中，如果 Agent A 不小心修改了 Agent B 的数据，调试极其困难。通过声明式白名单 + 装饰器，在框架层面杜绝这种问题。

### Q7: RAG 在面试系统中的具体作用？

**参考答案**：三个关键场景：生成更贴近真实面试的题目、让评分有据可依（评分为何是 7 分而非 8 分）、实现难度自适应的追问。

---

## 13. 扩展与改进方向

### 13.1 可扩展的技术方向

| 方向 | 具体内容 | 技术方案 |
|------|---------|---------|
| 语音面试 | 语音输入 → TTS 提问 → STT 回答 | Whisper + Edge TTS |
| 多模态 | 白板编程、架构图绘制评估 | GPT-4V / Claude Vision |
| 代码执行 | 在线运行候选人代码 | Docker Sandbox + Judge0 |
| 多语言 | 支持英文面试 | 多语言 Prompt + 多语言知识库 |
| 监控告警 | Agent 执行追踪 + 异常告警 | LangSmith / LangFuse |
| A/B 评估 | 对比不同 LLM 的面试效果 | 评分一致性分析 |
| 微服务化 | Agent 拆分为独立服务 | FastAPI + RabbitMQ |

### 13.2 可以写在简历上的关键词

基于本项目，你可以合法地在简历中写上：

> **AI/Agent 方向技能关键词**：
> - LangGraph 多 Agent 编排：Supervisor 模式、状态图设计、Agent 权限控制
> - RAG 检索增强生成：ChromaDB 向量库、文档分块策略、多场景检索
> - 混合路由策略：规则引擎 + LLM 决策 + 默认兜底
> - 防御式架构：三层降级（Redis→SQLite→Memory）、JSON 解析容错、循环检测
> - LLM 工程化：限流（滑动窗口）、缓存（TTL-LRU）、重试（指数退避）
> - FastAPI 异步服务：SSE 流式输出、会话管理、文件上传

---

## 附录

### A. 项目文件索引

| 文件 | 行数 | 核心内容 |
|------|------|---------|
| `agents/state.py` | 472 | 状态定义 + 权限 + 循环检测 |
| `agents/supervisor.py` | 1228 | Supervisor 路由 + 图构建 |
| `agents/resume_match_agent.py` | 622 | 简历解析 + 岗位匹配 |
| `agents/interview_agent.py` | 358 | 面试主持 |
| `agents/evaluator_agent.py` | 341 | 回答评估 |
| `agents/report_agent.py` | 229 | 报告生成 |
| `agents/direct_reply_agent.py` | 579 | 通用问答 |
| `agents/utils.py` | 478 | LLMClient/RateLimiter/Cache |
| `agents/redis_client.py` | 345 | Redis 用户信息存储 |
| `web_server.py` | 941 | FastAPI 服务 |
| `agent_core.py` | 1862 | V1 Agent + 工具定义 |
| `agent_graph.py` | 417 | V1 LangGraph 单 Agent 图 |
| `knowledge_base.py` | 776 | RAG 知识库 |
| `vector_db.py` | 421 | Chroma + 向量嵌入 |
| `resume_parser.py` | 637 | 简历文件解析 |
| `sqlite_db.py` | 761 | SQLite DB CRUD |
| `static/index.html` | 1191 | 前端 SPA |

### B. 学习路径建议

```
第 1 天：理解整体架构
  → 读 README.md
  → 读 agents/state.py（理解状态定义 + 权限系统）
  → 画出 Agent 交互图

第 2 天：深入路由系统
  → 读 agents/supervisor.py
  → 理解 9 步处理流程
  → 理解混合路由的三层逻辑

第 3 天：逐个 Agent 击破
  → resume_match_agent → interview_agent → evaluator_agent
  → 理解每个 Agent 的内部状态机

第 4 天：工程实践
  → 持久化策略 (ChatStore, Checkpointer)
  → 容错机制 (with_permissions, JSON 容错, 循环检测)
  → 工具函数 (utils.py 三大组件)

第 5 天：串联 + 面试准备
  → 跟踪一次完整面试的 state 变化
  → 准备技术问答（第 12 节）
  → 运行项目，实际体验
```

---

> 📝 **文档版本**：v1.0
> 📅 **生成日期**：2026-06-28
> 🎯 **目的**：帮助开发者快速掌握项目全貌，为 Agent/AI 方向求职做准备
