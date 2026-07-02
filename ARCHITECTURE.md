# 🏗️ AI 面试官 — 架构深度解析

> 深入分析多 Agent 系统的架构设计决策、数据流、容错机制与工程实践

---

## 目录

1. [架构演进：从 V1 到 V2](#1-架构演进从-v1-到-v2)
2. [LangGraph 图结构详解](#2-langgraph-图结构详解)
3. [状态管理设计模式](#3-状态管理设计模式)
4. [路由系统深入分析](#4-路由系统深入分析)
5. [Agent 间通信机制](#5-agent-间通信机制)
6. [并发与线程安全](#6-并发与线程安全)
7. [LLM 调用优化策略](#7-llm-调用优化策略)
8. [持久化架构细节](#8-持久化架构细节)

---

## 1. 架构演进：从 V1 到 V2

### 1.1 V1：单 Agent + Tool Calling

```
文件：agent_graph.py + agent_core.py

架构：
    User → LLM(带Tools) → Tool执行 → LLM → 回复
    单 Agent + 工具调用模式

问题：
    ✗ 工具越来越多，LLM 选择工具越来越不稳定
    ✗ 逻辑混杂在一个节点中，难以维护
    ✗ 没有权限控制，任何 tool 可以改任何状态
    ✗ 缺少流程控制（面试流程被硬编码）
```

**V1 的核心代码** (`agent_graph.py:init_llm()`)：

```python
# 单 Agent 图
llm = ChatOpenAI(model="deepseek-chat", temperature=0.7)
tools = [parse_resume_tool, evaluate_answer_tool, search_local_jobs, ...]
llm_with_tools = llm.bind_tools(tools)

graph = StateGraph(InterviewState)
graph.add_node("interviewer", interviewer_node)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("interviewer", route_by_tool_call, {"tools": "tools", "end": END})
graph.add_edge("tools", "interviewer")
```

### 1.2 V2：多 Agent + Supervisor

```
文件：agents/ 包

架构：
    User → Supervisor(路由) → Agent A/B/C → Supervisor → ...
    多 Agent 协作 + 星型拓扑

优势：
    ✓ 每个 Agent 职责清晰，各司其职
    ✓ 权限系统保证状态安全
    ✓ 路由规则可控，出问题能定位到具体 Agent
    ✓ 扩展新功能只需加新 Agent，不用改旧代码
    ✓ 混合路由（规则 + LLM）平衡成本与灵活性
```

**V2 的核心代码** (`supervisor.py:build_multi_agent_graph()`)：

```python
def build_multi_agent_graph():
    graph = StateGraph(MultiAgentState)

    # 注册节点
    graph.add_node("supervisor", with_permissions("supervisor")(supervisor_node))
    graph.add_node("resume_match", with_permissions("resume_match")(resume_match_node))
    graph.add_node("interview", with_permissions("interview")(interview_node))
    graph.add_node("evaluate", with_permissions("evaluate")(evaluate_node))
    graph.add_node("report", with_permissions("report")(report_node))
    graph.add_node("direct_reply", with_permissions("direct_reply")(direct_reply_node))

    # 星型拓扑：所有节点都回到 supervisor
    graph.add_edge(START, "supervisor")
    for agent in ["resume_match", "interview", "evaluate", "report", "direct_reply"]:
        graph.add_edge(agent, "supervisor")

    # 条件边：supervisor → 下一个 agent 或 END
    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "resume_match": "resume_match",
        "interview": "interview",
        "evaluate": "evaluate",
        "report": "report",
        "direct_reply": "direct_reply",
        "END": END,
    })

    checkpointer = create_checkpointer()
    return graph.compile(checkpointer=checkpointer)
```

### 1.3 架构选择的关键权衡

| 维度 | V1 (Single + Tools) | V2 (Multi-Agent + Supervisor) |
|------|---------------------|-------------------------------|
| 复杂度 | 低 | 中高 |
| 可维护性 | 低（逻辑耦合） | 高（职责分离） |
| 可扩展性 | 差（加功能 = 改核心） | 好（加 Agent 即可） |
| 调试难度 | 难（LLM 黑盒选择 tools） | 中（可追踪每个 Agent） |
| Token 消耗 | 低（单 LLM 调用） | 中（Supervisor 也消耗 token） |
| 可靠性 | 中（tool 调用可能出错） | 高（多级容错） |

---

## 2. LangGraph 图结构详解

### 2.1 核心概念

```
StateGraph: 有向图，节点处理状态，边定义流转
  ├── Node (节点)：接收 state → 处理 → 返回 state 变更
  ├── Edge (边)：普通边（固定路由） / 条件边（动态路由）
  ├── State (状态)：TypedDict，在图执行过程中累积
  └── Checkpointer: 每次 step 后自动保存状态快照

执行模型：
  graph.invoke(initial_state, config)
    → Step 1: START → supervisor
    → Step 2: supervisor → resume_match (条件边路由)
    → Step 3: resume_match → supervisor (固定边)
    → Step 4: supervisor → interview (条件边路由)
    → ... 直到 END
    → 返回最终 state
```

### 2.2 消息追加机制

```python
# MultiAgentState 中 messages 字段的特殊 reducer
messages: Annotated[list, add_messages]

# add_messages 是 LangGraph 的 operator.add
# 意味着：当节点返回 {"messages": [new_msg]} 时
# 新消息被追加到现有列表，而非替换

# 示例：
state = {"messages": [HumanMessage("你好")]}
# Agent 返回 {"messages": [AIMessage("你好！请上传简历。")]}
# 新 state = {"messages": [
#     HumanMessage("你好"),
#     AIMessage("你好！请上传简历。")
# ]}
```

### 2.3 中断与恢复 (Human-in-the-Loop)

```python
# 面试题发出后，系统需要等待用户输入
# 方案：当 interview Agent 发出问题后，
# Supervisor 检测到 waiting_for_user == True → 返回 END

# 用户下次发消息时：
# graph.ainvoke(
#     {"messages": [HumanMessage(user_answer)]},
#     {"configurable": {"thread_id": session_id}}
# )
# LangGraph 自动从上次中断处恢复，继续执行
```

---

## 3. 状态管理设计模式

### 3.1 状态渐进式构建 (Progressive State Building)

```
面试流程中的状态演变：

初始: {}
  ↓ resume_match 第1次: +resume_data
  ↓ resume_match 第2次: +job_matches
  ↓ resume_match 第3次: +selected_job
  ↓ interview 第1次:    +interview_questions[5], +current_question_idx=0
  ↓ evaluate 第1次:     +answers[1], +last_answer_score
  ↓ interview 第2次:    current_question_idx=1
  ↓ evaluate 第2次:     +answers[2], +last_answer_score
  ↓ ... 循环 ...
  ↓ report:             +final_report, +overall_score, 清理面试字段
```

这种模式的优点：**状态在每个 step 都是完整且一致的**，即使中途崩溃也能从 checkpoint 恢复。

### 3.2 字段分组与可见性

```python
# 按功能分组 + 按可见性分层

# 层 1: 框架层 (LangGraph 管理)
messages              # 所有 Agent 都可以追加

# 层 2: 控制层 (Supervisor 独占)
next_agent, _agent_history, _total_steps, _error_count, _last_error
waiting_for_user, _agent_signal

# 层 3: 业务层 (各 Agent 分治)
resume_data, job_matches, selected_job         # → resume_match
interview_questions, current_question_idx       # → interview
answers, evaluations, last_answer_score        # → evaluate
final_report, overall_score                     # → report
```

### 3.3 内部信号机制 (_agent_signal)

```
这是一个精巧的设计：

Agent 不能直接指定下一个 Agent（那是 Supervisor 的职责），
但可以通过 _agent_signal 给 Supervisor 发出"建议信号"：

evaluator Agent:
  全部答完 → _agent_signal = ""       → Supervisor 路由到 report
  还有题目 → _agent_signal = "interview" → Supervisor 路由到 interview

resume_match Agent:
  岗位选定 → _agent_signal = "interview" → Supervisor 路由到 interview

Supervisor 在 step 1 检查 agent_signal 后立即清空，
确保信号只生效一次。
```

---

## 4. 路由系统深入分析

### 4.1 规则路由的优先级设计

```
从高到低的优先级：

P0: 面试进行中 (最高优先级，不可中断)
    → 用户的任何回答都走 evaluate
    → 只有明确"结束面试"才跳转
    → 防止回答内容中的关键词误触发路由

P1: 明确结束意图
    → "结束面试" "我不想面了" 等

P2-P5: 意图驱动的路由
    → 简历意图 → resume_match
    → 岗位匹配意图 → resume_match
    → 个人信息查询 → direct_reply
    → 开始面试意图 → interview
    → 通用问答 → direct_reply

P6: 阶段驱动路由 (兜底)
    → 没有 resume_data → resume_match
    → 没有 selected_job → resume_match
    → 没有 questions → interview
    → questions 用完 + 没有 report → report
    → 有 report → FINISH
```

### 4.2 面试进行中的特殊保护

这是最精妙的路由逻辑：

```python
# 面试进行中的路由保护
current_idx = state.get("current_question_idx", 0)
if questions and current_idx < len(questions):
    if last_user_msg and len(last_user_msg.strip()) > 0:
        # 例外：用户明确说"结束面试"才跳转
        # 必须是短消息 + 严格匹配，防止误判
        if _is_explicit_end_interview(last_user_msg):
            return "report"
        # 否则一律视为回答 → 走评分
        return "evaluate"
```

为什么这很重要？

```
场景：面试官问"请描述你的项目经验"
用户回答："...这个项目我从简历投递到最终交付全程负责..."

如果没有保护：
  → "简历"关键词触发 resume_match 路由
  → 面试中断，用户困惑

有了保护：
  → 面试中，用户输入 = 回答
  → 正确路由到 evaluate
```

### 4.3 LLM 路由的容错

```python
def _llm_based_route(state: dict) -> Optional[str]:
    llm = _get_supervisor_llm()
    if llm is None:
        return None  # LLM 初始化失败 → 回退到默认路由

    for attempt in range(3):  # 最多 3 次重试
        try:
            response = llm.invoke([
                SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=state_summary),
            ])
            agent_name = response.content.strip().lower()

            # 验证合法性
            if agent_name in VALID_AGENTS:
                return agent_name
            else:
                logger.warning(f"LLM returned invalid agent: '{agent_name}', retry {attempt+1}/3")
        except Exception as e:
            logger.warning(f"LLM routing failed (attempt {attempt+1}/3): {e}")

    return None  # 3 次都失败 → 回退到默认路由
```

---

## 5. Agent 间通信机制

### 5.1 显式通信 vs 隐式通信

```
显式通信（本项目主要方式）：
  通过 state 字段传递信息
  Agent A 写入 state.X → Agent B 读取 state.X

隐式通信（辅助方式）：
  通过消息历史传递
  Agent A 在 messages 中追加消息 → Agent B 从 messages 中读取
```

### 5.2 Evaluator → Interview 的难度传递

```python
# Evaluator 设置难度
# evaluator_agent.py
return {
    "next_difficulty": "hard",    # ← 写入 state
    "last_answer_score": 9,
    "_agent_signal": "interview", # ← 建议 Supervisor 路由到 interview
}

# Interview 读取难度
# interview_agent.py
difficulty = state.get("next_difficulty", "normal")
if difficulty == "hard":
    prompt += "请出一道更有挑战性的题目..."
elif difficulty == "easy":
    prompt += "请出一道基础巩固题目..."
```

### 5.3 消息的注入模式

```python
# Agent 通过返回 {"messages": [AIMessage(...)]} 来与用户对话
# 所有 Agent 共享同一个消息流

# resume_match 的输出消息
AIMessage("已解析您的简历：张三，3年Python经验...请选择岗位：\n1. Python后端\n2. 全栈\n3. 数据工程")

# interview 的输出消息
AIMessage("【第1题/共5题】请描述你在项目中是如何使用Redis做缓存的？")

# evaluate 的输出消息
AIMessage("评分：8/10\n优点：...\n不足：...\n改进建议：...")
```

---

## 6. 并发与线程安全

### 6.1 FastAPI 异步与 LangGraph 同步的桥接

```python
# web_server.py 中的关键处理

# 问题：FastAPI 是异步的，LangGraph 是同步的
# 方案：asyncio.to_thread() + contextvars

import contextvars
session_var = contextvars.ContextVar("session_id")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    session_var.set(session_id)  # 通过 contextvar 传递给同步函数

    # 在线程池中执行同步的 LangGraph
    result = await asyncio.to_thread(
        graph.ainvoke,  # 或 graph.invoke()
        {"messages": [HumanMessage(content=request.message)]},
        {"configurable": {"thread_id": session_id}}
    )

    return result
```

### 6.2 全局单例的线程安全

```python
# agents/utils.py 中的单例模式

_llm_client: Optional[LLMClient] = None
_rate_limiter: Optional[RateLimiter] = None
_response_cache: Optional[ResponseCache] = None
_lock = threading.Lock()

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        with _lock:
            if _llm_client is None:  # Double-checked locking
                _llm_client = LLMClient()
    return _llm_client
```

### 6.3 滑动窗口限流

```python
class RateLimiter:
    def __init__(self, max_requests: int = 2, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._windows: Dict[str, deque] = {}  # thread_id → timestamps deque
        self._lock = threading.Lock()

    def check(self, thread_id: str) -> bool:
        now = time.time()
        with self._lock:
            if thread_id not in self._windows:
                self._windows[thread_id] = deque()
            window = self._windows[thread_id]

            # 清除窗口外的记录
            while window and now - window[0] > self.window:
                window.popleft()

            # 判断是否超限
            if len(window) >= self.max_requests:
                return False

            window.append(now)
            return True
```

---

## 7. LLM 调用优化策略

### 7.1 缓存策略

```python
class ResponseCache:
    """TTL + LRU 双层淘汰"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        timestamp, value = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None
        # LRU: move to end (most recently used)
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # 淘汰最久未使用
        self._cache[key] = (time.time(), value)
```

### 7.2 统一调用入口 safe_llm_call

```python
def safe_llm_call(
    thread_id: str,
    messages: list,
    cache_key: str = None,
    model: str = "deepseek-chat",
    temperature: float = 0.7,
) -> Optional[str]:
    """
    统一的 LLM 调用入口，整合限流、缓存、重试

    流程：
    1. 检查频控 → 超限则等待
    2. 检查缓存 → 命中则直接返回
    3. 调用 LLM (自动重试 3 次)
    4. 写入缓存
    5. 返回结果
    """
    limiter = get_rate_limiter()
    if not limiter.check(thread_id):
        time.sleep(0.5)  # 等待后重试
        if not limiter.check(thread_id):
            raise Exception("Rate limit exceeded")

    if cache_key:
        cache = get_response_cache()
        cached = cache.get(cache_key)
        if cached:
            return cached

    client = get_llm_client()
    result = client.call(messages, model=model, temperature=temperature)

    if cache_key and result:
        cache.set(cache_key, result)

    return result
```

### 7.3 多模型支持

```python
# 系统中不同的模型用途：
MODELS = {
    "supervisor": "deepseek-chat",     # 路由：轻量，temperature=0
    "interview": "deepseek-v4-pro",    # 出题：需要推理能力
    "evaluator": "deepseek-v4-pro",    # 评分：需要推理能力
    "direct_reply": "deepseek-chat",   # 闲聊：轻量即可
    "embedding": "text-embedding-v2",  # 向量：DashScope 专用
}
```

---

## 8. 持久化架构细节

### 8.1 为什么需要检查点 (Checkpoint)？

```
没有 Checkpoint 的问题：
  用户面试到第 3 题，服务重启 → 状态全丢 → 从头开始

有了 Checkpoint：
  每个 step 后状态自动保存
  服务重启 → 用户继续第 3 题 → 无感知

LangGraph Checkpoint 的优势：
  - 框架级别支持，不需要手写序列化
  - 自动在每次节点执行后保存
  - 支持时间旅行（可回溯历史状态）
  - 支持多种后端（Redis/SQLite/Postgres/Memory）
```

### 8.2 Redis 数据结构设计

```
ChatStore 的 Redis 结构：

chat:sessions → JSON Array
  [
    {"id": "abc123", "title": "张三的面试", "created_at": "..."},
    {"id": "def456", "title": "李四的面试", "created_at": "..."},
  ]

chat:history:{session_id} → JSON Array
  [
    {"role": "user", "content": "我要面试Python后端", "timestamp": "..."},
    {"role": "ai", "content": "好的，请上传简历", "timestamp": "..."},
    ...
  ]

UserInfoStore 的 Redis 结构：

user:{thread_id} → Hash
  {
    "name": "张三",
    "email": "zhangsan@example.com",
    "phone": "13800138000",
    "school": "北京大学",
    "skills": "Python, FastAPI, Redis",
    "summary": "3年Python后端开发经验...",
    "last_job": "python_backend"
  }
  TTL: 7天
```

### 8.3 SQLite 表结构

```sql
-- sqlite_db.py 定义的表结构

CREATE TABLE resumes (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    raw_text TEXT,          -- 原始简历文本
    structured_data TEXT,   -- JSON 结构化数据
    file_name TEXT,
    created_at TIMESTAMP
);

CREATE TABLE interviews (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    job_title TEXT,
    status TEXT,            -- "in_progress" | "completed"
    created_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE qa_records (
    id TEXT PRIMARY KEY,
    interview_id TEXT,
    question_number INTEGER,
    question TEXT,
    answer TEXT,
    score REAL,
    evaluation TEXT,
    created_at TIMESTAMP
);

CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    interview_id TEXT,
    overall_score REAL,
    report_text TEXT,
    recommendation TEXT,    -- "推荐录用" | "可考虑" | "不推荐"
    created_at TIMESTAMP
);
```

---

> 📝 深入理解了这些架构设计，你就能在面试中自信地讨论多 Agent 系统的方方面面
