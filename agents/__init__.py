#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 Agent 系统包 (agents/)
========================

基于 LangGraph Supervisor 模式的多 Agent 面试系统。

架构：
    Supervisor (路由器)
       │
       ├── ResumeMatch Agent (简历解析 + 岗位匹配)
       ├── Interview Agent   (面试主持)
       ├── Evaluator Agent   (回答评分)
       └── Report Agent      (报告生成)

核心设计原则：
    1. Agent 间状态隔离：每个 Agent 只能写自己的字段，其他字段只读
    2. Supervisor 容错：崩溃时有默认路由兜底
    3. 循环检测：同一 Agent 连续 3 次 + 总步数 > 20 → 强制结束
    4. 持久化检查点：优先 Redis，降级 SQLite
"""

from agents.state import (
    MultiAgentState,
    AGENT_PERMISSIONS,
    create_state_view,
    with_permissions,
    check_and_enforce_loop_limits,
)

from agents.supervisor import (
    supervisor_node,
    route_after_supervisor,
    create_checkpointer,
    build_multi_agent_graph,
)

# ---- 真实 Agent 节点 ----
from agents.resume_match_agent import resume_match_node
from agents.interview_agent import interview_node
from agents.evaluator_agent import evaluator_node
from agents.report_agent import report_node
from agents.direct_reply_agent import direct_reply_node

__all__ = [
    # State & Permissions
    "MultiAgentState",
    "AGENT_PERMISSIONS",
    "create_state_view",
    "with_permissions",
    "check_and_enforce_loop_limits",
    # Supervisor
    "supervisor_node",
    "route_after_supervisor",
    "create_checkpointer",
    "build_multi_agent_graph",
    # Real Agents
    "resume_match_node",
    "interview_node",
    "evaluator_node",
    "report_node",
    "direct_reply_node",
]
