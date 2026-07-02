#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 Agent 共享状态 & 权限控制系统 (state.py)
===========================================

核心设计：
1. MultiAgentState — 所有 Agent 共享的状态定义
2. AGENT_PERMISSIONS — 每个 Agent 的可写字段白名单
3. with_permissions() — 装饰器：自动拦截非法写入 + 异常兜底
4. check_and_enforce_loop_limits() — 循环检测（在条件边中调用）

权限规则：
    ┌─────────────────┬────────────────────────────────────────┐
    │ Agent            │ 可写字段                               │
    ├─────────────────┼────────────────────────────────────────┤
    │ supervisor       │ next_agent, _agent_history, _total_   │
    │                  │ steps, _error_count, _last_error      │
    │ resume_match     │ resume_data, job_matches, selected_job│
    │ interview        │ interview_questions, current_question_ │
    │                  │ idx, interview_stage                  │
    │ evaluate         │ evaluations, answers, last_answer_    │
    │                  │ score, current_question_idx           │
    │ report           │ final_report, overall_score           │
    │ (all)            │ messages (共享追加，所有人可读)        │
    └─────────────────┴────────────────────────────────────────┘

禁止行为：
    - Agent A 写入 Agent B 的字段 → 日志警告 + 静默丢弃
    - Agent 崩溃 → 记录错误，不清空已有状态
    - 无限循环 → 连续检测 + 总步数上限双重保护
"""

import os
import operator
import functools
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Set
from langgraph.graph import END, add_messages

logger = logging.getLogger("multi-agent.state")

# =============================================================================
# 第一部分：MultiAgentState 定义
# =============================================================================

class MultiAgentState(TypedDict, total=False):
    """
    多 Agent 共享状态

    所有 Agent 节点都接收和返回此状态（或其子集）。
    通过 with_permissions 装饰器确保每个 Agent 只能写白名单字段。
    """

    # ---- 消息历史（LangGraph 使用 add_messages reducer 自动追加） ----
    messages: Annotated[list, add_messages]  # 对话历史（所有 Agent 共享）

    # ---- Supervisor 路由控制 ----
    next_agent: str              # 下一步路由目标："resume_match"|"interview"|"evaluate"|"report"|"FINISH"

    # ---- 内部追踪字段（Supervisor 专用） ----
    _agent_history: List[str]    # Agent 调用历史，如 ["resume_match","interview","evaluate","interview",...]
    _total_steps: int            # 总执行步数（每次通过 supervisor 时 +1）
    _error_count: Dict[str, int] # 每个 Agent 的错误计数，如 {"resume_match": 1, "evaluate": 2}
    _last_error: str             # 最近的错误信息
    _supervisor_retries: int     # Supervisor 自身的重试次数
    _asked_for_resume: bool      # resume_match 已询问过用户提供简历路径（防止重复调度）
    _agent_signal: str           # Agent 给 Supervisor 的信号（如 "FINISH"），Supervisor 优先读取并清空
    waiting_for_user: bool       # Agent 正在等待用户输入，Supervisor 应停止调度

    # ---- 业务状态：简历 & 匹配（ResumeMatch Agent 专用） ----
    resume_data: dict            # 解析后的简历结构化数据
    job_matches: list            # 岗位匹配结果列表
    selected_job: dict           # 用户选择的岗位

    # ---- 业务状态：面试（Interview + Evaluate Agent 专用） ----
    interview_questions: list    # 生成的面试题列表（5 题）
    current_question_idx: int    # 当前题目索引（0-based）
    answers: list                # 已回答列表 [{"question":..., "answer":..., "evaluation":..., "score":...}, ...]
    evaluations: list            # 评分记录列表
    last_answer_score: float     # 最近一次回答的评分（1-10）

    # ---- 业务状态：收尾（Report Agent 专用） ----
    final_report: str            # 最终面试报告文本
    overall_score: float         # 综合评分

    # ---- 面试流程控制 ----
    interview_stage: str         # "idle"|"parsing"|"matching"|"interviewing"|"evaluating"|"reporting"|"done"
    next_difficulty: str         # 下一题的难度等级："easy"|"normal"|"hard"（evaluator → interview 传递）

    # ---- 通用问答（direct_reply Agent 专用） ----
    _direct_reply_count: int     # 通用问答轮数计数器（防止无限循环，上限 5 轮）


# =============================================================================
# 第二部分：Agent 权限白名单
# =============================================================================

AGENT_PERMISSIONS: Dict[str, Set[str]] = {
    # Supervisor：控制字段 + 追踪字段
    "supervisor": {
        "next_agent",
        "_agent_history",
        "_total_steps",
        "_error_count",
        "_last_error",
        "_supervisor_retries",
        "_agent_signal",
        "waiting_for_user",
        "interview_stage",
    },

    # ResumeMatch Agent：简历解析 + 岗位匹配
    "resume_match": {
        "resume_data",
        "job_matches",
        "selected_job",
        "interview_stage",
        "_asked_for_resume",
        "_agent_signal",
        "waiting_for_user",
    },

    # Interview Agent：面试主持
    "interview": {
        "interview_questions",
        "current_question_idx",
        "interview_stage",
        "_agent_signal",
        "next_difficulty",
    },

    # Evaluator Agent：回答评分
    "evaluate": {
        "evaluations",
        "answers",
        "last_answer_score",
        "current_question_idx",
        "interview_stage",
        "_agent_signal",
        "next_difficulty",
    },

    # Report Agent：报告生成 + 面试状态清理
    "report": {
        "final_report",
        "overall_score",
        "interview_stage",
        "_agent_signal",
        # 清空面试状态（防止报告后误入面试流程）
        "interview_questions",
        "current_question_idx",
        "answers",
        "last_answer_score",
        "waiting_for_user",
    },

    # DirectReply Agent：通用问答（无简历时处理技术问题/闲聊）
    "direct_reply": {
        "_agent_signal",
        "_direct_reply_count",
    },
}

# messages 字段是 LangGraph 管理的（operator.add 追加），任何 Agent 都可以通过
# 返回 {"messages": [...]} 来追加消息，这是框架级别的行为，不在此权限系统控制。


# =============================================================================
# 第三部分：StateView — 只读状态视图
# =============================================================================

class StateView:
    """
    只读状态视图 —— 传给 Agent 节点函数，防止意外修改非己字段。

    使用方式：
        view = create_state_view(state, "resume_match")
        resume = view.read("resume_data")          # ✅ 可以读
        view.read("interview_questions")           # ✅ 可以读（读不限制）
        # 注意：view 只提供读取，写入通过节点函数返回值控制
    """

    def __init__(self, state: dict, agent_name: str):
        self._state = state
        self._agent = agent_name
        self._allowed = AGENT_PERMISSIONS.get(agent_name, set())

    def read(self, key: str, default=None):
        """读取任意状态字段（所有 Agent 都可读）"""
        return self._state.get(key, default)

    def can_write(self, key: str) -> bool:
        """检查是否有权写入某字段"""
        return key in self._allowed or key == "messages"

    def get_allowed_keys(self) -> Set[str]:
        """返回当前 Agent 可写的字段集合"""
        return self._allowed.copy()

    def __repr__(self):
        return f"StateView(agent='{self._agent}', allowed={self._allowed})"


def create_state_view(state: dict, agent_name: str) -> StateView:
    """创建只读状态视图的工厂函数"""
    return StateView(state, agent_name)


# =============================================================================
# 第四部分：with_permissions 装饰器 — 写入拦截 + 异常兜底
# =============================================================================

def with_permissions(agent_name: str):
    """
    Agent 节点装饰器：自动过滤非法写入 + 异常兜底。

    功能：
    1. 节点正常返回 → 过滤掉不在白名单中的字段（静默丢弃 + 日志警告）
    2. 节点抛出异常 → 记录错误，不清空已有状态，返回错误信息
    3. 自动递增 _error_count

    使用方式：
        @with_permissions("resume_match")
        def resume_match_node(state):
            # state 是原始 dict（可读所有字段）
            # 返回值只有白名单字段会被写入
            return {"resume_data": {...}, "job_matches": [...]}
    """

    def decorator(node_fn):
        @functools.wraps(node_fn)
        def wrapper(state: dict) -> dict:
            allowed = AGENT_PERMISSIONS.get(agent_name, set())

            try:
                # 执行 Agent 节点
                result = node_fn(state)

                if not isinstance(result, dict):
                    return result

                # ---- 权限过滤：只保留白名单字段 ----
                filtered = {}
                blocked = []

                for key, value in result.items():
                    if key == "messages":
                        # messages 是框架管理字段，允许通过
                        filtered[key] = value
                    elif key in allowed:
                        filtered[key] = value
                    else:
                        blocked.append(key)

                if blocked:
                    logger.warning(
                        f"[PERMISSION] Agent '{agent_name}' 试图写入未授权字段: {blocked} | "
                        f"已静默丢弃。允许写入: {sorted(allowed)}"
                    )

                return filtered

            except Exception as e:
                # ---- 异常兜底：记录错误，不清空状态 ----
                error_msg = f"Agent '{agent_name}' 异常: {type(e).__name__}: {str(e)}"
                logger.error(f"[EXCEPTION] {error_msg}")

                # 递增该 Agent 的错误计数
                error_count = state.get("_error_count", {}).copy()
                error_count[agent_name] = error_count.get(agent_name, 0) + 1

                return {
                    "_last_error": error_msg,
                    "_error_count": error_count,
                    # 关键：不返回任何业务字段，避免污染状态
                }

        return wrapper

    return decorator


# =============================================================================
# 第五部分：循环检测 & 强制终止（在条件边中调用）
# =============================================================================

def check_and_enforce_loop_limits(state: dict, next_agent: str) -> Optional[str]:
    """
    循环检测 + 强制终止逻辑。

    在 Supervisor 的条件边函数中调用，检查是否触发终止条件。

    检测规则（按优先级）：
    1. 同一非 Supervisor Agent 连续出现 >= 3 次 → 强制 END
    2. 总步数 > 20 → 强制 END（主保护）
    3. 总步数 > 50 → 强制 END（Supervisor 兜底）

    参数：
        state: 当前状态
        next_agent: Supervisor 决定的下一个 Agent

    返回：
        - None: 正常，不拦截
        - END: 触发终止条件
    """
    agent_history = list(state.get("_agent_history", []))
    total_steps = state.get("_total_steps", 0)

    # Rule 1: 同一 Agent 连续 5 次才拦截
    # resume_match 需要最多 4 次合法调用（parse + match + show_prompt + select）
    # 如果用户重新匹配岗位，可能需要第 5 次（re-match）
    # 排除 Supervisor（它不执行业务）和 FINISH（它是终端信号，不是 Agent）
    _IGNORED_AGENTS = {"supervisor", "FINISH"}

    if len(agent_history) >= 5:
        last_five = agent_history[-5:]
        if len(set(last_five)) == 1:
            repeated_agent = last_five[0]
            if repeated_agent not in _IGNORED_AGENTS:
                logger.warning(
                    f"[LOOP] Agent '{repeated_agent}' 连续出现 5 次 (history: {agent_history}) → 强制 END"
                )
                return END

    # Pre-emptive check: would adding next_agent make it 6+ consecutive?
    # (Allow 5 — resume_match needs up to 4 calls: parse + match + show + select,
    #  plus 1 buffer for re-match)
    if len(agent_history) >= 5:
        last_five = agent_history[-5:]
        if len(set(last_five)) == 1 and last_five[0] == next_agent and next_agent not in _IGNORED_AGENTS:
            logger.warning(
                f"[LOOP] Agent '{next_agent}' 即将连续第 6 次 (history: {agent_history}) → 强制 END"
            )
            return END

    # Rule 2: 总步数 > 20
    if total_steps > 50:
        logger.warning(
            f"[LOOP] 总步数 {total_steps} > 50 → 强制 END. "
            f"Agent 历史: {agent_history}"
        )
        return END

    return None


# =============================================================================
# 第六部分：状态工具函数
# =============================================================================

def format_state_summary(state: dict) -> str:
    """
    生成人类可读的状态摘要（调试用）

    参数：
        state: 当前状态字典

    返回：
        str: 格式化的摘要文本
    """
    lines = [
        "=" * 50,
        "MultiAgentState Summary",
        "=" * 50,
        f"  Total Steps:     {state.get('_total_steps', 0)}",
        f"  Next Agent:      {state.get('next_agent', 'N/A')}",
        f"  Agent History:   {state.get('_agent_history', [])}",
        f"  Error Count:     {state.get('_error_count', {})}",
        f"  Last Error:      {state.get('_last_error', 'None')}",
        f"  Stage:           {state.get('interview_stage', 'N/A')}",
        f"  Resume Parsed:   {'Yes' if state.get('resume_data') else 'No'}",
        f"  Job Matches:     {len(state.get('job_matches', []))}",
        f"  Job Selected:    {'Yes' if state.get('selected_job') else 'No'}",
        f"  Questions:       {len(state.get('interview_questions', []))}",
        f"  Answers:         {len(state.get('answers', []))}",
        f"  Final Report:    {'Yes' if state.get('final_report') else 'No'}",
        f"  Messages Count:  {len(state.get('messages', []))}",
        "=" * 50,
    ]
    return "\n".join(lines)


# =============================================================================
# 第七部分：自检（直接运行此文件）
# =============================================================================

if __name__ == "__main__":
    print("=== MultiAgentState & Permissions Self-Check ===\n")

    # Test 1: Permission system
    print("[1] Testing permission system...")
    test_state = {"resume_data": {"name": "Test"}, "interview_questions": []}
    view = create_state_view(test_state, "resume_match")
    assert view.read("resume_data") == {"name": "Test"}, "Read failed"
    assert view.read("interview_questions") == [], "Cross-read should be allowed"
    assert view.can_write("resume_data") == True, "Should be able to write resume_data"
    assert view.can_write("interview_questions") == False, "Should NOT write interview_questions"
    print("    [OK] Permission read/write checks passed")

    # Test 2: with_permissions decorator
    print("[2] Testing with_permissions decorator...")

    @with_permissions("resume_match")
    def test_node_ok(state):
        return {"resume_data": {"name": "OK"}, "interview_questions": ["bad"]}  # interview_questions should be filtered

    @with_permissions("resume_match")
    def test_node_crash(state):
        raise ValueError("Simulated crash")

    result_ok = test_node_ok({})
    assert "resume_data" in result_ok, "resume_data should be kept"
    assert "interview_questions" not in result_ok, "interview_questions should be filtered"
    print("    [OK] Write filtering works correctly")

    result_crash = test_node_crash({"_error_count": {}})
    assert "_last_error" in result_crash, "Should record error"
    assert "Simulated crash" in result_crash["_last_error"], "Error message should be preserved"
    assert "resume_data" not in result_crash, "Should NOT write business data on crash"
    print("    [OK] Exception handling works correctly")

    # Test 3: Loop detection
    print("[3] Testing loop detection...")

    # Test 3a: 5 consecutive same agents → END
    state_loop = {
        "_agent_history": ["interview", "interview", "interview", "interview", "interview"],
        "_total_steps": 5,
    }
    result = check_and_enforce_loop_limits(state_loop, "interview")
    assert result == END, f"5-in-a-row should trigger END, got {result}"
    print("    [OK] 5-in-a-row loop detection works")

    # Test 3b: Pre-emptive check: 5 consecutive + next same → 6th → END
    state_loop2 = {
        "_agent_history": ["resume_match", "resume_match", "resume_match", "resume_match", "resume_match"],
        "_total_steps": 5,
    }
    result = check_and_enforce_loop_limits(state_loop2, "resume_match")
    assert result == END, f"Pre-emptive 6th should trigger END, got {result}"
    print("    [OK] Pre-emptive loop detection works")

    # Test 3c: 4 consecutive → allowed (resume_match needs 4 calls)
    state_four = {
        "_agent_history": ["resume_match", "resume_match", "resume_match", "resume_match"],
        "_total_steps": 4,
    }
    result = check_and_enforce_loop_limits(state_four, "interview")
    assert result is None, f"4-in-a-row should be allowed, got {result}"
    print("    [OK] 4-in-a-row allowed")

    # Test 3d: Total steps > 50
    state_many = {"_agent_history": ["a"] * 51, "_total_steps": 51}
    result2 = check_and_enforce_loop_limits(state_many, "evaluate")
    assert result2 == END, f">50 steps should trigger END, got {result2}"
    print("    [OK] Total steps > 50 detection works")

    # Test 3e: Normal case → no block
    state_normal = {"_agent_history": ["resume_match", "interview"], "_total_steps": 2}
    result3 = check_and_enforce_loop_limits(state_normal, "evaluate")
    assert result3 is None, f"Normal case should return None, got {result3}"
    print("    [OK] Normal routing passes loop check")

    # Test 4: State view
    print("[4] Testing state summary...")
    summary = format_state_summary({"_total_steps": 3, "_agent_history": ["a", "b", "c"]})
    assert "3" in summary, "Summary should contain step count"
    print("    [OK] State summary works")

    print("\n=== ALL STATE TESTS PASSED ===")
