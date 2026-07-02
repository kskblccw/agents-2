#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervisor Agent — 多 Agent 系统路由器 (supervisor.py)
========================================================

核心职责：
1. 分析当前状态 → 决定下一步交给哪个 Agent
2. 规则优先（0 token 消耗）+ LLM 兜底（处理模糊场景）
3. 自身崩溃时 → 默认路由兜底，不阻塞流程
4. Agent 失败时 → 重试 3 次 → 降级兜底处理
5. 循环检测 → 在条件边中拦截无限循环

设计原则：
    - Supervisor 不执行具体业务逻辑，只做路由决策
    - 规则能覆盖 80% 的场景，LLM 只处理剩余的 20%
    - 任何异常都不会导致流程卡死

检查点策略：
    优先 Redis（生产环境） → 降级 SQLite（持久化） → 降级 Memory（开发/测试）
"""
from dotenv import load_dotenv
load_dotenv()
import os
import logging
from typing import Optional, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import (
    MultiAgentState,
    check_and_enforce_loop_limits,
    with_permissions,
    format_state_summary,
)

logger = logging.getLogger("multi-agent.supervisor")

# =============================================================================
# 第一部分：Supervisor 系统提示词 & LLM 初始化
# =============================================================================

SUPERVISOR_SYSTEM_PROMPT = """你是多 Agent 面试系统的总调度员（Supervisor）。

你的唯一任务：根据当前对话状态，决定下一步交给哪个 Agent 处理。

可选 Agent 及职责：
- "resume_match" — 简历未解析、岗位未匹配、用户需要修改简历或岗位选择时
- "interview"  — 需要生成面试题、提出下一个问题时
- "evaluate"  — 候选人刚回答了问题，需要评分和反馈时
- "report"    — 所有问题已回答完毕，需要生成最终报告时
- "direct_reply" — 用户询问技术问题、面试建议或一般性聊天时（不需要简历）
- "FINISH"    — 对话可以结束时

路由规则：
1. 用户提到了简历文件 → "resume_match"
2. 简历解析完但没匹配岗位 → "resume_match"
3. 岗位已选但没面试题 → "interview"
4. 用户刚回答完问题 → "evaluate"
5. 所有题目答完 → "report"
6. 报告已生成 → "FINISH"
7. 用户只是聊天/问候 → "FINISH"（直接回复即可）

你必须只回复一个 agent 名称，不要回复其他内容。"""

# 延迟初始化 LLM（避免导入时加载模型）
_supervisor_llm = None


def _get_supervisor_llm():
    """延迟初始化 Supervisor 专用 LLM（轻量模型，路由不需要强推理）"""
    global _supervisor_llm
    if _supervisor_llm is None:
        try:
            from langchain_openai import ChatOpenAI
            _supervisor_llm = ChatOpenAI(
                model=os.getenv("SUPERVISOR_MODEL", "deepseek-chat"),  # 路由用轻量模型即可
                temperature=0.0,  # 路由决策需要确定性
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                request_timeout=30,  # 路由超时设短一点
            )
        except Exception as e:
            logger.warning(f"Supervisor LLM init failed: {e}, will use rule-based routing only")
            return None
    return _supervisor_llm


# =============================================================================
# 第二部分：规则路由 — 覆盖 80% 场景，0 token 消耗
# =============================================================================

def _rule_based_route(state: dict) -> Optional[str]:
    
    """
    基于规则的确定性路由（优先执行，不消耗 LLM token）。

    返回值：
        - str: 确定的 Agent 名称
        - None: 规则无法确定，需要 LLM 决策
    """
    messages = state.get("messages", [])
    resume_data = state.get("resume_data")
    job_matches = state.get("job_matches")
    selected_job = state.get("selected_job")
    questions = state.get("interview_questions", [])
    answers = state.get("answers", [])
    final_report = state.get("final_report")
    error_count = state.get("_error_count", {})

    # === 阶段 0：优先级路由检测（从高到低） ===
    last_user_msg = _get_last_user_message(messages)

    # ---- 优先级 0：正在面试中 → 用户输入直接走评分 ----
    # 如果有面试题且当前索引在范围内，用户的任何输入都应视为回答，
    # 避免回答中的"简历""结束"等词触发误路由
    current_idx = state.get("current_question_idx", 0)
    if questions and current_idx < len(questions):
        if last_user_msg and len(last_user_msg.strip()) > 0:
            # 例外：用户【明确】要求结束面试（短消息且仅含结束关键词）
            # 使用严格匹配，避免"这个项目结束了"被误判
            if _is_explicit_end_interview(last_user_msg):
                print("[DEBUG]   → report (explicit end interview during Q&A)")
                return "report"
            print("[DEBUG]   → evaluate (in interview, user answering Q{}/{})"
                  .format(current_idx + 1, len(questions)))
            return "evaluate"

    # ---- 优先级 1：明确结束面试 → 生成报告 ----
    if last_user_msg and _is_explicit_end_interview(last_user_msg):
        if state.get("interview_questions") or state.get("answers"):
            print("[DEBUG]   → report (end interview)")
            return "report"

    # ---- 优先级 2-7：意图检测（简历上传、岗位匹配、个人信息、开始面试、通用问答） ----
    if last_user_msg:
        is_personal = _has_personal_info_intent(last_user_msg)
        is_general = _has_general_qa_intent(last_user_msg)
        is_resume = _has_resume_intent(last_user_msg)
        is_match_job = _has_match_job_intent(last_user_msg)

        print(f"[DEBUG] _rule_based_route Phase 0:")
        print(f"  last_user_msg: '{last_user_msg[:100]}'")
        print(f"  personal_info: {is_personal}, general_qa: {is_general}, "
              f"resume: {is_resume}, match_job: {is_match_job}")

        # 0a0：纯数字输入 + 有岗位匹配结果 → 选择岗位
        if last_user_msg.strip().isdigit() and state.get("job_matches"):
            print("[DEBUG]   → resume_match (digit selection)")
            return "resume_match"

        # 0a：简历上传意图 → resume_match
        if is_resume:
            print("[DEBUG]   → resume_match (resume intent)")
            return "resume_match"

        # 0a5：岗位匹配意图 → resume_match（需要简历才能匹配）
        if is_match_job:
            if resume_data and resume_data.get("name"):
                print("[DEBUG]   → resume_match (match job + has resume)")
                return "resume_match"
            else:
                # 没有简历 → 也路由到 resume_match，让它提示用户先上传简历
                print("[DEBUG]   → resume_match (match job but no resume)")
                return "resume_match"

        # 0b：个人信息查询 → direct_reply
        if is_personal:
            print("[DEBUG]   → direct_reply (personal info intent)")
            return "direct_reply"

        # 0b5：面试开始意图（必须在通用问答之前，否则"开始面试"会被误判为 general Q&A）
        if last_user_msg and _has_start_interview_intent(last_user_msg):
            if state.get("selected_job"):
                print("[DEBUG]   → interview (start interview + has selected_job)")
                return "interview"
            else:
                # 有开始面试意图但没有选择岗位 → 提示用户先选岗位
                print("[DEBUG]   → resume_match (start interview but no selected_job)")
                return "resume_match"

        # 0c：通用问答/闲聊 → direct_reply
        if is_general:
            print(f"[DEBUG]   → direct_reply (general Q&A)")
            return "direct_reply"

        # 0d：Phase 0 未命中任何规则 → 打印调试信息
        print(f"[DEBUG]   Phase 0 fallthrough: no rule matched for '{last_user_msg[:80]}'")

    # === 阶段 1：简历未解析 → resume_match ===
    if not resume_data or not resume_data.get("name"):
        # 简历未解析且 Phase 0 未命中（非 resume/general/personal）→ 让 resume_match 询问
        if not resume_data:
            return "resume_match"

    # === 阶段 2：岗位未匹配/未选择 → resume_match ===
    # 铁律：如果 Agent 正在等待用户输入（waiting_for_user=True），停止调度
    if state.get("waiting_for_user"):
        return "FINISH"

    if resume_data and (not job_matches or len(job_matches) == 0):
        return "resume_match"

    if job_matches and not selected_job:
        # 用户可能刚回复了选择
        last_user_msg = _get_last_user_message(messages)
        if last_user_msg and _has_job_selection_intent(last_user_msg):
            return "resume_match"  # 让 resume_match agent 处理选择
        return "resume_match"

    # === 阶段 3：面试题未生成 → interview ===
    if selected_job and (not questions or len(questions) == 0):
        # 如果已有报告（面试已完成），只有明确"开始面试"才重启
        if state.get("final_report"):
            last_user_msg = _get_last_user_message(messages)
            if last_user_msg and _has_start_interview_intent(last_user_msg):
                print("[DEBUG]   → interview (restart after report)")
                return "interview"
            # 非开始意图 → 让 Phase 0 处理（greeting/general Q&A）
            return None
        return "interview"

    # === 阶段 4：面试进行中 ===
    if questions and len(questions) > 0:
        answered_count = len(answers)
        total_count = len(questions)

        # 4a：已回答完所有问题 → 生成报告
        if answered_count >= total_count:
            if not final_report:
                return "report"
            return "FINISH"

        # 4a2：用户要求提前结束面试 → 生成报告
        last_user_msg = _get_last_user_message(messages)
        if last_user_msg and _has_end_interview_intent(last_user_msg):
            logger.info("[SUPERVISOR] User requested early termination → report")
            return "report"

        # 4b：用户刚回答问题 → evaluate（但先排除结束意图和岗位选择意图）
        last_msg = messages[-1] if messages else None
        if last_msg and hasattr(last_msg, 'type') and last_msg.type == 'human':
            # 避免：用户选岗位的消息也进入 evaluate
            if not _has_job_selection_intent(last_msg.content if hasattr(last_msg, 'content') else ""):
                return "evaluate"

        # 4c：评分完成后 → interview（问下一题）
        last_ai_msg = _get_last_ai_message(messages)
        if last_ai_msg and _contains_evaluation(last_ai_msg):
            return "interview"

        # 4d：没有面试题但有岗位 → interview
        return "interview"

    # === 阶段 5：某个 Agent 连续失败 3 次 → 降级处理 ===
    for agent, count in error_count.items():
        if count >= 3:
            logger.warning(f"[SUPERVISOR] Agent '{agent}' failed {count} times → fallback")
            if agent == "resume_match":
                # 简历解析失败，尝试让 interview 直接用原始文本
                return "interview"
            elif agent == "evaluate":
                # 评分失败，跳过评分直接下一题
                return "interview"
            elif agent == "interview":
                # 问题生成失败，直接结束
                return "report"
            else:
                return "FINISH"

    # === 规则无法确定 → LLM 决策 ===
    return None


def _default_route(state: dict) -> str:
    """
    Supervisor 崩溃时的默认路由（确定性，永远不会失败）。

    按面试流程阶段依次推进，不依赖 LLM。
    """
    if not state.get("resume_data"):
        return "resume_match"
    if not state.get("job_matches") or not state.get("selected_job"):
        return "resume_match"
    if not state.get("interview_questions"):
        return "interview"

    questions = state.get("interview_questions", [])
    answers = state.get("answers", [])
    if len(answers) >= len(questions):
        if not state.get("final_report"):
            return "report"
        return "FINISH"

    # 默认 → 继续面试
    return "interview"


# =============================================================================
# 第三部分：LLM 路由 — 处理模糊场景
# =============================================================================

def _llm_based_route(state: dict) -> str:
    """
    调用 LLM 进行路由决策（仅在规则无法确定时使用）。
    自带重试逻辑：3 次失败 → 降级为默认路由。
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    # LLM 返回的小写名称 → 标准名称映射（确保与 supervisor_node Step 7 的校验列表一致）
    _AGENT_NAME_MAP = {
        "resume_match": "resume_match",
        "interview": "interview",
        "evaluate": "evaluate",
        "report": "report",
        "finish": "FINISH",
        "direct_reply": "direct_reply",
    }

    llm = _get_supervisor_llm()
    if llm is None:
        return _default_route(state)

    # 构建上下文消息
    context = _build_routing_context(state)
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ]

    # 重试 3 次
    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            raw = response.content.strip().lower()

            # 解析 LLM 输出（大小写不敏感匹配 → 标准名称）
            for lower_name, std_name in _AGENT_NAME_MAP.items():
                if lower_name in raw:
                    return std_name

            logger.warning(f"[SUPERVISOR] LLM returned unrecognized agent: '{raw}' → using default route")
            return _default_route(state)

        except Exception as e:
            logger.warning(
                f"[SUPERVISOR] LLM routing attempt {attempt + 1}/3 failed: {e}"
            )
            if attempt == 2:  # 最后一次
                logger.error(f"[SUPERVISOR] All LLM routing attempts failed → default route")
                return _default_route(state)

    return _default_route(state)


# =============================================================================
# 第四部分：硬编码优先级检查 + 辅助函数
# =============================================================================


def _first_priority_check(state: dict) -> Optional[str]:
    """
    第一优先级硬编码检查 —— 不依赖任何路由函数，不会被异常跳过。

    这些规则是"铁律"，在任何情况下都必须执行，确保不会出现无限循环。
    """
    resume_data = state.get("resume_data")
    messages = state.get("messages", [])

    # 铁律 0：Agent 正在等待用户输入 → 停止调度
    if state.get("waiting_for_user"):
        return "FINISH"

    # 铁律 1：当 resume 不存在 + 已问过简历 + 用户没给文件路径时，
    # 优先检查是否是个人查询/通用问答/岗位匹配，否则 FINISH
    if not resume_data or not resume_data.get("name"):
        if state.get("_asked_for_resume"):
            last_user_msg = _get_last_user_message(messages)

            # 用户提供了文件路径 → 放行到 resume_match 解析
            if last_user_msg and _has_file_path(last_user_msg):
                return None  # 不拦截，让 _rule_based_route 处理

            # 岗位匹配请求但没有简历 → 提示先上传
            if last_user_msg and _has_match_job_intent(last_user_msg):
                logger.info(
                    f"[SUPERVISOR] First-priority: match_job without resume → FINISH"
                )
                return "FINISH"

            # 可能是个人信息查询或技术问题 → 交给 direct_reply
            if last_user_msg and (_has_personal_info_intent(last_user_msg) or
                                   _has_general_qa_intent(last_user_msg)):
                dr_count = state.get("_direct_reply_count", 0)
                if dr_count < 5:
                    logger.info(
                        f"[SUPERVISOR] First-priority: _asked_for_resume=True "
                        f"but user asked Q&A → direct_reply"
                    )
                    return "direct_reply"

            # 不是文件路径，不是 Q&A，也不是岗位匹配 → 放弃
            logger.info(
                f"[SUPERVISOR] First-priority: _asked_for_resume=True, "
                f"user didn't provide resume file → FINISH"
            )
            return "FINISH"

    return None


def _make_return(
    next_agent: str,
    total_steps: int,
    agent_history: list,
    error_count: dict,
    last_error: str = "",
) -> dict:
    """
    构建 Supervisor 返回值（统一入口）。

    保证每次返回都清空 _agent_signal，防止残留信号污染下一轮。
    """
    return {
        "next_agent": next_agent,
        "_total_steps": total_steps,
        "_agent_history": agent_history,
        "_error_count": error_count,
        "_last_error": last_error,
        "_agent_signal": "",  # 每次返回都清空，防止残留
        "interview_stage": _derive_stage(next_agent),
    }


# =============================================================================
# 第五部分：Supervisor 节点（主入口）
# =============================================================================

@with_permissions("supervisor")
def supervisor_node(state: dict) -> dict:
    """
    Supervisor 主节点：决定下一步路由目标。

    执行流程：
    -1. 重置残留的 next_agent（防止检查点中的旧值污染本次路由）
    0. Agent 主动信号（最优先，任何其他逻辑之前）
    1. 更新步数计数器
    2. 检查硬性终止条件（总步数 > 50）
    3. 规则路由（优先，0 token）
    4. LLM 路由（规则不确定时）
    5. 异常兜底 → 默认路由
    6. 记录 Agent 历史用于循环检测
    """
    # ---- Step -1：重置残留的 next_agent（防止检查点旧值跳过路由） ----
    stale_next = state.get("next_agent")
    stale_signal = state.get("_agent_signal", "")
    if stale_next or stale_signal:
        logger.info(
            f"[SUPERVISOR] Resetting stale state: next_agent='{stale_next}', "
            f"_agent_signal='{stale_signal}'"
        )

    print(f"[DEBUG] supervisor_node 被调用")
    print(f"[DEBUG]   next_agent: '{stale_next}', _agent_signal: '{stale_signal}'")
    print(f"[DEBUG]   resume_data: {bool(state.get('resume_data'))}, "
          f"job_matches: {len(state.get('job_matches', []))}, "
          f"questions: {len(state.get('interview_questions', []))}")
    last_usr = _get_last_user_message(state.get("messages", []))
    print(f"[DEBUG]   last_user_msg: '{last_usr[:80] if last_usr else 'N/A'}'")
    if last_usr:
        print(f"[DEBUG]   personal_info_intent: {_has_personal_info_intent(last_usr)}, "
              f"general_qa_intent: {_has_general_qa_intent(last_usr)}, "
              f"resume_intent: {_has_resume_intent(last_usr)}")

    # ---- Step 0：Agent 主动 FINISH 信号（仅在同一次调用内生效） ----
    # 关键守护条件：只有当最后一条消息不是用户消息时，才认为 _agent_signal
    # 是当前调用内产生的。如果最后一条是用户消息，说明已经进入新调用，
    # _agent_signal 是上一次调用残留的，必须忽略并走路由逻辑。
    messages_list = list(state.get("messages", []))
    last_msg = messages_list[-1] if messages_list else None
    last_msg_is_human = (
        last_msg
        and hasattr(last_msg, 'type')
        and last_msg.type == 'human'
    )

    if stale_signal == "FINISH" and not last_msg_is_human:
        logger.info("[SUPERVISOR] Agent signal FINISH (intra-invocation) → END")
        return {"next_agent": "FINISH", "_agent_signal": "", "interview_stage": _derive_stage("FINISH")}

    # ---- Step 1：更新步数 ----
    total_steps = state.get("_total_steps", 0) + 1
    agent_history = list(state.get("_agent_history", []))
    error_count = state.get("_error_count", {}).copy()

    # ---- Step 2：硬性终止条件（总步数 > 50，Supervisor 兜底）----
    if total_steps > 50:
        logger.error(f"[SUPERVISOR] CRITICAL: total_steps={total_steps} > 50 → FORCE END")
        return _make_return("FINISH", total_steps, agent_history, error_count, "Force finish: >50 steps")

    # ---- Step 3：第一优先级硬编码检查（不依赖任何路由函数）----
    # 这些规则绕过 _rule_based_route，确保不会被异常或 LLM 路由覆盖。
    first = _first_priority_check(state)
    if first is not None:
        agent_history.append(first)
        logger.info(
            f"[SUPERVISOR] First-priority check → '{first}' "
            f"(step {total_steps}, _asked_for_resume={state.get('_asked_for_resume')})"
        )
        return _make_return(first, total_steps, agent_history, error_count)

    # ---- Step 4：优先处理 Agent 显式信号 ----
    agent_signal = state.get("_agent_signal", "")
    if agent_signal:
        if agent_signal in ("resume_match", "interview", "evaluate", "report", "FINISH"):
            agent_history.append(agent_signal)
            logger.info(
                f"[SUPERVISOR] Agent signal='{agent_signal}' → respecting it "
                f"(step {total_steps})"
            )
            return _make_return(agent_signal, total_steps, agent_history, error_count)
        else:
            logger.warning(f"[SUPERVISOR] Invalid agent signal: '{agent_signal}' → ignoring")

    # ---- Step 5：规则路由 ----
    try:
        next_agent = _rule_based_route(state)
    except Exception as e:
        logger.error(f"[SUPERVISOR] Rule-based routing crashed: {e}")
        next_agent = None

    # ---- Step 6：LLM 路由（规则无法确定时）----
    if next_agent is None:
        try:
            next_agent = _llm_based_route(state)
        except Exception as e:
            logger.error(f"[SUPERVISOR] LLM routing crashed: {e}")
            next_agent = _default_route(state)

    # ---- Step 7：最终兜底 ----
    if next_agent is None or next_agent not in ["resume_match", "interview", "evaluate", "report", "direct_reply", "FINISH"]:
        logger.warning(f"[SUPERVISOR] Invalid agent '{next_agent}' → default route")
        next_agent = _default_route(state)

    # ---- Step 8：记录历史 ----
    agent_history.append(next_agent)

    # ---- Step 9：日志（调试用）----
    logger.info(
        f"[SUPERVISOR] Step {total_steps}: → '{next_agent}' "
        f"(stage={state.get('interview_stage', '?')}, "
        f"history={agent_history[-5:]})"
    )

    return _make_return(next_agent, total_steps, agent_history, error_count,
                        state.get("_last_error", ""))


# =============================================================================
# 第六部分：条件边 — 路由 + 循环检测
# =============================================================================

def route_after_supervisor(state: dict) -> str:
    """
    条件边函数：将 Supervisor 的决策映射到实际节点，同时执行循环检测。

    这是 LangGraph 的 conditional_edges 回调，必须返回节点名或 END。
    """
    next_agent = state.get("next_agent", "FINISH")

    # ---- 循环检测（传入 state + next_agent 做预判）----
    force_end = check_and_enforce_loop_limits(state, next_agent)
    if force_end is not None:
        return force_end

    # ---- 路由映射 ----
    routing_map = {
        "resume_match": "resume_match",
        "interview": "interview",
        "evaluate": "evaluate",
        "report": "report",
        "direct_reply": "direct_reply",
        "FINISH": END,
    }

    target = routing_map.get(next_agent)
    if target is None:
        logger.warning(f"[SUPERVISOR] Unknown agent '{next_agent}' → END")
        return END

    return target


# =============================================================================
# 第六部分：检查点工厂 — Redis → SQLite → Memory
# =============================================================================

def create_checkpointer(prefer: str = "auto") -> Any:
    """
    创建检查点存储器（自动降级）。

    策略：
        "auto"   → 依次尝试 Redis → SQLite → Memory
        "redis"  → 仅使用 Redis（失败抛异常）
        "sqlite" → 仅使用 SQLite
        "memory" → 仅使用 Memory

    返回：
        BaseCheckpointSaver 实例
    """
    # --- 尝试 Redis ---
    if prefer in ("auto", "redis"):
        try:
            from langgraph.checkpoint.redis import RedisSaver

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            saver = RedisSaver(redis_url=redis_url)
            saver.setup()  # 关键：创建 RediSearch 索引（FT.CREATE checkpoint / checkpoint_write）
            logger.info(f"[CHECKPOINT] Using RedisSaver ({redis_url})")
            return saver
        except ImportError:
            logger.debug("redis or langgraph-checkpoint-redis not installed")
        except Exception as e:
            logger.warning(f"[CHECKPOINT] Redis unavailable ({e}), trying SQLite...")

    # --- 降级 SQLite ---
    if prefer in ("auto", "sqlite"):
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_dir = os.getenv("CHECKPOINT_DIR", "./data")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "checkpoints.db")

            conn = sqlite3.connect(db_path, check_same_thread=False)
            logger.info(f"[CHECKPOINT] Using SqliteSaver at {db_path}")
            return SqliteSaver(conn)
        except Exception as e:
            logger.warning(f"[CHECKPOINT] SQLite unavailable ({e}), falling back to Memory...")

    # --- 降级 Memory ---
    logger.info("[CHECKPOINT] Using MemorySaver (in-memory, not persistent)")
    return MemorySaver()


# =============================================================================
# 第七部分：图构建器（占位 — 后续 Phase 补充 Agent 节点）
# =============================================================================

def build_multi_agent_graph(checkpointer=None):
    """
    构建多 Agent 图（完整版）。

    使用真实的 Agent 节点（非占位）。
    """
    if checkpointer is None:
        checkpointer = create_checkpointer()

    # 延迟导入，避免循环依赖
    from agents.resume_match_agent import resume_match_node
    from agents.interview_agent import interview_node
    from agents.evaluator_agent import evaluator_node
    from agents.report_agent import report_node
    from agents.direct_reply_agent import direct_reply_node

    graph = StateGraph(MultiAgentState)

    # ---- Supervisor 节点 ----
    graph.add_node("supervisor", supervisor_node)

    # ---- 真实 Agent 节点 ----
    graph.add_node("resume_match", resume_match_node)
    graph.add_node("interview", interview_node)
    graph.add_node("evaluate", evaluator_node)
    graph.add_node("report", report_node)
    graph.add_node("direct_reply", direct_reply_node)

    # ---- 边 ----
    graph.add_edge(START, "supervisor")

    # Supervisor → 条件路由到各 Agent
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "resume_match": "resume_match",
            "interview": "interview",
            "evaluate": "evaluate",
            "report": "report",
            "direct_reply": "direct_reply",
            END: END,
        },
    )

    # 各 Agent 完成后 → 回到 Supervisor
    graph.add_edge("resume_match", "supervisor")
    graph.add_edge("interview", "supervisor")
    graph.add_edge("evaluate", "supervisor")
    graph.add_edge("report", "supervisor")
    graph.add_edge("direct_reply", "supervisor")

    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# 第八部分：辅助函数
# =============================================================================

def _get_last_user_message(messages: list) -> Optional[str]:
    """获取最近一条用户消息文本"""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, list):
                return content[0].get('text', '') if content else ''
            return str(content)
    return None


def _get_last_ai_message(messages: list) -> Optional[str]:
    """获取最近一条 AI 消息文本"""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            if hasattr(msg, 'content') and msg.content:
                return str(msg.content)
    return None


def _has_resume_intent(text: str) -> bool:
    """
    检测用户消息是否包含【简历上传/解析】意图。

    只匹配明确的上传意图短语和文件扩展名，不匹配句子中泛泛出现的"简历"一词。
    （岗位匹配用 _has_match_job_intent 单独检测）
    """
    if not text:
        return False
    text_lower = text.lower()

    # 文件扩展名（高置信度：用户提供了文件路径）
    extensions = ['.pdf', '.docx', '.doc', '.txt']
    if any(ext in text_lower for ext in extensions):
        return True

    # 明确的简历上传/解析短语（多词组合，避免单字误判）
    upload_phrases = [
        '上传简历', '提供简历', '发简历', '提交简历',
        '这是我的简历', '我的简历路径', '简历文件',
        '解析简历', '简历解析', '简历路径',
        'upload resume', 'parse resume', 'resume file',
        'resume path', 'my resume',
    ]
    return any(phrase in text_lower for phrase in upload_phrases)


def _has_file_path(text: str) -> bool:
    """
    严格检测用户消息是否包含文件路径（仅用于 _first_priority_check 放行判断）。

    与 _has_resume_intent 的区别：此函数只检查文件扩展名，
    不匹配"简历""上传"等关键词，防止"帮我匹配岗位"被误认为提供了简历。
    """
    if not text:
        return False
    import re
    patterns = [
        r'\.pdf\b', r'\.docx?\b', r'\.txt\b',
        r'[A-Za-z]:[\\/]',  # Windows absolute path
        r'/[^\s]+\.(?:pdf|docx?|txt)\b',  # Unix absolute path
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _has_match_job_intent(text: str) -> bool:
    """
    检测用户消息是否包含【岗位匹配】意图。

    这些关键词意味着用户想根据简历匹配岗位（需要已有简历数据）。
    """
    if not text:
        return False
    keywords = [
        '匹配岗位', '帮我匹配', '推荐岗位', '岗位匹配',
        '匹配职位', '推荐职位', '有什么岗位', '有哪些岗位',
        '找岗位', '找工作', '职位推荐', '岗位推荐',
        '帮我推荐', '看看岗位', '查看岗位', '显示岗位',
    ]
    return any(kw in text for kw in keywords)


def _has_job_selection_intent(text: str) -> bool:
    """检测用户消息是否包含岗位选择意图（注意：排除'匹配岗位'这种非选择意图）"""
    # 排除"匹配"意图 — 那是要重新匹配，不是选择
    if '匹配' in text:
        return False
    keywords = ['选', '第', '这个', '那个', '选择']
    text_lower = text.lower()
    # 纯数字也视为选择意图（如 "1", "2" 等）
    if text.strip().isdigit():
        return True
    return any(kw.lower() in text_lower for kw in keywords)


def _has_general_qa_intent(text: str) -> bool:
    """
    检测用户消息是否属于「通用问答」场景（不需要简历就能回答）。

    判断策略（从严格到宽松，反匹配优先）：
    1. 反匹配：包含简历/文件上传关键词 → 直接返回 False
    2. 技术问题英文句式
    3. 技术问题中文句式
    4. 面试咨询关键词
    5. 问候/闲聊（短消息）
    6. 包含技术术语（最宽松）
    """
    if not text:
        return False

    text_lower = text.lower().strip()

    # ---- 反匹配：简历上传意图 → 不是通用问答 ----
    if _has_resume_intent(text):
        return False

    # ---- Category A: 技术问题英文句式 ----
    tech_starters_en = [
        "what is", "how to", "how do", "explain", "define",
        "tell me about", "difference between", "when to", "why does",
        "what are", "what does", "can you explain", "meaning of",
        "compare", "describe", "what's the", "how does",
    ]
    for kw in tech_starters_en:
        if text_lower.startswith(kw) or f" {kw} " in f" {text_lower} ":
            return True

    # ---- Category B: 技术问题中文句式 ----
    tech_starters_cn = [
        "什么是", "如何", "怎么", "解释", "介绍一下",
        "什么区别", "为什么", "什么意思", "怎么理解",
        "说明一下", "讲讲", "能说说", "请教",
        "有啥区别", "怎样", "能否解释",
    ]
    for kw in tech_starters_cn:
        if kw in text_lower:
            return True

    # ---- Category C: 面试咨询 ----
    interview_advice_kw = [
        "如何准备面试", "面试技巧", "面试经验", "面试注意",
        "how to prepare", "interview tips", "面试题", "常见面试",
        "技术面试", "behavioral interview", "coding interview",
        "prepare for interview", "面试准备",
    ]
    for kw in interview_advice_kw:
        if kw.lower() in text_lower:
            return True

    # ---- Category D: 问候/闲聊 ----
    greetings = {
        "hi", "hello", "hey", "你好", "您好", "早上好", "晚上好",
        "thanks", "thank you", "谢谢", "感谢", "how are you",
        "最近怎么样", "what can you do", "你能做什么", "你是谁",
        "who are you", "who r u", "good morning", "good afternoon",
        "good evening", "哈喽", "嗨", "在吗", "在不在",
    }
    # 精确匹配或短消息包含问候词
    if text_lower in greetings:
        return True
    if len(text_lower) < 40 and any(g in text_lower for g in greetings):
        return True

    # ---- Category E: 包含技术术语（兜底检测） ----
    tech_terms = [
        "python", "java", "javascript", "typescript", "golang", "go语言",
        "rust", "c++", "c#", "php", "ruby", "swift", "kotlin",
        "fastapi", "django", "flask", "spring", "react", "vue", "angular",
        "docker", "kubernetes", "k8s", "aws", "cloud", "devops",
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "git", "linux", "api", "rest", "graphql", "microservice", "微服务",
        "algorithm", "算法", "数据结构", "data structure",
        "design pattern", "设计模式",
        "concurrency", "并发", "parallel", "并行", "async", "异步",
        "machine learning", "机器学习", "deep learning", "深度学习",
        "ai", "agent", "llm", "大模型", "大语言模型", "langchain",
        "garbage collection", "gil", "memory", "内存", "线程", "进程",
        "thread", "process", "lambda", "closure", "闭包",
        "decorator", "装饰器", "oop", "面向对象", "functional", "函数式",
        "编程", "coding", "前端", "后端", "全栈", "full stack",
        "frontend", "backend", "nginx", "ci/cd", "jenkins", "github",
        "面试官", "面试", "面试系统", "怎么用",
    ]
    if any(term in text_lower for term in tech_terms):
        return True

    return False


def _has_personal_info_intent(text: str) -> bool:
    """
    检测用户消息是否在查询个人信息（需要从 Redis/state/RAG 获取）。

    判断策略：
    1. 反匹配：纯技术问题不含第一人称 → 不是个人查询
    2. 第一人称 + 个人信息关键词
    3. 问系统知道什么/记得什么
    4. 个人身份确认

    """
    if not text:
        return False

    # ---- 反匹配：简历上传意图 → 不是个人查询（交给 resume_match） ----
    if _has_resume_intent(text):
        return False

    # ---- Category A：第一/第三人称 + 个人信息关键词（高置信度） ----
    first_person_markers = ["我", "我的", "我叫", "我是"]
    personal_info_fields = [
        "名字", "姓名", "叫什么", "是谁",
        "生日", "出生", "几月几号", "出生日期",
        "电话", "手机", "联系方式", "号码",
        "邮箱", "email", "邮件",
        "学校", "大学", "学院", "学历", "专业",
    ]

    has_first_person = any(m in text for m in first_person_markers)
    has_personal_field = any(f in text for f in personal_info_fields)

    if has_first_person and has_personal_field:
        return True

    # 第三人称所有格：xxx的生日、xxx的名字（如 "曹星桥的生日是多久"）
    possessive_personal_patterns = [
        "的名字", "的姓名", "的生日", "的出生日期", "的电话",
        "的手机", "的邮箱", "的学校", "的学历", "的专业",
    ]
    if any(p in text for p in possessive_personal_patterns):
        return True

    # ---- Category B：问系统知道什么/记得什么 ----
    system_knowledge = [
        "你知道我", "你记得我", "你认识我",
        "我的简历", "我的信息", "我的资料",
        "知道我", "记得我", "认识我",
    ]
    if any(kw in text for kw in system_knowledge):
        return True

    # ---- Category C：个人身份确认 ----
    identity_keywords = ["我是谁", "你认识我吗", "还记得我吗"]
    if any(kw in text for kw in identity_keywords):
        return True

    return False


def _has_start_interview_intent(text: str) -> bool:
    """
    检测用户消息是否包含开始面试意图。

    这些关键词意味着用户想启动模拟面试流程（前提是已选择岗位）。
    """
    keywords = [
        '开始面试', '开始模拟面试', '启动面试', '进入面试',
        '开始吧', '可以面试了', '准备好了', '面试开始',
        '开始问答', '开始提问', '出题吧',
        'start interview', 'begin interview', 'let\'s start',
    ]
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _is_explicit_end_interview(text: str) -> bool:
    """
    严格检测用户是否【明确】要求结束面试。

    与 _has_end_interview_intent 的区别：
    - 此函数只匹配短消息（< 15 字）且仅包含结束关键词
    - 不会误判"这个项目结束了"这类正常回答
    - 用于 Priority 0 面试进行中的例外处理
    """
    if not text:
        return False
    text_stripped = text.strip()
    # 消息必须很短（真正的结束命令不会长篇大论）
    if len(text_stripped) > 15:
        return False
    keywords = [
        '结束面试', '终止面试', '停止面试', '退出面试',
        '结束', '终止', '停止', '退出', '不继续了',
        'finish', 'end', 'stop', 'quit', 'exit', 'cancel',
    ]
    text_lower = text_stripped.lower()
    return any(kw.lower() == text_lower or kw.lower() in text_lower for kw in keywords)


def _has_end_interview_intent(text: str) -> bool:
    """
    检测用户消息是否包含结束面试意图。

    这些关键词意味着用户想提前终止面试，直接生成报告。
    """
    keywords = ['结束面试', '结束', '退出', '终止', '不继续了', '到此为止', '停止面试',
                '不想继续', 'finish', 'end', 'stop', 'quit', 'exit', 'cancel']
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _contains_evaluation(text: str) -> bool:
    """检测文本是否包含评分/评价内容"""
    keywords = ['评分', '评价', '得分', 'score', 'evaluat', '表现']
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _build_routing_context(state: dict) -> str:
    """构建路由上下文（给 LLM 看的精简版状态摘要）"""
    parts = [
        "## 当前状态",
        f"- 简历已解析: {'是' if state.get('resume_data') else '否'}",
        f"- 岗位匹配数: {len(state.get('job_matches', []))}",
        f"- 已选岗位: {'是' if state.get('selected_job') else '否'}",
        f"- 面试题数: {len(state.get('interview_questions', []))}",
        f"- 已回答数: {len(state.get('answers', []))}",
        f"- 报告已生成: {'是' if state.get('final_report') else '否'}",
        f"- 总步数: {state.get('_total_steps', 0)}",
        f"- 错误: {state.get('_last_error', '无')}",
        "",
        "## 最近消息",
    ]

    messages = state.get("messages", [])
    for msg in messages[-3:]:  # 只看最近 3 条
        role = getattr(msg, 'type', 'unknown')
        content = getattr(msg, 'content', '')
        if isinstance(content, str) and len(content) > 150:
            content = content[:150] + "..."
        parts.append(f"[{role}] {content}")

    parts.append("\n请决定下一步交给哪个 Agent。只回复 agent 名称。")
    return "\n".join(parts)


def _derive_stage(next_agent: str) -> str:
    """从路由目标推导面试阶段"""
    mapping = {
        "resume_match": "parsing",
        "interview": "interviewing",
        "evaluate": "evaluating",
        "report": "reporting",
        "direct_reply": "chatting",
        "FINISH": "done",
    }
    return mapping.get(next_agent, "idle")


def _placeholder_node(agent_name: str):
    """
    占位 Agent 节点（后续替换为真实 Agent 实现）。

    当前行为：打印日志 + 返回空状态（让 Supervisor 继续路由）。
    """

    @with_permissions(agent_name)
    def node(state: dict) -> dict:
        logger.info(f"[PLACEHOLDER] Agent '{agent_name}' called (not yet implemented)")
        # 模拟：根据 Agent 类型返回合理的占位数据
        if agent_name == "resume_match" and not state.get("resume_data"):
            return {
                "interview_stage": "parsing",
                "resume_data": {"_placeholder": True, "name": "placeholder"},
            }
        if agent_name == "resume_match" and state.get("resume_data") and not state.get("job_matches"):
            return {
                "interview_stage": "matching",
                "job_matches": [{"_placeholder": True}],
            }
        if agent_name == "interview" and not state.get("interview_questions"):
            return {
                "interview_stage": "interviewing",
                "interview_questions": ["(placeholder question 1)", "(placeholder question 2)"],
                "current_question_idx": 0,
            }
        if agent_name == "evaluate":
            answers = state.get("answers", [])
            return {
                "interview_stage": "evaluating",
                "answers": answers + [{"_placeholder": True, "score": 7}],
            }
        if agent_name == "report":
            return {
                "interview_stage": "reporting",
                "final_report": "(placeholder report)",
                "overall_score": 7.0,
            }
        return {}

    return node


# =============================================================================
# 第九部分：自检（直接运行此文件）
# =============================================================================

if __name__ == "__main__":
    print("=== Supervisor Self-Check ===\n")

    # Test 1: Rule-based routing
    print("[1] Testing rule-based routing...")

    # Empty state → resume_match
    result = _rule_based_route({})
    assert result == "resume_match", f"Empty state should route to resume_match, got {result}"
    print("    [OK] Empty state → resume_match")

    # Resume parsed, no jobs → resume_match
    result = _rule_based_route({"resume_data": {"name": "Test"}})
    assert result == "resume_match", f"Resume but no jobs should route to resume_match, got {result}"
    print("    [OK] Resume parsed, no jobs → resume_match")

    # Jobs matched, no selection → resume_match
    result = _rule_based_route({
        "resume_data": {"name": "Test"},
        "job_matches": [{"id": "1"}],
    })
    assert result == "resume_match", f"Jobs but no selection → resume_match, got {result}"
    print("    [OK] Jobs but no selection → resume_match")

    # Job selected, no questions → interview
    result = _rule_based_route({
        "resume_data": {"name": "Test"},
        "job_matches": [{"id": "1"}],
        "selected_job": {"id": "1"},
    })
    assert result == "interview", f"Job selected, no questions → interview, got {result}"
    print("    [OK] Job selected, no questions → interview")

    # All questions answered → report
    result = _rule_based_route({
        "resume_data": {"name": "Test"},
        "job_matches": [{"id": "1"}],
        "selected_job": {"id": "1"},
        "interview_questions": ["Q1", "Q2"],
        "answers": [{"q": "Q1"}, {"q": "Q2"}],
    })
    assert result == "report", f"All answered → report, got {result}"
    print("    [OK] All questions answered → report")

    # Report done → FINISH
    result = _rule_based_route({
        "resume_data": {"name": "Test"},
        "job_matches": [{"id": "1"}],
        "selected_job": {"id": "1"},
        "interview_questions": ["Q1", "Q2"],
        "answers": [{"q": "Q1"}, {"q": "Q2"}],
        "final_report": "Great!",
    })
    assert result == "FINISH", f"Report done → FINISH, got {result}"
    print("    [OK] Report done → FINISH")

    # Agent failed 3 times → fallback
    result = _rule_based_route({
        "_error_count": {"resume_match": 3},
        "resume_data": {"name": "Test"},
        "job_matches": [{"id": "1"}],
        "selected_job": {"id": "1"},
    })
    assert result == "interview", f"Agent failed 3x → fallback to interview, got {result}"
    print("    [OK] Agent failed 3x → fallback to interview")

    print()

    # Test 2: Default route (crash fallback)
    print("[2] Testing default route...")
    dr = _default_route({})
    assert dr == "resume_match", f"Default empty → resume_match, got {dr}"
    print("    [OK] Default route: empty → resume_match")

    dr = _default_route({"resume_data": {"name": "T"}, "job_matches": [{"id": "1"}], "selected_job": {"id": "1"}, "interview_questions": ["Q"], "answers": [{"q":"Q","a":"A"}]})
    assert dr == "report", f"Default all done → report, got {dr}"
    print("    [OK] Default route: all done → report")
    print()

    # Test 3: Supervisor node
    print("[3] Testing supervisor_node...")
    result = supervisor_node({})
    assert "next_agent" in result, f"Should have next_agent, got {result}"
    assert result["_total_steps"] == 1, f"First step should be 1, got {result['_total_steps']}"
    assert len(result["_agent_history"]) == 1, f"History should have 1 entry"
    print(f"    [OK] supervisor_node → {result['next_agent']} (step {result['_total_steps']})")
    print()

    # Test 4: Checkpointer
    print("[4] Testing checkpointer creation...")
    cp = create_checkpointer("memory")
    assert cp is not None, "Checkpointer should not be None"
    print(f"    [OK] Checkpointer created: {type(cp).__name__}")

    try:
        cp = create_checkpointer("sqlite")
        print(f"    [OK] SQLite checkpointer: {type(cp).__name__}")
    except Exception as e:
        print(f"    [WARN] SQLite checkpointer failed: {e}")

    # Clean up test DB
    import os as _os, shutil as _sh
    for p in ["data/checkpoints.db", "data/checkpoints.db-wal", "data/checkpoints.db-shm", "data/", "checkpoints.db"]:
        try:
            if _os.path.isfile(p):
                _os.remove(p)
            elif _os.path.isdir(p) and p == "data/" and not _os.listdir(p):
                _sh.rmtree(p, ignore_errors=True)
        except Exception:
            pass
    print()

    # Test 5: Build graph
    print("[5] Testing graph build...")
    graph = build_multi_agent_graph()
    assert graph is not None, "Graph should not be None"
    print(f"    [OK] Graph built successfully")
    print()

    print("=== ALL SUPERVISOR TESTS PASSED ===")
