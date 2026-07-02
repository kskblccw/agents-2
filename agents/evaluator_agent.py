#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator Agent — 回答评分 (evaluator_agent.py)
================================================

职责：
1. 对候选人的回答进行 1-10 分评分
2. 给出具体评价和改进建议
3. 决定是否需要追问

工具：
- RAG 知识库检索（获取评分标准和参考答案）
- LLM 评分 + 生成评语

权限：可写 evaluations, answers, last_answer_score, current_question_idx, interview_stage
"""

import json
import re
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage

from agents.state import with_permissions
from agents.utils import safe_llm_call, build_cache_key

# search_knowledge_base 定义在 agent_core.py，避免从 agents 包循环导入
try:
    from agent_core import search_knowledge_base
except ImportError:
    search_knowledge_base = None
logger = logging.getLogger("multi-agent.evaluator")

# =============================================================================
# System Prompts
# =============================================================================

EVALUATION_PROMPT = """你是严格的技术面试评分员。请评价候选人的回答。

{rag_context}
【面试问题】
{question}

【候选人回答】
{answer}

【岗位背景】
{job_context}

请从以下角度评价：
1. 回答的相关性和完整性（是否覆盖问题要点）
2. 技术深度和准确性（是否有细节、案例、数据）
3. 表达清晰度（逻辑是否通顺）

请以 JSON 格式输出：
{{
    "score": <整数 1-10>,
    "comment": "<2-3 句话简短评价>",
    "strength": "<回答的优点>",
    "improvement": "<需要改进的地方>",
    "need_follow_up": <true 或 false>
}}

评分标准：
- 9-10：回答完整，有具体案例和量化数据，技术深度好
- 7-8：核心正确，要点覆盖，但部分细节不足
- 5-6：方向对但浮于表面，缺乏具体性
- 3-4：回答不完整，重要要点遗漏
- 1-2：严重错误或完全偏离问题

注意：只输出 JSON，不要有其他文字。"""


# =============================================================================
# 工具函数
# =============================================================================

def _get_rag_context_for_evaluation(question: str, answer: str, job_title: str = "") -> str:
    """从 RAG 知识库检索评分参考"""
    try:
        from knowledge_base import get_rag_chain
        rag = get_rag_chain()
        if rag:
            context = rag.retrieve_for_answer_evaluation(
                question=question,
                answer=answer,
                job_title=job_title,
                top_k=3,
            )
            return context if context else ""
    except Exception as e:
        logger.warning(f"[evaluator] RAG retrieval failed: {e}")
    return ""


def _parse_evaluation_response(response: str) -> dict:
    """解析 LLM 评分响应为结构化 dict"""
    default = {
        "score": 5,
        "comment": "评分解析失败，按默认 5 分处理。",
        "strength": "（未识别）",
        "improvement": "（未识别）",
        "need_follow_up": False,
    }

    logger.info(f"[evaluator] Raw LLM response ({len(response)} chars):\n{response[:500]}")

    try:
        json_str = response.strip()
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()

        logger.info(f"[evaluator] JSON to parse ({len(json_str)} chars):\n{json_str[:300]}")

        data = json.loads(json_str)
        if isinstance(data, dict):
            score = int(data.get('score', 5))
            score = max(1, min(10, score))  # clamp 1-10
            logger.info(
                f"[evaluator] Parsed OK: score={score}, "
                f"need_follow_up={data.get('need_follow_up', False)}, "
                f"comment='{str(data.get('comment', ''))[:60]}'"
            )
            return {
                "score": score,
                "comment": str(data.get('comment', '')),
                "strength": str(data.get('strength', '')),
                "improvement": str(data.get('improvement', '')),
                "need_follow_up": bool(data.get('need_follow_up', False)),
            }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"[evaluator] JSON parse failed: {type(e).__name__}: {e}")

    # JSON 失败 → 正则提取评分数字
    score_match = re.search(r'"score"\s*:\s*(\d+)', response)
    comment_match = re.search(r'"comment"\s*:\s*"([^"]+)"', response)
    if score_match:
        default["score"] = max(1, min(10, int(score_match.group(1))))
        logger.info(f"[evaluator] Regex fallback: score={default['score']}")
    if comment_match:
        default["comment"] = comment_match.group(1)

    return default


# =============================================================================
# Agent 主节点
# =============================================================================

@with_permissions("evaluate")
def evaluator_node(state: dict) -> dict:
    """
    Evaluator Agent 主节点。

    流程：
    1. 获取当前问题和候选人刚才的回答
    2. 从 RAG 检索评分参考
    3. 调用 LLM 进行评分
    4. 更新 answers 列表和评分状态
    """
    import traceback

    # ---- 入口调试 ----
    print("[DEBUG] ========== evaluator_node 被调用 ==========")
    print(f"[DEBUG] current_question_idx: {state.get('current_question_idx')}")
    print(f"[DEBUG] questions count: {len(state.get('interview_questions', []))}")
    print(f"[DEBUG] answers count: {len(state.get('answers', []))}")
    print(f"[DEBUG] selected_job: {bool(state.get('selected_job'))}")
    print(f"[DEBUG] _agent_signal: {state.get('_agent_signal', 'N/A')}")

    result = {}

    messages = state.get("messages", [])
    questions = state.get("interview_questions", [])
    current_idx = state.get("current_question_idx", 0)
    selected_job = state.get("selected_job", {})
    answers = list(state.get("answers", []))
    evaluations = list(state.get("evaluations", []))
    thread_id = state.get("_agent_history", ["?"])[-1] if state.get("_agent_history") else "?"

    total_questions = len(questions) if questions else 0
    print(f"[DEBUG] total_questions={total_questions}, current_idx={current_idx}")

    # ---- 获取候选人回答 ----
    # 从最近的消息中提取用户回答
    user_answer = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, list):
                content = content[0].get('text', '') if content else ''
            user_answer = str(content)
            break

    print(f"[DEBUG] user_answer: '{user_answer[:100]}...' ({len(user_answer)} chars)")

    if not user_answer.strip():
        print("[DEBUG] ⚠️  user_answer is empty — returning early")
        result["messages"] = [AIMessage(content="我没有收到您的回答，请再说一遍？")]
        return result

    # ---- 检测结束面试意图（防御性检查：正常应由 Supervisor 拦截）----
    answer_stripped = user_answer.strip()
    is_short = len(answer_stripped) < 15
    short_end_kw = ['结束', '退出', '终止', '不继续了', '到此为止', '停止', '不想继续',
                    'finish', 'end', 'stop', 'quit', 'exit', 'cancel']
    full_end_kw = ['结束面试', '终止面试', '退出面试']
    if (is_short and any(kw in answer_stripped.lower() for kw in short_end_kw)) or \
       any(kw in answer_stripped for kw in full_end_kw):
        print("[DEBUG] ⚠️  Detected end-interview intent in evaluator")
        logger.info(f"[evaluator] User requested early termination → routing to report")
        result["_agent_signal"] = "report"
        result["interview_stage"] = "evaluating"
        result["answers"] = answers
        result["messages"] = [AIMessage(
            content="好的，面试到此结束。正在为您生成面试报告，请稍候..."
        )]
        return result

    # ---- 获取当前问题 ----
    if questions is None or len(questions) == 0:
        print("[DEBUG] ⚠️  questions is empty — cannot evaluate")
        logger.error(
            f"[evaluator] questions is empty or None (current_idx={current_idx}). "
            f"State may have been lost. Falling back to default score."
        )
        result["messages"] = [AIMessage(content="面试题列表为空，无法评分。请联系管理员。")]
        return result

    if current_idx >= len(questions):
        print(f"[DEBUG] ⚠️  current_idx ({current_idx}) >= len(questions) ({len(questions)}) — no question to evaluate")
        logger.warning(
            f"[evaluator] current_idx={current_idx} >= len(questions)={len(questions)}. "
            f"All questions may have been answered."
        )
        result["messages"] = [AIMessage(content="当前没有待评分的问题。")]
        return result

    current_question = questions[current_idx]

    # ---- 执行评分 ----
    print("[DEBUG] 准备调用 evaluate_answer_tool...")
    try:
        from agent_core import evaluate_answer_tool

        # 调用工具评分
        raw_result = evaluate_answer_tool.invoke({
            "question": current_question,
            "answer": user_answer,
        })
        print(f"[DEBUG] evaluate_answer_tool 返回原始结果 ({type(raw_result).__name__}, {len(str(raw_result))} chars):")
        print(f"[DEBUG]   {str(raw_result)[:500]}")

        # 解析工具返回的 JSON 字符串
        if isinstance(raw_result, str):
            score_data = json.loads(raw_result)
        elif isinstance(raw_result, dict):
            score_data = raw_result
        else:
            score_data = json.loads(str(raw_result))

        score = int(score_data.get("score", 5))
        score = max(1, min(10, score))
        comment = str(score_data.get("comment", ""))

        print(f"[DEBUG] 解析后: score={score}, comment='{comment[:80]}'")
        print(f"[DEBUG] 准备更新 state: answers {len(answers)} -> {len(answers) + 1}, next_idx={current_idx + 1}")

        # 更新状态
        answer_record = {
            "question_idx": current_idx,
            "question": current_question,
            "answer": user_answer,
            "evaluation": comment,
            "score": score,
        }
        answers.append(answer_record)

        result["answers"] = answers
        result["evaluations"] = evaluations
        result["last_answer_score"] = float(score)
        result["current_question_idx"] = current_idx + 1
        result["interview_stage"] = "evaluating"

        # 构建评语
        score_emoji = "🌟" if score >= 8 else "👍" if score >= 6 else "💡" if score >= 4 else "⚠️"
        eval_msg = f"{score_emoji} 评分：{score}/10\n\n📝 评语：{comment}"

        # 路由决策
        next_idx = current_idx + 1
        has_more = next_idx < total_questions

        if has_more:
            result["_agent_signal"] = "interview"
            result["next_difficulty"] = "normal"
            eval_msg += f"\n\n准备进入第 {next_idx + 1}/{total_questions} 题..."
            print(f"[DEBUG] 还有更多问题 -> _agent_signal='interview'")
        else:
            result["_agent_signal"] = ""
            eval_msg += f"\n\n🎉 所有 {total_questions} 道题已答完，正在生成面试报告..."
            print(f"[DEBUG] 最后一题 -> _agent_signal=''")

        result["messages"] = [AIMessage(content=eval_msg)]
        print(f"[DEBUG] evaluator 完成: answers={len(answers)}, score={score}")

    except Exception as e:
        import traceback
        print(f"[DEBUG] ❌ 评分异常: {type(e).__name__}: {e}")
        traceback.print_exc()

        # 降级：默认 5 分，继续面试
        answers.append({
            "question_idx": current_idx,
            "question": current_question,
            "answer": user_answer,
            "evaluation": f"评分异常（{str(e)[:80]}），按默认 5 分。",
            "score": 5,
        })
        result["answers"] = answers
        result["current_question_idx"] = current_idx + 1
        result["last_answer_score"] = 5.0
        result["interview_stage"] = "evaluating"

        next_idx = current_idx + 1
        has_more = next_idx < total_questions
        if has_more:
            result["_agent_signal"] = "interview"
            result["next_difficulty"] = "normal"
            fallback_msg = f"收到您的回答（默认评分 5/10）。\n\n准备进入第 {next_idx + 1}/{total_questions} 题..."
        else:
            result["_agent_signal"] = ""
            fallback_msg = f"收到您的回答（默认评分 5/10）。\n\n🎉 所有 {total_questions} 道题已答完，正在生成面试报告..."
        result["messages"] = [AIMessage(content=fallback_msg)]
        print(f"[DEBUG] 降级完成: answers={len(answers)}")

    return result
