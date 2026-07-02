#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Agent — 面试报告生成 (report_agent.py)
==============================================

职责：
1. 汇总所有问答记录和评分
2. 生成综合面试评估报告
3. 包含：综合评分、各题得分、优势/不足、录用建议

工具：
- LLM 生成报告文本

权限：可写 final_report, overall_score, interview_stage
"""

import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage

from agents.state import with_permissions
from agents.utils import safe_llm_call, build_cache_key

logger = logging.getLogger("multi-agent.report")

# =============================================================================
# System Prompt
# =============================================================================

REPORT_PROMPT = """请为这场面试生成一份详细的评分与反馈报告：

【候选人】{candidate_name}
【目标职位】{job_title} @ {company}
【完成情况】回答了 {answered}/{total} 个问题

【问答记录】
{qa_records}

请生成详细的面试报告，包括：

## 一、评分指标（请为每项打分 0-10，保留 1 位小数）
1. 技术能力评分：基于回答的技术深度和准确性
2. 表达沟通评分：基于回答的清晰度和逻辑性
3. 经验匹配评分：基于经历与职位的相关度
4. 综合素质评分：综合整体表现

## 二、各项评分的详细说明
请对上述每项评分给出 2-3 句话的具体说明

## 三、优势分析
列出候选人的主要优势（3-5 条）

## 四、需要改进的地方
列出需要改进的地方（3-5 条）

## 五、综合评价与建议
给出综合评价，是否推荐进入下一轮面试

## 六、录用建议
- 推荐度：强烈推荐 / 推荐 / 一般 / 不推荐
- 建议（如适用）

请用专业的语言回答，评分要客观公正。"""


# =============================================================================
# 工具函数
# =============================================================================

def _format_qa_records(answers: list) -> str:
    """格式化问答记录"""
    if not answers:
        return "（无问答记录）"

    parts = []
    for i, a in enumerate(answers, 1):
        q = a.get('question', '?')
        ans = a.get('answer', '?')
        score = a.get('score', '?')
        ev = a.get('evaluation', '')

        parts.append(f"### 问题 {i}")
        parts.append(f"Q: {q}")
        parts.append(f"A: {ans[:300]}")  # 截断过长回答
        parts.append(f"得分: {score}/10")
        if ev:
            parts.append(f"评价: {ev}")
        parts.append("")

    return '\n'.join(parts)


def _calculate_overall_score(answers: list) -> float:
    """计算综合评分"""
    if not answers:
        return 0.0

    scores = [a.get('score', 5) for a in answers]
    if not scores:
        return 0.0

    # 加权：前几题权重略高（考察核心能力）
    weights = [1.2, 1.1, 1.0, 0.9, 0.8][:len(scores)]
    total_weight = sum(weights)
    weighted_sum = sum(s * w for s, w in zip(scores, weights))

    return round(weighted_sum / total_weight, 1)


# =============================================================================
# Agent 主节点
# =============================================================================

@with_permissions("report")
def report_node(state: dict) -> dict:
    """
    Report Agent 主节点。

    流程：
    1. 汇总所有问答记录
    2. 计算综合评分
    3. 调用 LLM 生成结构化报告
    """
    resume_data = state.get("resume_data", {})
    selected_job = state.get("selected_job", {})
    answers = state.get("answers", [])
    questions = state.get("interview_questions", [])
    thread_id = state.get("_agent_history", ["?"])[-1] if state.get("_agent_history") else "?"

    # 清空面试状态的基础返回值（无论走哪个分支都要清空）
    result = {
        "_agent_signal": "FINISH",
        "interview_stage": "done",
        "interview_questions": [],
        "current_question_idx": 0,
        "answers": [],
        "last_answer_score": 0.0,
        "waiting_for_user": False,  # 防止残留值阻塞后续路由
    }

    # 如果已有报告，直接返回（不重复生成）
    if state.get("final_report"):
        result["final_report"] = state["final_report"]
        result["overall_score"] = state.get("overall_score", 0.0)
        result["messages"] = [AIMessage(content=state["final_report"])]
        return result

    # ---- 计算综合评分 ----
    overall_score = _calculate_overall_score(answers)

    try:
        # 准备数据
        candidate_name = resume_data.get('name', '未知候选人')
        job_title = selected_job.get('title', '未知职位')
        company = selected_job.get('company', '未知公司')
        qa_records = _format_qa_records(answers)

        # 构建 prompt
        prompt = REPORT_PROMPT.format(
            candidate_name=candidate_name,
            job_title=job_title,
            company=company,
            answered=len(answers),
            total=len(questions),
            qa_records=qa_records,
        )

        # 调用 LLM（报告不缓存，每次都重新生成）
        cache_key = build_cache_key(
            "report",
            candidate_name,
            job_title,
            str(len(answers)),
        )
        report_text = safe_llm_call(
            prompt=prompt,
            thread_id=thread_id,
            cache_key=cache_key,
            temperature=0.5,
        )

        # 组装最终报告
        header = (
            f"# 面试评估报告\n\n"
            f"**候选人**：{candidate_name}\n"
            f"**目标职位**：{job_title} @ {company}\n"
            f"**完成题目**：{len(answers)}/{len(questions)}\n"
            f"**综合评分**：{overall_score}/10\n\n"
            f"---\n\n"
        )
        final_report = header + report_text

        result["final_report"] = final_report
        result["overall_score"] = overall_score
        result["messages"] = [AIMessage(content=final_report)]

        logger.info(
            f"[report] Generated report for {candidate_name}: overall={overall_score}/10"
        )

    except Exception as e:
        logger.error(f"[report] Report generation failed: {e}")
        # 降级：生成简单文本报告
        fallback_report = (
            f"# 面试评估报告（基础版）\n\n"
            f"**候选人**：{resume_data.get('name', '未知')}\n"
            f"**目标职位**：{selected_job.get('title', '未知')}\n"
            f"**完成题目**：{len(answers)}/{len(questions)}\n"
            f"**综合评分**：{overall_score}/10\n\n"
            f"---\n\n"
            f"## 各题得分\n\n"
        )
        for i, a in enumerate(answers, 1):
            fallback_report += f"{i}. {a.get('question', '?')[:50]}... — {a.get('score', '?')}/10\n"

        fallback_report += (
            f"\n## 说明\n"
            f"由于报告生成系统暂时出现问题，以上为基础评分汇总。"
            f"综合评分 {overall_score}/10。\n"
        )

        result["final_report"] = fallback_report
        result["overall_score"] = overall_score
        result["messages"] = [AIMessage(content=fallback_report)]

    return result
