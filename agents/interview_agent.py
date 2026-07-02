#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interview Agent — 面试主持 (interview_agent.py)
================================================

职责：
1. 根据简历 + 岗位生成 5 道个性化面试题
2. 逐题提问，等待候选人回答
3. 根据回答质量调整后续问题难度

工具：
- RAG 知识库检索（获取参考面试题和考点）
- LLM 生成面试问题

权限：可写 interview_questions, current_question_idx, interview_stage
"""

import json
import re
import logging
from typing import List

from langchain_core.messages import AIMessage

from agents.state import with_permissions
from agents.utils import safe_llm_call, build_cache_key

logger = logging.getLogger("multi-agent.interview")

# =============================================================================
# System Prompt（面试题生成专用）
# =============================================================================

QUESTION_GEN_PROMPT = """作为一位资深面试官，请为以下候选人针对该职位生成 5 个面试问题。

【候选人简历】
{resume_text}

【目标职位】
{job_text}

{rag_context}
【难度调整】
{difficulty_instruction}

【历史反馈】
{memory_context}

请生成 5 个有针对性的面试问题，涵盖：
1. 技术技能相关（基于简历技能和职位要求）
2. 项目经验相关（基于简历中的工作经历）
3. 问题解决能力相关
4. 团队协作相关
5. 职业规划相关

请以 JSON 格式输出：
{{"questions": ["问题1", "问题2", "问题3", "问题4", "问题5"]}}

注意：只输出 JSON，不要有其他文字。"""


INTERVIEW_CONDUCT_PROMPT = """你是面试主持人。当前面试进行到第 {current}/{total} 题。

你的任务：向候选人提出当前问题，并在回答后引导流程。

规则：
- 每次只问一个问题
- 问题结尾必须有问号
- 如果候选人回答简短，可以说"能详细说说吗？"
- 不要一次性问多个问题

【难度指令】
{difficulty_instruction}

当前问题：{question}

请自然地提问。可以加上简短的开场白（如"好的，下一题："），但不要评价之前的回答。"""


# =============================================================================
# 工具函数
# =============================================================================

def _format_resume_for_prompt(resume_data: dict) -> str:
    """格式化简历数据为 prompt 可用文本"""
    if not resume_data:
        return "（无简历数据）"

    name = resume_data.get('name', '未知')
    skills = resume_data.get('skills', [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',')]
    skills_str = ', '.join(skills) if skills else '无'

    experience = resume_data.get('experience', [])
    if isinstance(experience, list):
        exp_parts = []
        for exp in experience:
            if isinstance(exp, dict):
                exp_parts.append(
                    f"{exp.get('company', '')} {exp.get('position', '')} "
                    f"({exp.get('duration', '')}): {exp.get('description', '')}"
                )
            else:
                exp_parts.append(str(exp))
        exp_str = '\n'.join(exp_parts)
    else:
        exp_str = str(experience)

    return f"姓名：{name}\n技能：{skills_str}\n经历：\n{exp_str}"


def _format_job_for_prompt(job_data: dict) -> str:
    """格式化岗位数据为 prompt 可用文本"""
    if not job_data:
        return "（无岗位数据）"

    return (
        f"职位：{job_data.get('title', '未知')}\n"
        f"公司：{job_data.get('company', '未知')}\n"
        f"地点：{job_data.get('location', '未知')}\n"
        f"职责：{job_data.get('responsibilities', '无')}\n"
        f"要求：{job_data.get('requirements', '无')}"
    )


def _get_rag_context_for_questions(job_title: str, skills: List[str]) -> str:
    """
    从 RAG 知识库检索相关问题上下文。

    返回：格式化的检索结果文本，或空字符串。
    """
    try:
        from knowledge_base import get_rag_chain
        rag = get_rag_chain()
        if rag:
            context = rag.retrieve_for_question_generation(
                job_title=job_title,
                resume_skills=skills if skills else [],
                top_k=4,
            )
            return context if context else ""
    except Exception as e:
        logger.warning(f"[interview] RAG retrieval failed: {e}")
    return ""


def _build_difficulty_instruction(difficulty: str) -> str:
    """
    根据难度等级构建给 LLM 的提问指令。

    参数：
        difficulty: "easy" | "normal" | "hard" | ""（空字符串表示无特殊指令）

    返回：
        str: 难度指令文本
    """
    if not difficulty or difficulty == "normal":
        return "本次提问保持正常难度，不偏难也不偏易。"

    if difficulty == "easy":
        return (
            "请降低难度提问。将问题拆解得更简单，给出适当的提示和引导，"
            "使用更基础、更概念性的表述。如果原问题比较复杂，可以先问一个"
            "更基础的相关问题作为铺垫。语气要鼓励和支持。"
        )

    if difficulty == "hard":
        return (
            "请提高难度提问。要求候选人提供更深入的技术细节、实际项目案例"
            "或量化数据。可以追加追问（如'能举个具体例子吗？''遇到什么困难？如何解决的？'），"
            "考察候选人是否具备真实的一线经验。"
        )

    return "本次提问保持正常难度。"


def _parse_generated_questions(response: str) -> List[str]:
    """解析 LLM 生成的面试题 JSON"""
    # 尝试 JSON 解析
    try:
        # 提取 JSON 部分
        json_str = response.strip()
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0].strip()

        data = json.loads(json_str)
        if isinstance(data, dict) and 'questions' in data:
            questions = [q.strip() for q in data['questions'] if q and len(q.strip()) > 5]
            if len(questions) >= 3:
                return questions[:5]
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # JSON 失败 → 正则提取引号内问题
    quoted = re.findall(r'"([^"]+\?)"', response)
    if len(quoted) >= 3:
        return quoted[:5]

    quoted = re.findall(r'"([^"]+)"', response)
    if len(quoted) >= 3:
        # 过滤太短的
        valid = [q for q in quoted if len(q) > 10]
        if len(valid) >= 3:
            return valid[:5]

    return []


# =============================================================================
# Agent 主节点
# =============================================================================

@with_permissions("interview")
def interview_node(state: dict) -> dict:
    """
    Interview Agent 主节点。

    两种模式：
    1. 无面试题 → 生成 5 道题
    2. 有面试题 → 提出当前题目
    """
    result = {}

    resume_data = state.get("resume_data", {})
    selected_job = state.get("selected_job", {})
    questions = state.get("interview_questions", [])
    current_idx = state.get("current_question_idx", 0)
    thread_id = state.get("_agent_history", ["?"])[-1] if state.get("_agent_history") else "?"

    # ---- 场景 1：生成面试题 ----
    if not questions or len(questions) == 0:
        try:
            # 准备数据
            resume_text = _format_resume_for_prompt(resume_data)
            job_text = _format_job_for_prompt(selected_job)
            skills = resume_data.get('skills', [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(',')]

            # RAG 检索
            job_title = selected_job.get('title', '')
            rag_context = _get_rag_context_for_questions(job_title, skills)

            # 难度调整
            match_score = selected_job.get('match_score', 70)
            if match_score >= 80:
                difficulty = "候选人匹配度高，请侧重深度问题，考察技术细节和复杂场景处理能力。"
            elif match_score >= 60:
                difficulty = "候选人匹配度良好，请平衡广度和深度问题。"
            else:
                difficulty = "候选人匹配度一般，请侧重基础和广度，考察基础知识和学习能力。"

            # 构建 prompt
            prompt = QUESTION_GEN_PROMPT.format(
                resume_text=resume_text,
                job_text=job_text,
                rag_context=rag_context,
                difficulty_instruction=difficulty,
                memory_context="（首次面试，无历史记录）",
            )

            # 调用 LLM（带缓存）
            cache_key = build_cache_key(
                "gen_questions",
                resume_data.get('name', '?'),
                job_title,
                str(skills),
            )
            response = safe_llm_call(
                prompt=prompt,
                thread_id=thread_id,
                cache_key=cache_key,
                temperature=0.8,  # 更高温度增加问题多样性
            )

            # 解析问题
            questions = _parse_generated_questions(response)

            if not questions:
                # 降级：使用默认问题
                questions = [
                    f"请介绍一下你自己，以及你为什么对{job_title}这个职位感兴趣？",
                    f"请详细描述你最成功的项目经历，你在其中扮演什么角色？",
                    f"你如何在团队中协作解决技术难题？请举例说明。",
                    f"如果遇到技术债务和业务需求冲突，你会如何平衡？",
                    f"你对自己未来3年的职业发展有什么规划？",
                ]

            result["interview_questions"] = questions
            result["current_question_idx"] = 0
            result["interview_stage"] = "interviewing"
            result["_agent_signal"] = "FINISH"
            result["messages"] = [AIMessage(
                content=f"面试题已生成（共 {len(questions)} 题）。\n\n"
                f"准备好了吗？让我们开始吧！\n\n"
                f"📋 第 1/{len(questions)} 题：\n{questions[0]}"
            )]

            logger.info(f"[interview] Generated {len(questions)} questions for {job_title}")

        except Exception as e:
            logger.error(f"[interview] Question generation failed: {e}")
            result["messages"] = [AIMessage(
                content=f"生成面试题时遇到问题：{str(e)}。请重试。"
            )]
            result["_agent_signal"] = "FINISH"

        return result

    # ---- 场景 2：提出当前题目（支持难度调整）----
    if 0 <= current_idx < len(questions):
        current_q = questions[current_idx]

        # 读取 evaluator 传来的下一题难度等级
        next_difficulty = state.get("next_difficulty", "")
        difficulty_instruction = _build_difficulty_instruction(next_difficulty)

        try:
            # 使用 LLM 进行难度感知的提问
            conduct_prompt = INTERVIEW_CONDUCT_PROMPT.format(
                current=current_idx + 1,
                total=len(questions),
                difficulty_instruction=difficulty_instruction,
                question=current_q,
            )
            response = safe_llm_call(
                prompt=conduct_prompt,
                thread_id=thread_id,
                temperature=0.7,
            )
            question_msg = response.strip()
        except Exception as e:
            logger.warning(f"[interview] LLM conduct failed: {e}, using fallback")
            # 降级：直接显示题目
            difficulty_hint = ""
            if next_difficulty == "easy":
                difficulty_hint = "（本题为基础难度，放轻松回答即可）"
            elif next_difficulty == "hard":
                difficulty_hint = "（本题为挑战难度，请深入思考后回答）"
            question_msg = (
                f"📋 第 {current_idx + 1}/{len(questions)} 题：\n{current_q}"
                + (f"\n{difficulty_hint}" if difficulty_hint else "")
            )

        result["messages"] = [AIMessage(content=question_msg)]
        result["interview_stage"] = "interviewing"
        result["_agent_signal"] = "FINISH"
        result["next_difficulty"] = ""  # 清空，避免影响后续
        return result

    # 所有题目已完成
    result["interview_stage"] = "interviewing"
    return result
