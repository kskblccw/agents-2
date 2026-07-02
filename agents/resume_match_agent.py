#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResumeMatch Agent — 简历解析 + 岗位匹配 (resume_match_agent.py)
===============================================================

职责：
1. 解析用户上传的简历文件（PDF/DOCX/TXT）
2. 将简历与岗位库进行语义匹配
3. 向用户展示匹配结果并确认选择

工具：
- parse_resume_tool: 解析简历文件
- match_jobs_tool:   执行岗位匹配

权限：可写 resume_data, job_matches, selected_job, interview_stage
"""

import os
import json
import logging
from typing import Dict, Any, List

from langchain_core.messages import AIMessage


from agents.state import with_permissions, MultiAgentState
from agents.utils import safe_llm_call, build_cache_key, get_rate_limiter
from agents.redis_client import get_user_info_store, _extract_thread_id
from agents.supervisor import _has_match_job_intent

logger = logging.getLogger("multi-agent.resume_match")

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """你是简历分析师和岗位匹配专家。

你的任务是一次只做一件事，做完后必须返回 FINISH 信号。

任务流程（分步骤调用，不要一次做完）：
1. 如果用户提供了简历文件路径 → 调用 parse_resume_tool 解析简历 → 返回解析结果 + FINISH
2. 如果用户说"帮我匹配岗位"且简历已存在 → 调用 match_jobs_tool 匹配岗位 → 返回岗位列表 + FINISH
3. 如果用户回复数字选择岗位 → 记录选择 → 返回确认信息 + FINISH

规则：
- 一次只做一个操作，不要连续做多个
- 做完当前任务后立即返回 FINISH，等待用户下一步指令
- 解析失败时告知用户原因
- 不要编造简历里没有的信息
- 展示匹配结果时，简要说明每个岗位为什么匹配"""


# =============================================================================
# 工具函数
# =============================================================================

def _parse_resume_inline(file_path: str) -> dict:
    """
    内联简历解析（直接调用，非 LangChain tool）。

    支持 PDF/DOCX/TXT 格式，使用 resume_parser.py。
    """
    from resume_parser import ResumeParser

    try:
        parser = ResumeParser()
        result = parser.parse_file(file_path)

        if isinstance(result, dict) and result.get('name'):
            return {
                "success": True,
                "data": result,
                "error": None,
            }
        else:
            return {
                "success": False,
                "data": result,
                "error": "解析成功但未提取到姓名，可能不是标准简历格式",
            }
    except FileNotFoundError:
        return {"success": False, "data": None, "error": f"文件不存在: {file_path}"}
    except Exception as e:
        return {"success": False, "data": None, "error": f"简历解析异常: {type(e).__name__}: {str(e)}"}


def _match_jobs_inline(resume_data: dict) -> dict:
    """
    内联岗位匹配 — 完全复制 test_match.py 的调用方式。

    使用 MatcherAgent 进行语义匹配。
    """
    from agent_core import MatcherAgent

    matcher = MatcherAgent()
    matcher.initialize('jobs.json', force_reset=False)

    matches = matcher.match_resume(resume_data, top_k=3)

    if matches:
        return {"success": True, "matches": matches, "error": None}
    else:
        return {"success": False, "matches": [], "error": "未找到匹配的岗位"}


def _safe_get_last_user_msg(messages: list) -> str:
    """安全地从消息列表中提取最近一条用户消息"""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, list):
                content = content[0].get('text', '') if content else ''
            return str(content)
    return ""


def _extract_file_path(text: str) -> str:
    """从用户消息中提取文件路径"""
    import re
    print(f"[DEBUG] 原始文本: {text}")
    patterns = [
        r'([A-Za-z]:[\\/][^\s]+\.(?:pdf|docx?|txt))',
        r'([A-Za-z]:[\\/][\w\s\u4e00-\u9fff\\/]+\.(?:pdf|docx?|txt))',
        r'([\\/][^\s]+\.(?:pdf|docx?|txt))',
        r'([^\s]+\.(?:pdf|docx?|txt))',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"[DEBUG] 匹配到路径: {match.group(1)}")
            return match.group(1)
    print(f"[DEBUG] 所有模式都未匹配")
    return ""


def _extract_job_selection(text: str, matches_count: int) -> int:
    """提取用户选择的岗位序号（1-based），返回 0 表示未识别"""
    text = text.strip()
    print(f"[DEBUG] _extract_job_selection 输入: '{text}', matches_count={matches_count}")
    import re
    patterns = [
        r'选(?:择)?\s*(\d+)',
        r'第\s*(\d+)\s*个',
        r'^(\d+)$',
        r'[第选]?\s*([一二三])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num_str = match.group(1)
            cn_map = {'一': 1, '二': 2, '三': 3}
            if num_str in cn_map:
                print(f"[DEBUG] 匹配中文数字: {num_str} -> {cn_map[num_str]}")
                return cn_map[num_str]
            num = int(num_str)
            if 1 <= num <= matches_count:
                print(f"[DEBUG] 匹配数字: {num} (范围 1-{matches_count})")
                return num
            print(f"[DEBUG] 数字 {num} 超出范围 1-{matches_count}，尝试下一个模式...")
    print(f"[DEBUG] _extract_job_selection: 所有模式未匹配，返回 0")
    return 0


# =============================================================================
# Agent 主节点
# =============================================================================

@with_permissions("resume_match")
def resume_match_node(state: dict) -> dict:
    print(f"[DEBUG] resume_match_node 被调用，selected_job={state.get('selected_job')}, job_matches={len(state.get('job_matches', []))}")
    """
    ResumeMatch Agent 主节点。

    根据当前状态决定执行哪个子任务：
    1. 简历未解析 → 尝试解析
    2. 岗位未匹配 → 尝试匹配
    3. 等待用户选择 → 处理选择
    """
    thread_id = _extract_thread_id(state)  # 统一使用 redis_client 的 ID 提取逻辑
    resume_data = state.get("resume_data", {})
    job_matches = state.get("job_matches", [])
    selected_job = state.get("selected_job")
    messages = state.get("messages", [])

    result = {}

    # ---- 场景 1：简历未解析 ----
    if not resume_data or not resume_data.get("name"):
        try:
            # ---- 检查 Redis 中是否有之前存储的用户信息 ----
            redis_job = None
            redis_user = None
            try:
                store = get_user_info_store()
                redis_user = store.get(thread_id)
                redis_job = store.get_job(thread_id)
                if redis_user or redis_job:
                    logger.info(
                        f"[resume_match] Redis has data for '{thread_id}': "
                        f"user_fields={list(redis_user.keys()) if redis_user else []}, "
                        f"has_job={bool(redis_job)}"
                    )
            except Exception as e:
                logger.warning(f"[resume_match] Redis check skipped: {e}")

            # ---- 如果 Redis 有用户信息 + 上一次的岗位选择，优先询问 ----
            if redis_user and redis_user.get("name") and redis_job and redis_job.get("title"):
                last_user_msg = ""
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'human':
                        raw = msg.content if hasattr(msg, 'content') else ""
                        if isinstance(raw, list):
                            raw = raw[0].get('text', '') if raw else ''
                        last_user_msg = str(raw)
                        break

                # 先恢复简历基本信息到 state（避免后续重复解析）
                restored_resume = {
                    "name": redis_user.get("name", ""),
                    "skills": redis_user.get("skills", "").split(", ") if redis_user.get("skills") else [],
                    "phone": redis_user.get("phone", ""),
                    "email": redis_user.get("email", ""),
                    "school": redis_user.get("school", ""),
                    "summary": redis_user.get("summary", ""),
                    "_from_redis": True,
                }
                result["resume_data"] = restored_resume

                # 检测用户意图：是想用上次的岗位还是想重新匹配
                use_previous = False
                re_match = False
                if last_user_msg:
                    use_previous = any(kw in last_user_msg for kw in [
                        '用上次', '用之前的', '用上一次', '用上回', '之前那个',
                        '上次的岗位', '之前的岗位', '就用上次', '继续用',
                        '是', '是的', '可以', '好的', '行', '对', '嗯',
                    ]) and not any(kw in last_user_msg for kw in ['重新', '换', '不要上次', '不用上次', '不'])
                    re_match = _has_match_job_intent(last_user_msg) and not use_previous

                # 如果用户说"帮我匹配岗位" → 重新匹配（不询问上次的岗位）
                if re_match:
                    # 清空 Redis 中的旧岗位，重新匹配
                    result["job_matches"] = []
                    result["_agent_signal"] = "FINISH"
                    return result

                # 询问用户是否使用上次的岗位
                if not use_previous:
                    job_title = redis_job.get("title", "未知岗位")
                    job_company = redis_job.get("company", "未知公司")
                    job_score = redis_job.get("match_score", 0)
                    result["messages"] = [AIMessage(
                        content=f"检测到您之前选择过岗位：\n\n"
                        f"📌 **{job_title}** @ {job_company}\n"
                        f"匹配度：{job_score}%\n\n"
                        f"请问是否使用该岗位直接开始面试？\n"
                        f"- 回复 **\"是\"** 或 **\"开始面试\"** → 直接进入面试\n"
                        f"- 回复 **\"帮我匹配岗位\"** → 重新匹配其他岗位"
                    )]
                    result["_agent_signal"] = "FINISH"
                    # 不设 waiting_for_user（避免死锁），靠 _agent_signal: FINISH 结束本轮
                    return result

                # 用户确认使用上次的岗位 → 直接设置
                result["selected_job"] = redis_job
                result["job_matches"] = [redis_job]  # 确保 job_matches 非空
                result["interview_stage"] = "matching"
                result["messages"] = [AIMessage(
                    content=f"已恢复上次选择的岗位：{redis_job.get('title', '未知')} @ "
                    f"{redis_job.get('company', '未知')}\n"
                    f"匹配度：{redis_job.get('match_score', 0)}%\n\n"
                    f"准备开始模拟面试！请输入'开始面试'。"
                )]
                result["_agent_signal"] = "FINISH"
                return result

            # ---- Redis 有用户信息但没有岗位（或只有简历信息），
            #      且用户明确要求匹配岗位 → 先恢复简历，走匹配流程 ----
            if redis_user and redis_user.get("name") and _has_match_job_intent(
                _safe_get_last_user_msg(messages)
            ):
                restored_resume = {
                    "name": redis_user.get("name", ""),
                    "skills": redis_user.get("skills", "").split(", ") if redis_user.get("skills") else [],
                    "phone": redis_user.get("phone", ""),
                    "email": redis_user.get("email", ""),
                    "school": redis_user.get("school", ""),
                    "summary": redis_user.get("summary", ""),
                    "_from_redis": True,
                }
                result["resume_data"] = restored_resume
                # 不设 _agent_signal，让 supervisor 重新路由（此时 resume_data 非空 → 进入场景 2 匹配岗位）
                result["_agent_signal"] = ""
                return result

            # ---- 无 Redis 数据：从用户消息中提取文件路径 ----
            last_user_msg = _safe_get_last_user_msg(messages)

            file_path = _extract_file_path(str(last_user_msg))

            if not file_path:
                if state.get("_asked_for_resume"):
                    logger.info(
                        "[resume_match] Already asked for resume — signaling FINISH"
                    )
                    return {"_agent_signal": "FINISH"}

                # 首次询问简历路径，必须带上 _agent_signal: FINISH，
                # 否则 supervisor 会重复路由到 resume_match 造成循环
                return {
                    **result,
                    "messages": [AIMessage(
                        content="请提供您的简历文件路径（支持 PDF、DOCX、TXT 格式）。"
                        "\n例如：e:/my_resume.pdf 或 /home/user/resume.docx"
                    )],
                    "_asked_for_resume": True,
                    "_agent_signal": "FINISH",
                    "interview_stage": "parsing",
                }

            parse_result = _parse_resume_inline(file_path)

            if parse_result["success"]:
                result["resume_data"] = parse_result["data"]
                result["interview_stage"] = "parsing"

                # ---- 将个人信息持久化到 Redis ----
                try:
                    _data = parse_result["data"]

                    # 提取联系方式
                    _contact = _data.get("contact", {})
                    if not isinstance(_contact, dict):
                        _contact = {}
                    _phone = _contact.get("phone", "")
                    _email = _contact.get("email", "")

                    # 提取教育信息
                    _education = _data.get("education", [])
                    _school = ""
                    if isinstance(_education, list) and _education:
                        _edu = _education[0] if isinstance(_education[0], dict) else {}
                        _school = _edu.get("school", "") or _edu.get("college", "") or ""

                    # 提取技能
                    _skills = _data.get("skills", [])
                    if isinstance(_skills, list):
                        _skills_str = ", ".join(_skills)
                    elif isinstance(_skills, str):
                        _skills_str = _skills
                    else:
                        _skills_str = ""

                    # 构建用户信息（包含 birthday，简历可能不含但后续可从 RAG 补全）
                    user_info = {
                        "name": _data.get("name", ""),
                        "birthday": _data.get("birthday", ""),
                        "phone": _phone,
                        "email": _email,
                        "school": _school,
                        "skills": _skills_str,
                        "summary": (_data.get("summary", "") or "")[:200],
                    }

                    store = get_user_info_store()
                    store.store(thread_id, user_info)
                    logger.info(
                        f"[resume_match] User info stored to Redis: "
                        f"thread_id={thread_id}, fields={[k for k, v in user_info.items() if v]}"
                    )
                except Exception as e:
                    logger.warning(f"[resume_match] Redis store skipped: {e}")

                skills = parse_result["data"].get("skills", [])
                skills_str = ', '.join(skills) if isinstance(skills, list) else str(skills)
                result["messages"] = [AIMessage(
                    content=f"简历解析成功！\n"
                    f"- 姓名：{parse_result['data'].get('name', '未知')}\n"
                    f"- 技能：{skills_str}\n"
                    f"- 经验：{parse_result['data'].get('experience', '无')}\n\n"
                    f"请输入'帮我匹配岗位'进行岗位匹配。"
                )]
                result["_agent_signal"] = "FINISH"
            else:
                result["messages"] = [AIMessage(
                    content=f"简历解析失败：{parse_result['error']}\n请检查文件路径和格式后重试。"
                )]
                result["_agent_signal"] = "FINISH"
                result["_asked_for_resume"] = True

        except Exception as e:
            logger.error(f"[resume_match] Parse error: {e}")
            result["messages"] = [AIMessage(
                content=f"处理简历时遇到问题：{str(e)}。请检查文件后重试。"
            )]
            result["interview_stage"] = "parsing"
            result["_agent_signal"] = "FINISH"
            result["_asked_for_resume"] = True

        return result

    # ---- 场景 1.5：Redis 恢复的简历，用户确认使用上次岗位 ----
    if resume_data and resume_data.get("_from_redis") and (not job_matches or len(job_matches) == 0):
        last_user_msg = _safe_get_last_user_msg(messages)
        print(f"[DEBUG] 场景1.5: Redis恢复的简历, last_user_msg='{last_user_msg}'")

        # 检查 Redis 中是否有缓存的岗位
        redis_job = None
        try:
            store = get_user_info_store()
            redis_job = store.get_job(thread_id)
        except Exception:
            pass

        # 用户确认使用上一次岗位 → 直接设置
        if redis_job and redis_job.get("title") and last_user_msg:
            is_confirm = any(kw in last_user_msg for kw in [
                '是', '是的', '可以', '好的', '行', '对', '嗯', '使用',
                '开始面试', '开始模拟面试', '启动面试', '进入面试', '开始吧',
            ]) and not any(kw in last_user_msg for kw in [
                '重新', '换', '不要', '不用', '帮我匹配', '匹配岗位', '匹配职位',
            ])

            if is_confirm:
                print(f"[DEBUG] 场景1.5: 用户确认使用上次岗位 → 直接设置")
                result["selected_job"] = redis_job
                result["job_matches"] = [redis_job]  # 确保 job_matches 非空
                result["interview_stage"] = "matching"
                # 清除 _from_redis 标记
                clean_resume = dict(resume_data)
                clean_resume.pop("_from_redis", None)
                result["resume_data"] = clean_resume
                result["messages"] = [AIMessage(
                    content=f"已恢复上次选择的岗位：{redis_job.get('title', '未知')} @ "
                    f"{redis_job.get('company', '未知')}\n"
                    f"匹配度：{redis_job.get('match_score', 0)}%\n\n"
                    f"准备开始模拟面试！请输入'开始面试'。"
                )]
                result["_agent_signal"] = "FINISH"
                return result

            # 用户要求重新匹配 → 清空 Redis 岗位标记，正常走匹配流程
            if _has_match_job_intent(last_user_msg):
                print(f"[DEBUG] 场景1.5: 用户要求重新匹配 → 正常走匹配流程")
                # 清除 _from_redis 标记，正常走后续流程
                clean_resume = dict(resume_data)
                clean_resume.pop("_from_redis", None)
                result["resume_data"] = clean_resume
                # 不设 _agent_signal，让后续场景 2 执行匹配

    # ---- 场景 2：岗位未匹配 ----
    if not job_matches or len(job_matches) == 0:
        print("[DEBUG] 进入岗位匹配分支")
        result["waiting_for_user"] = True

        try:
            cache_key = build_cache_key("match_jobs", str(resume_data.get("skills", "")))
            match_result = _match_jobs_inline(resume_data)

            if match_result["success"] and match_result["matches"]:
                result["job_matches"] = match_result["matches"]
                result["interview_stage"] = "matching"

                match_text = "为您匹配到以下岗位：\n\n"
                for i, m in enumerate(match_result["matches"], 1):
                    title = m.get('title', '未知')
                    company = m.get('company', '未知')
                    score = m.get('match_score', 0)
                    reason = m.get('reason', '')
                    match_text += f"{i}. {title} @ {company}（匹配度：{score}%）\n"
                    if reason:
                        reason_short = reason[:100].replace('\n', ' ')
                        match_text += f"   {reason_short}...\n"
                    match_text += "\n"
                match_text += "请选择一个岗位（回复数字 1-3）开始模拟面试。"

                result["messages"] = [AIMessage(content=match_text)]
                result["_agent_signal"] = "FINISH"
                result["waiting_for_user"] = False
            else:
                result["messages"] = [AIMessage(
                    content=f"岗位匹配失败：{match_result.get('error', '未知错误')}\n"
                    f"当前岗位库可能不支持您的技能方向。"
                )]
                result["interview_stage"] = "matching"
                result["_agent_signal"] = "FINISH"
                result["waiting_for_user"] = False

        except Exception as e:
            logger.error(f"[resume_match] Match error: {e}")
            result["messages"] = [AIMessage(content=f"岗位匹配时遇到问题：{str(e)}")]
            result["interview_stage"] = "matching"
            result["_agent_signal"] = "FINISH"
            result["waiting_for_user"] = False
            print("[DEBUG] 准备返回，_agent_signal =", result.get("_agent_signal"))
        print(f"[DEBUG] 场景2 返回，_agent_signal='{result.get('_agent_signal')}' (空=等待用户选择)")
        return result

    # ---- 场景 3：等待用户选择岗位 ----
    if job_matches and not selected_job:
        print(f"[DEBUG] 进入场景3（岗位选择），selected_job={selected_job}")
        last_user_msg = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                raw = msg.content if hasattr(msg, 'content') else ""
                if isinstance(raw, list):
                    raw = raw[0].get('text', '') if raw else ''
                last_user_msg = str(raw)
                break

        # ---- 检测重新匹配意图：用户想根据最新简历重新匹配岗位 ----
        if last_user_msg and _has_match_job_intent(last_user_msg):
            print(f"[DEBUG] 场景3: 检测到重新匹配意图 → 清除旧匹配，重新进入场景2")
            result["job_matches"] = []  # 清空旧匹配
            result["selected_job"] = {}  # 清空旧选择
            result["_agent_signal"] = "FINISH"  # 先结束本轮，让 supervisor 重新路由到 resume_match(场景2)
            return result

        choice = _extract_job_selection(str(last_user_msg), len(job_matches))
        print(f"[DEBUG] 场景3 choice={choice}, last_user_msg='{last_user_msg}'")

        # 如果用户说"开始面试"但没有选岗位 → 自动选择第一个匹配
        is_start_intent = any(kw in str(last_user_msg) for kw in
            ['开始面试', '开始模拟面试', '启动面试', '进入面试', '开始吧'])

        if is_start_intent and not (last_user_msg.strip().isdigit()):
            choice = 1  # 自动选第一个
            print(f"[DEBUG] 场景3: 检测到开始面试意图，自动选择第 1 个岗位")

        if choice > 0:
            chosen = job_matches[choice - 1]
            result["selected_job"] = {
                "title": chosen.get('title', '未知'),
                "company": chosen.get('company', '未知'),
                "location": chosen.get('location', ''),
                "responsibilities": chosen.get('responsibilities', ''),
                "requirements": chosen.get('requirements', ''),
                "match_score": chosen.get('match_score', 0),
            }
            print(f"[DEBUG] 设置 selected_job = {chosen.get('title')}")

            # ---- 将选中的岗位持久化到 Redis（供下次会话使用） ----
            try:
                store = get_user_info_store()
                store.store_job(thread_id, result["selected_job"])
                logger.info(
                    f"[resume_match] Selected job cached to Redis: "
                    f"{chosen.get('title')} @ {chosen.get('company')}"
                )
            except Exception as e:
                logger.warning(f"[resume_match] Redis job cache skipped: {e}")

            result["interview_stage"] = "matching"

            if is_start_intent:
                # 用户意图是开始面试 → 直接告知并将路由权交给 supervisor
                result["messages"] = [AIMessage(
                    content=f"已自动选择：{chosen.get('title', '未知')} @ {chosen.get('company', '未知')}\n"
                    f"匹配度：{chosen.get('match_score', 0)}%\n\n"
                    f"正在开始模拟面试..."
                )]
                # 不设 _agent_signal，让 supervisor 重新路由到 interview
                result["_agent_signal"] = ""
                result["waiting_for_user"] = False
            else:
                result["messages"] = [AIMessage(
                    content=f"已选择：{chosen.get('title', '未知')} @ {chosen.get('company', '未知')}\n"
                    f"匹配度：{chosen.get('match_score', 0)}%\n\n"
                    f"准备开始模拟面试！请输入'开始面试'。"
                )]
                result["_agent_signal"] = "FINISH"
        else:
            # 用户输入不是数字也不是岗位选择 → 重新显示岗位列表
            match_text = "请选择一个岗位（回复数字 1-3）开始模拟面试。\n\n"
            match_text += "为您匹配到以下岗位：\n\n"
            for i, m in enumerate(job_matches, 1):
                title = m.get('title', '未知')
                company = m.get('company', '未知')
                score = m.get('match_score', 0)
                reason = m.get('reason', '')
                match_text += f"{i}. {title} @ {company}（匹配度：{score}%）\n"
                if reason:
                    reason_short = reason[:100].replace('\n', ' ')
                    match_text += f"   {reason_short}...\n"
                match_text += "\n"
            result["messages"] = [AIMessage(content=match_text)]
            result["_agent_signal"] = "FINISH"
        print(f"[DEBUG] 场景3 返回，_agent_signal={result.get('_agent_signal')}, selected_job={result.get('selected_job')}")

        return result

    # ---- 场景 4：已全部完成 ----
    # 但如果用户想重新匹配岗位，允许重做
    last_user_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            raw = msg.content if hasattr(msg, 'content') else ""
            if isinstance(raw, list):
                raw = raw[0].get('text', '') if raw else ''
            last_user_msg = str(raw)
            break

    # 重新匹配意图 → 清空旧匹配，不设 FINISH 信号，
    # 让 supervisor 在同一次调用中重新路由到场景 2 执行匹配
    if last_user_msg and _has_match_job_intent(last_user_msg):
        print(f"[DEBUG] 场景4: 检测到重新匹配意图 → 清除旧匹配，立即进入场景2")
        result["job_matches"] = []
        result["selected_job"] = {}
        # 关键：不设 _agent_signal，让 supervisor 重新路由到 resume_match（场景2）
        return result

    # 已有岗位和简历 → 直接提示可以开始面试
    if selected_job and resume_data and resume_data.get("name"):
        result["messages"] = [AIMessage(
            content=f"您已选择岗位：{selected_job.get('title', '未知')} @ "
            f"{selected_job.get('company', '未知')}\n\n"
            f"请输入'开始面试'开始模拟面试，或输入'帮我匹配岗位'重新匹配。"
        )]
    result["_agent_signal"] = "FINISH"
    return result