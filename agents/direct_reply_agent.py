#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectReply Agent — 通用问答（无简历时） (direct_reply_agent.py)
===============================================================

职责：
1. 回答技术概念问题（"什么是 GIL？"、"FastAPI 怎么用？"）
2. 提供面试建议和咨询
3. 处理一般性问候和闲聊
4. 温和提醒用户可以提供简历以使用完整面试功能

设计原则：
- 一次只回答一个问题，回答完立即返回 FINISH
- 使用轻量级模型（默认 deepseek-chat，可通过 DIRECT_REPLY_MODEL 切换）
- 支持多模型：DeepSeek / 阿里云百炼 / 智谱
- 不使用简历数据、不操作岗位、不生成面试题

权限：可写 _agent_signal, _direct_reply_count
"""

import os
import logging
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agents.state import with_permissions
from agents.redis_client import get_user_info_store, _extract_thread_id

# 从 supervisor 导入路由检测函数（避免重复定义）
try:
    from agents.supervisor import _has_personal_info_intent
except ImportError:
    # 如果导入失败（循环依赖防御），定义本地版本
    def _has_personal_info_intent(text: str) -> bool:
        if not text:
            return False
        # 简历上传 → 不是
        if any(kw in text.lower() for kw in ['.pdf', '.docx', '简历', 'resume', '上传']):
            return False
        first_person = ["我", "我的", "我叫", "我是"]
        fields = ["名字", "姓名", "叫什么", "是谁", "生日", "出生", "电话", "手机",
                   "邮箱", "email", "学校", "大学", "学院", "学历", "专业"]
        if any(m in text for m in first_person) and any(f in text for f in fields):
            return True
        if any(kw in text for kw in ["你知道我", "你记得我", "我的简历", "我的信息", "我是谁"]):
            return True
        return False

logger = logging.getLogger("multi-agent.direct_reply")

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """你是 AI 面试系统的智能助手。当用户没有上传简历但询问技术问题、面试咨询或进行一般性聊天时，由你负责回答。

## 回答规则
1. **简洁专业**：回答控制在 200-500 字以内，不要过于冗长
2. **技术问题**：给出清晰的定义 + 关键要点 + 简短示例（如适用）
3. **面试建议**：给出 2-3 条实用可操作的建议
4. **问候闲聊**：友好回应，简单介绍系统功能
5. **身份定位**：你是面试系统的助手，不是正式面试官

## 温和提醒（重要）
- 回答末尾可以用一句话提醒用户提供简历，但**不要强制或压迫用户**
- 提醒示例："💡 如果您想体验完整的 AI 模拟面试，可以提供您的简历文件路径。"
- 如果用户明确表示不需要，就不要反复提醒

## 禁止行为
- ❌ 不要假装你能解析简历（你没有这个能力）
- ❌ 不要生成面试题目
- ❌ 不要评价用户的技术水平
- ❌ 不要编造用户简历信息"""


# =============================================================================
# LLM 工厂函数 — 多模型支持
# =============================================================================

def _get_direct_reply_llm() -> ChatOpenAI:
    """
    根据环境变量 DIRECT_REPLY_MODEL 创建对应的 LLM 客户端。

    支持的模型前缀：
    - deepseek-* → DeepSeek API（默认）
    - qwen-*     → 阿里云百炼（DashScope）
    - glm-*      → 智谱 AI（BigModel）

    环境变量：
    - DIRECT_REPLY_MODEL: 模型名称（默认 "deepseek-chat"）
    - DEEPSEEK_API_KEY:   DeepSeek API Key
    - DASHSCOPE_API_KEY:  阿里云百炼 API Key
    - ZHIPU_API_KEY:      智谱 API Key
    """
    model_name = os.getenv("DIRECT_REPLY_MODEL", "deepseek-chat")

    if model_name.startswith("qwen-"):
        # 阿里云百炼（DashScope）— OpenAI 兼容接口
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "使用 qwen 系列模型需要设置 DASHSCOPE_API_KEY 环境变量"
            )
        logger.info(f"[direct_reply] Using DashScope model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            request_timeout=30,
        )

    elif model_name.startswith("glm-"):
        # 智谱 AI（BigModel）— OpenAI 兼容接口
        api_key = os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError(
                "使用 glm 系列模型需要设置 ZHIPU_API_KEY 环境变量"
            )
        logger.info(f"[direct_reply] Using Zhipu model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            request_timeout=30,
        )

    else:
        # DeepSeek（默认）
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "使用 DeepSeek 模型需要设置 DEEPSEEK_API_KEY 环境变量"
            )
        logger.info(f"[direct_reply] Using DeepSeek model: {model_name}")
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            request_timeout=30,
        )


# 延迟初始化 LLM 单例
_direct_reply_llm: ChatOpenAI = None


def _get_llm() -> ChatOpenAI:
    """获取 direct_reply 专用 LLM 单例"""
    global _direct_reply_llm
    if _direct_reply_llm is None:
        _direct_reply_llm = _get_direct_reply_llm()
    return _direct_reply_llm


# =============================================================================
# 个人信息查询辅助函数
# =============================================================================


def _detect_personal_query_type(text: str) -> str:
    """
    检测用户正在查询哪个个人信息字段。

    参数：
        text: 用户消息

    返回：
        str: 字段名 ('name'|'birthday'|'phone'|'email'|'school'|'all')
              'all' 表示查全部信息
    """
    mapping = {
        'name': ['名字', '姓名', '叫什么', '我是谁', '是谁'],
        'birthday': ['生日', '出生', '几月几号', '出生日期'],
        'phone': ['电话', '手机', '联系方式', '号码'],
        'email': ['邮箱', 'email', '邮件'],
        'school': ['学校', '大学', '学院', '学历', '专业'],
    }
    for field, keywords in mapping.items():
        if any(kw in text for kw in keywords):
            return field
    return 'all'  # 无法确定具体字段 → 查全部


def _query_user_info(thread_id: str, state: dict, query_type: str = 'all') -> dict:
    """
    多级查询用户信息：Redis → state resume_data → 无数据。

    注意：不再回退到 RAG（知识库），因为 RAG 返回的是通用知识库内容
    （如 cxq.md 的候选人信息），不是当前用户的数据，会导致跨用户数据泄露。

    参数：
        thread_id: 会话 ID
        state: 当前 MultiAgentState
        query_type: 查询类型 ('name'|'birthday'|...|'all')

    返回：
        {"found": bool, "data": dict, "source": str}
    """
    logger.info(
        f"[direct_reply] Querying user info: thread_id={thread_id}, "
        f"query_type={query_type}"
    )

    # === Level 1: Redis（优先，最快） ===
    store = get_user_info_store()
    if query_type == 'all':
        redis_data = store.get(thread_id)
    else:
        redis_data = store.get(thread_id, query_type)

    if redis_data and any(v for v in redis_data.values()):
        logger.info(
            f"[direct_reply] Redis HIT: thread_id={thread_id}, "
            f"fields={list(redis_data.keys())}"
        )
        return {"found": True, "data": redis_data, "source": "redis"}

    logger.info(
        f"[direct_reply] Redis MISS for thread_id={thread_id}, "
        f"falling back to state resume_data"
    )

    # === Level 2: state resume_data ===
    resume_data = state.get("resume_data", {})
    if resume_data and resume_data.get("name"):
        info = _extract_user_info_from_resume(resume_data)
        if info:
            # 补写 Redis（下次直接命中）
            try:
                store.store(thread_id, info)
                logger.info(
                    f"[direct_reply] Backfilled Redis from state: "
                    f"thread_id={thread_id}, fields={list(info.keys())}"
                )
            except Exception as e:
                logger.warning(f"[direct_reply] Redis backfill failed: {e}")
        field_data = _filter_by_query_type(info, query_type)
        if field_data:
            return {"found": True, "data": field_data, "source": "state"}

    # === Level 3: 无数据（不再回退到 RAG） ===
    logger.info(
        f"[direct_reply] No user info found: thread_id={thread_id}, "
        f"has_resume_data={bool(resume_data)}"
    )
    return {"found": False, "data": {}, "source": "none"}


def _extract_user_info_from_resume(resume_data: dict) -> dict:
    """
    从 resume_data 中提取用户信息字段。

    参数：
        resume_data: 简历解析结果

    返回：
        dict: {name, birthday, phone, email, school, skills, summary}
    """
    info = {}

    # 基本信息
    if resume_data.get("name"):
        info["name"] = resume_data["name"]
    if resume_data.get("birthday"):
        info["birthday"] = resume_data["birthday"]

    # contact 可能是 dict 或 string
    contact = resume_data.get("contact", {})
    if isinstance(contact, dict):
        if contact.get("phone"):
            info["phone"] = contact["phone"]
        if contact.get("email"):
            info["email"] = contact["email"]
    elif isinstance(contact, str):
        info["phone"] = contact  # 降级

    # education 可能是 list[dict]
    education = resume_data.get("education", [])
    if isinstance(education, list) and education:
        edu = education[0] if isinstance(education[0], dict) else {}
        school = edu.get("school", "") or edu.get("college", "")
        if school:
            info["school"] = school

    # skills
    skills = resume_data.get("skills", [])
    if isinstance(skills, list):
        info["skills"] = ", ".join(skills)
    elif isinstance(skills, str):
        info["skills"] = skills

    # summary
    if resume_data.get("summary"):
        info["summary"] = resume_data["summary"][:200]

    return info


def _filter_by_query_type(info: dict, query_type: str) -> dict:
    """按查询类型过滤信息字段。query_type='all' 返回全部非空字段。"""
    if not info:
        return {}
    if query_type == 'all':
        return {k: v for k, v in info.items() if v}
    value = info.get(query_type, "")
    return {query_type: value} if value else {}


# ---- 回复格式化 ----

_RESPONSE_TEMPLATES = {
    'name': "根据您的简历，您的姓名是 **{value}**。",
    'birthday': "根据您的简历信息，您的生日是 **{value}**。",
    'phone': "您在简历中登记的电话是 **{value}**。",
    'email': "您在简历中登记的邮箱是 **{value}**。",
    'school': "根据您的简历，您就读于 **{value}**。",
}


def _format_user_info_response(user_info: dict, query_type: str, state: dict) -> str:
    """
    将用户信息格式化为自然语言回复。

    参数：
        user_info: {"found": True, "data": {...}, "source": "redis"|"state"|"rag"}
        query_type: 查询的字段名
        state: 当前状态

    返回：
        str: 人类可读的回复文本
    """
    source = user_info.get("source", "unknown")
    data = user_info.get("data", {})

    # RAG 来源（文本格式，非结构化）
    if source == "rag":
        rag_text = data.get("_rag_text", "")
        return (
            f"根据知识库信息：\n\n{rag_text}\n\n"
            "💡 如果您想更新个人信息，可以上传最新简历。"
        )

    # 结构化数据
    if query_type != 'all' and query_type in data:
        value = data[query_type]
        template = _RESPONSE_TEMPLATES.get(query_type, "您的 **{field}** 是 **{value}**。")
        reply = template.format(value=value, field=query_type)
        source_note = ""
        if source == "state":
            source_note = "\n（此信息来自当前会话，暂时未持久化存储。）"
        return reply + source_note

    if query_type == 'all' and data:
        lines = ["根据已有的信息："]
        for field, label in [
            ('name', '姓名'), ('birthday', '生日'), ('phone', '电话'),
            ('email', '邮箱'), ('school', '学校'), ('skills', '技能'),
        ]:
            val = data.get(field, "")
            if val:
                lines.append(f"- {label}：{val}")
        lines.append("\n💡 如信息有误，可以重新上传最新简历。")
        return "\n".join(lines)

    # 无特定字段
    return "我目前没有您的个人信息记录。请先上传简历文件，我就能记住您了！"


def _is_greeting(text: str) -> bool:
    """检测用户消息是否为打招呼/问候"""
    if not text:
        return False
    greetings = {
        "hi", "hello", "hey", "你好", "您好", "早上好", "晚上好",
        "哈喽", "嗨", "在吗", "在不在", "good morning", "good afternoon",
        "good evening", "what's up", "how are you", "最近怎么样",
    }
    text_stripped = text.strip().lower()
    if text_stripped in greetings:
        return True
    if len(text_stripped) < 30 and any(g in text_stripped for g in greetings):
        return True
    # 问系统能做什么
    capability_q = ["你能做什么", "你会什么", "你能干嘛", "你可以做什么", "你能帮我什么"]
    if any(q in text_stripped for q in capability_q):
        return True
    return False


def _build_greeting_with_resume(state: dict) -> str:
    """构建有简历时的个性化问候回复（展示当前状态）"""
    resume_data = state.get("resume_data", {})
    name = resume_data.get("name", "用户")
    has_selected_job = bool(state.get("selected_job"))
    has_report = bool(state.get("final_report"))
    questions = state.get("interview_questions", [])
    answers = state.get("answers", [])

    lines = [f"您好，{name}！我已看到您的简历信息。"]
    lines.append("")

    if has_report:
        overall = state.get("overall_score", "?")
        lines.append(f"📊 您已完成一次模拟面试，综合评分：**{overall}/10**。")
        lines.append("")
        lines.append("您可以：")
        if has_selected_job:
            job_title = state.get("selected_job", {}).get("title", "未知岗位")
            lines.append(f"- 输入 **\"开始面试\"** 对 **{job_title}** 重新开始面试")
        lines.append("- 输入 **\"帮我匹配岗位\"** 重新匹配其他岗位")
        lines.append("- 直接问我技术问题（如\"什么是 GIL？\"）")
        lines.append("- 询问个人信息（如\"我的生日是多久？\"）")
    elif questions and len(answers) < len(questions):
        remaining = len(questions) - len(answers)
        lines.append(f"🎯 您正在进行模拟面试，还有 **{remaining}** 道题待回答。")
        lines.append("")
        lines.append("继续回答面试题，或输入 **\"结束面试\"** 提前终止。")
    elif has_selected_job:
        job_title = state.get("selected_job", {}).get("title", "未知岗位")
        lines.append("您可以：")
        lines.append(f"- 输入 **\"开始面试\"** 开始 **{job_title}** 的模拟面试")
        lines.append("- 输入 **\"帮我匹配岗位\"** 重新匹配其他岗位")
        lines.append("- 直接问我技术问题（如\"什么是 GIL？\"）")
        lines.append("- 询问个人信息（如\"我的生日是多久？\"）")
    else:
        lines.append("您可以：")
        lines.append("- 输入 **\"帮我匹配岗位\"** 来匹配合适的工作")
        lines.append("- 直接问我技术问题（如\"什么是 GIL？\"）")
        lines.append("- 询问个人信息（如\"我的生日是多久？\"）")

    return "\n".join(lines)

    return "\n".join(lines)


def _build_no_info_response(state: dict) -> dict:
    """构建「无个人信息」的回复"""
    return {
        "messages": [AIMessage(
            content="我目前没有您的个人信息。\n\n"
            "请先上传您的简历文件（支持 PDF、DOCX、TXT 格式），"
            "我就能解析并记住您的姓名、联系方式、学校等信息。"
        )],
        "_agent_signal": "FINISH",
        "_direct_reply_count": state.get("_direct_reply_count", 0) + 1,
    }


# =============================================================================
# Agent 主节点
# =============================================================================

@with_permissions("direct_reply")
def direct_reply_node(state: dict) -> dict:
    """
    DirectReply Agent 主节点。

    处理不需要简历的通用问答（技术问题、面试咨询、闲聊等）。
    每次调用只回答一个问题，完成后返回 FINISH 信号。

    参数：
        state: MultiAgentState 字典

    返回：
        dict: 包含 AIMessage 回复 + _agent_signal + _direct_reply_count
    """
    messages = state.get("messages", [])

    # ---- 提取最近一条用户消息 ----
    last_user_msg = ""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'human':
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, list):
                content = content[0].get('text', '') if content else ''
            last_user_msg = str(content)
            break

    if not last_user_msg.strip():
        return {"_agent_signal": "FINISH"}

    # ---- 个人信息查询快速路径（不调用 LLM，直接查 Redis/state/RAG） ----
    if _has_personal_info_intent(last_user_msg):
        thread_id = _extract_thread_id(state)
        query_type = _detect_personal_query_type(last_user_msg)
        user_info = _query_user_info(thread_id, state, query_type)

        if user_info["found"]:
            answer = _format_user_info_response(user_info, query_type, state)
            return {
                "messages": [AIMessage(content=answer)],
                "_agent_signal": "FINISH",
                "_direct_reply_count": state.get("_direct_reply_count", 0) + 1,
            }
        else:
            return _build_no_info_response(state)

    # ---- 问候快速路径：检测到打招呼 + 有简历 → 个性化回复 ----
    if _is_greeting(last_user_msg):
        has_resume = (
            state.get("resume_data", {}).get("name") or
            state.get("selected_job")
        )
        if has_resume:
            answer = _build_greeting_with_resume(state)
            return {
                "messages": [AIMessage(content=answer)],
                "_agent_signal": "FINISH",
                "_direct_reply_count": state.get("_direct_reply_count", 0) + 1,
            }

    # ---- 构建最近对话上下文（最多 3 轮） ----
    recent_msgs = []
    for msg in messages[-6:]:
        role = "用户" if (hasattr(msg, 'type') and msg.type == 'human') else "助手"
        content = msg.content if hasattr(msg, 'content') else ""
        if isinstance(content, list):
            content = content[0].get('text', '') if content else ''
        content_str = str(content)
        if len(content_str) > 300:
            content_str = content_str[:300] + "..."
        recent_msgs.append(f"[{role}] {content_str}")

    context = "\n".join(recent_msgs)

    # ---- 构建 prompt ----
    prompt = f"""对话历史：
{context}

请回答用户最后的问题：{last_user_msg}

要求：
- 简洁专业，200-500 字
- 如果是技术问题，给出清晰定义和关键点
- 如果是面试咨询，给出实用建议
- 回答末尾可以用一句话温和提醒用户可以上传简历"""

    # ---- 调用 LLM ----
    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        answer = response.content.strip()

        # 递增计数器
        dr_count = state.get("_direct_reply_count", 0) + 1

        return {
            "messages": [AIMessage(content=answer)],
            "_agent_signal": "FINISH",
            "_direct_reply_count": dr_count,
        }

    except Exception as e:
        logger.error(f"[direct_reply] LLM call failed: {e}")
        dr_count = state.get("_direct_reply_count", 0) + 1

        # 降级回复（不依赖 LLM）
        fallback = (
            f"抱歉，处理您的问题时遇到了临时故障（{type(e).__name__}）。\n\n"
            "您可以：\n"
            "1. 稍后重试您的问题\n"
            "2. 提供简历文件路径以开始完整的模拟面试体验"
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "_agent_signal": "FINISH",
            "_direct_reply_count": dr_count,
        }
