#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 面试官 Web 服务 (web_server.py)
===================================
基于 FastAPI 的多 Agent 面试系统 Web 展示端。

架构：
    FastAPI → LangGraph Supervisor 多 Agent 图 → DeepSeek LLM
    ChatStore (Redis → SQLite → Memory) 管理聊天历史

启动：
    uvicorn web_server:app --reload --host 0.0.0.0 --port 8000
访问：
    http://localhost:8000

依赖：
    pip install python-multipart redis
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# ==================== 环境变量 ====================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.getenv("DEEPSEEK_API_KEY"):
    print("[WARN] DEEPSEEK_API_KEY not set — LLM calls will fail")

# ==================== FastAPI 初始化 ====================
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web_server")

app = FastAPI(title="AI 面试官 Web", version="1.0.0")

# ==================== 全局状态 ====================
graph = None          # LangGraph 编译后的图
checkpointer = None   # 检查点存储器
chat_store = None     # ChatStore 实例


# =============================================================================
# ChatStore — 聊天历史持久化（Redis → SQLite → Memory）
# =============================================================================

class ChatStore:
    """
    聊天历史三层存储：Redis → SQLite → 内存。

    Redis 数据结构：
        chat:sessions          → JSON array of session metadata
        chat:history:{sid}     → JSON array of messages {role, content, timestamp}
    """

    def __init__(self):
        self._backend = None     # "redis" | "sqlite" | "memory"
        self._redis = None
        self._sqlite_conn = None
        self._sessions = []      # memory fallback: list of session dicts
        self._history = {}       # memory fallback: {session_id: [messages]}

        self._init_redis()
        if not self._backend:
            self._init_sqlite()
        if not self._backend:
            self._init_memory()

        logger.info(f"[ChatStore] Using backend: {self._backend}")

    # ---- Redis ----
    def _init_redis(self):
        try:
            import redis as redis_lib
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            r = redis_lib.Redis.from_url(
                redis_url,
                socket_connect_timeout=3,
                socket_timeout=3,
                decode_responses=True,
            )
            r.ping()
            self._redis = r
            self._backend = "redis"
            logger.info("[ChatStore] Redis connected")
        except Exception as e:
            logger.warning(f"[ChatStore] Redis unavailable: {e}")

    # ---- SQLite ----
    def _init_sqlite(self):
        try:
            import sqlite3
            db_dir = Path("data")
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "web_chat.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS web_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES web_sessions(session_id)
                )
            """)
            conn.commit()
            self._sqlite_conn = conn
            self._backend = "sqlite"
            logger.info(f"[ChatStore] SQLite connected at {db_path}")
        except Exception as e:
            logger.warning(f"[ChatStore] SQLite unavailable: {e}")

    # ---- Memory ----
    def _init_memory(self):
        self._backend = "memory"
        logger.warning("[ChatStore] Using in-memory storage (data lost on restart)")

    # ============ Public API ============

    def get_sessions(self) -> list:
        """返回会话列表，按 updated_at 倒序"""
        if self._backend == "redis":
            try:
                raw = self._redis.get("chat:sessions")
                if raw:
                    sessions = json.loads(raw)
                    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
                    return sessions
            except Exception as e:
                logger.error(f"[ChatStore] get_sessions redis error: {e}")
            return []

        elif self._backend == "sqlite":
            try:
                cur = self._sqlite_conn.execute(
                    "SELECT session_id, title, created_at, updated_at "
                    "FROM web_sessions ORDER BY updated_at DESC"
                )
                return [
                    {
                        "session_id": row[0],
                        "title": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                    }
                    for row in cur.fetchall()
                ]
            except Exception as e:
                logger.error(f"[ChatStore] get_sessions sqlite error: {e}")
                return []

        else:  # memory
            self._sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            return self._sessions

    def get_history(self, session_id: str) -> list:
        """返回某个会话的完整聊天记录"""
        if self._backend == "redis":
            try:
                raw = self._redis.get(f"chat:history:{session_id}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.error(f"[ChatStore] get_history redis error: {e}")
            return []

        elif self._backend == "sqlite":
            try:
                cur = self._sqlite_conn.execute(
                    "SELECT role, content, timestamp FROM web_messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
                return [
                    {"role": row[0], "content": row[1], "timestamp": row[2]}
                    for row in cur.fetchall()
                ]
            except Exception as e:
                logger.error(f"[ChatStore] get_history sqlite error: {e}")
                return []

        else:  # memory
            return self._history.get(session_id, [])

    def append_message(self, session_id: str, role: str, content: str):
        """追加一条消息到会话历史"""
        ts = datetime.now().isoformat()
        msg = {"role": role, "content": content, "timestamp": ts}

        if self._backend == "redis":
            try:
                key = f"chat:history:{session_id}"
                raw = self._redis.get(key)
                history = json.loads(raw) if raw else []
                history.append(msg)
                self._redis.set(key, json.dumps(history, ensure_ascii=False))
            except Exception as e:
                logger.error(f"[ChatStore] append_message redis error: {e}")

        elif self._backend == "sqlite":
            try:
                self._sqlite_conn.execute(
                    "INSERT INTO web_messages (session_id, role, content, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, ts),
                )
                self._sqlite_conn.commit()
            except Exception as e:
                logger.error(f"[ChatStore] append_message sqlite error: {e}")

        else:  # memory
            if session_id not in self._history:
                self._history[session_id] = []
            self._history[session_id].append(msg)

    def ensure_session(self, session_id: str, first_message: str):
        """
        确保会话存在于索引中。
        如果是新会话 → 提取标题并创建记录。
        如果是已有会话 → 更新 updated_at。
        """
        title = first_message[:30] + ("..." if len(first_message) > 30 else "")
        ts = datetime.now().isoformat()

        if self._backend == "redis":
            try:
                raw = self._redis.get("chat:sessions")
                sessions = json.loads(raw) if raw else []

                found = False
                for s in sessions:
                    if s["session_id"] == session_id:
                        s["updated_at"] = ts
                        found = True
                        break

                if not found:
                    sessions.append({
                        "session_id": session_id,
                        "title": title,
                        "created_at": ts,
                        "updated_at": ts,
                    })

                self._redis.set("chat:sessions", json.dumps(sessions, ensure_ascii=False))
            except Exception as e:
                logger.error(f"[ChatStore] ensure_session redis error: {e}")

        elif self._backend == "sqlite":
            try:
                cur = self._sqlite_conn.execute(
                    "SELECT session_id FROM web_sessions WHERE session_id = ?",
                    (session_id,),
                )
                if cur.fetchone():
                    self._sqlite_conn.execute(
                        "UPDATE web_sessions SET updated_at = ? WHERE session_id = ?",
                        (ts, session_id),
                    )
                else:
                    self._sqlite_conn.execute(
                        "INSERT INTO web_sessions (session_id, title, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        (session_id, title, ts, ts),
                    )
                self._sqlite_conn.commit()
            except Exception as e:
                logger.error(f"[ChatStore] ensure_session sqlite error: {e}")

        else:  # memory
            for s in self._sessions:
                if s["session_id"] == session_id:
                    s["updated_at"] = ts
                    return
            self._sessions.append({
                "session_id": session_id,
                "title": title,
                "created_at": ts,
                "updated_at": ts,
            })

    def delete_session(self, session_id: str):
        """删除会话及其所有消息"""
        if self._backend == "redis":
            try:
                # 删除历史消息
                self._redis.delete(f"chat:history:{session_id}")
                # 从索引中移除
                raw = self._redis.get("chat:sessions")
                if raw:
                    sessions = json.loads(raw)
                    sessions = [s for s in sessions if s["session_id"] != session_id]
                    self._redis.set("chat:sessions", json.dumps(sessions, ensure_ascii=False))
            except Exception as e:
                logger.error(f"[ChatStore] delete_session redis error: {e}")

        elif self._backend == "sqlite":
            try:
                self._sqlite_conn.execute(
                    "DELETE FROM web_messages WHERE session_id = ?", (session_id,)
                )
                self._sqlite_conn.execute(
                    "DELETE FROM web_sessions WHERE session_id = ?", (session_id,)
                )
                self._sqlite_conn.commit()
            except Exception as e:
                logger.error(f"[ChatStore] delete_session sqlite error: {e}")

        else:  # memory
            self._history.pop(session_id, None)
            self._sessions = [s for s in self._sessions if s["session_id"] != session_id]


# =============================================================================
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    interview_stage: Optional[str] = None
    session_id: str
    error: bool = False


# =============================================================================
# 工具函数
# =============================================================================

def extract_last_ai_message(messages: list) -> str:
    """从 LangGraph 消息列表中提取最后一条 AI 消息"""
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            content = msg.content if hasattr(msg, 'content') else ""
            if isinstance(content, list):
                content = content[0].get('text', '') if content else ''
            return str(content) if content else "(空回复)"
    return "(系统未生成回复)"


def build_status(state_values: dict, session_id: str) -> dict:
    """从 MultiAgentState 构建前端友好的状态 JSON"""
    resume = state_values.get("resume_data", {}) or {}
    job = state_values.get("selected_job", {}) or {}
    questions = state_values.get("interview_questions", []) or []
    answers = state_values.get("answers", []) or []
    report = state_values.get("final_report", "") or ""

    return {
        "session_id": session_id,
        "interview_stage": state_values.get("interview_stage", "idle"),
        "resume_parsed": bool(resume.get("name")),
        "candidate_name": resume.get("name", ""),
        "job_selected": bool(job.get("title")),
        "selected_job": {
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "match_score": job.get("match_score", 0),
        } if job.get("title") else None,
        "questions_total": len(questions),
        "questions_answered": len(answers),
        "current_question_idx": state_values.get("current_question_idx", 0),
        "scores": [a.get("score", 0) for a in answers],
        "report_ready": bool(report),
        "overall_score": state_values.get("overall_score", None),
    }


# =============================================================================
# 启动 & 关闭事件
# =============================================================================

@app.on_event("startup")
async def startup():
    global graph, checkpointer, chat_store

    logger.info("=" * 50)
    logger.info("AI 面试官 Web 服务 启动中...")
    logger.info("=" * 50)

    # 0. Monkey-patch _extract_thread_id：用 contextvars 传递正确的 thread_id
    #    需要同时 patch redis_client 模块 AND 所有顶层 import 了该函数的 Agent 模块
    import agents.redis_client as redis_mod
    import agents.resume_match_agent as rm_mod
    import agents.direct_reply_agent as dr_mod
    import contextvars
    _original_extract = redis_mod._extract_thread_id
    _current_session_id = contextvars.ContextVar("web_session_id", default="")

    def _patched_extract(state: dict) -> str:
        # 优先使用 contextvar（由 web_server 在调用前设置，跨 asyncio.to_thread 传递）
        tid = _current_session_id.get()
        if tid:
            return tid
        # 降级到原始逻辑
        return _original_extract(state)

    redis_mod._extract_thread_id = _patched_extract
    rm_mod._extract_thread_id = _patched_extract
    dr_mod._extract_thread_id = _patched_extract
    # 将 ContextVar 挂到 app 上供后续使用
    app.state.session_ctx = _current_session_id
    logger.info("[Startup] Patched _extract_thread_id in redis_client, resume_match, direct_reply")
    chat_store = ChatStore()

    # 2. 创建检查点（auto: Redis → SQLite → Memory）
    from agents.supervisor import create_checkpointer
    try:
        checkpointer = create_checkpointer("auto")
        logger.info(f"[Startup] Checkpointer: {type(checkpointer).__name__}")
    except Exception as e:
        logger.warning(f"[Startup] Auto checkpointer failed: {e}, falling back to Memory")
        try:
            checkpointer = create_checkpointer("memory")
        except Exception as e2:
            logger.error(f"[Startup] Memory checkpointer also failed: {e2}")
            raise

    # 3. 构建多 Agent 图
    try:
        from agents.supervisor import build_multi_agent_graph
        graph = build_multi_agent_graph(checkpointer=checkpointer)
        logger.info("[Startup] Multi-agent graph built successfully")
        logger.info("[Startup]   Supervisor → ResumeMatch | Interview | Evaluator | Report | DirectReply")
    except Exception as e:
        logger.error(f"[Startup] Graph build failed: {e}")
        raise

    logger.info("[Startup] ✅ Server ready at http://localhost:8000")


@app.on_event("shutdown")
async def shutdown():
    if chat_store and chat_store._sqlite_conn:
        try:
            chat_store._sqlite_conn.close()
        except Exception:
            pass
    logger.info("[Shutdown] Server stopped")


# =============================================================================
# API 端点
# =============================================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "graph_loaded": graph is not None,
        "chat_backend": chat_store._backend if chat_store else "none",
    }


@app.get("/")
async def root():
    """返回前端页面"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return FileResponse(str(html_path))


# ---- /api/chat ----
@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(req: ChatRequest):
    if not graph:
        return ChatResponse(
            reply="系统未就绪，请稍后刷新页面重试。",
            session_id=req.session_id,
            error=True,
        )

    msg = req.message.strip()
    if not msg:
        return ChatResponse(
            reply="请输入消息。",
            session_id=req.session_id,
            error=True,
        )

    # ---- 特殊命令拦截：不经过 Agent，直接处理 ----
    msg_lower = msg.lower()
    if msg_lower in ('reset', '重置'):
        # 直接调用 reset 逻辑
        try:
            await api_reset(req.session_id)
            return ChatResponse(
                reply="✅ 会话已重置。所有状态已清空，请上传简历或输入问题开始新对话。",
                interview_stage="idle",
                session_id=req.session_id,
                error=False,
            )
        except Exception as e:
            return ChatResponse(
                reply=f"重置失败：{e}",
                session_id=req.session_id,
                error=True,
            )

    if msg_lower in ('status', '状态'):
        try:
            status_data = await api_status(req.session_id)
            stage = status_data.get("interview_stage", "idle")
            lines = ["📊 **当前会话状态**", ""]
            lines.append(f"- 面试阶段：{stage}")
            lines.append(f"- 简历已解析：{'✅' if status_data.get('resume_parsed') else '❌'}")
            lines.append(f"- 岗位已选择：{'✅' if status_data.get('job_selected') else '❌'}")
            if status_data.get('selected_job'):
                j = status_data['selected_job']
                lines.append(f"- 已选岗位：{j.get('title', '?')} @ {j.get('company', '?')}")
            lines.append(f"- 面试题数：{status_data.get('questions_total', 0)}")
            lines.append(f"- 已回答：{status_data.get('questions_answered', 0)}")
            if status_data.get('scores'):
                lines.append(f"- 各题得分：{status_data['scores']}")
            lines.append(f"- 报告已生成：{'✅' if status_data.get('report_ready') else '❌'}")
            return ChatResponse(
                reply="\n".join(lines),
                interview_stage=stage,
                session_id=req.session_id,
                error=False,
            )
        except Exception as e:
            return ChatResponse(reply=f"获取状态失败：{e}", session_id=req.session_id, error=True)

    config = {"configurable": {"thread_id": req.session_id}}

    try:
        from langchain_core.messages import HumanMessage

        # 将 thread_id 注入 ContextVar，确保 _extract_thread_id() 能读取正确的 session
        app.state.session_ctx.set(req.session_id)

        result = await asyncio.to_thread(
            graph.invoke,
            {
                "messages": [HumanMessage(content=msg)],
            },
            config,
        )

        # 提取 AI 回复
        msgs = result.get("messages", [])
        reply = extract_last_ai_message(msgs)

        # 获取当前阶段
        try:
            current = await asyncio.to_thread(graph.get_state, config)
            stage = current.values.get("interview_stage", "idle") if current and current.values else "idle"
        except Exception:
            stage = "idle"

        # 持久化到 ChatStore
        if chat_store:
            await asyncio.to_thread(chat_store.append_message, req.session_id, "user", msg)
            await asyncio.to_thread(chat_store.append_message, req.session_id, "ai", reply)
            await asyncio.to_thread(chat_store.ensure_session, req.session_id, msg)

        return ChatResponse(
            reply=reply,
            interview_stage=stage,
            session_id=req.session_id,
            error=False,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[api/chat] Error: {error_msg}")

        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            reply = "抱歉，服务器响应超时，请稍后重试。"
        else:
            reply = f"处理请求时发生错误：{error_msg}"

        return ChatResponse(
            reply=reply,
            session_id=req.session_id,
            error=True,
        )


# ---- /api/chat/stream (SSE 流式输出) ----
@app.post("/api/chat/stream")
async def api_chat_stream(req: ChatRequest):
    """
    SSE 流式聊天端点。
    使用 LangGraph astream() 实现节点级流式输出，
    每个节点完成后推送进度事件，最终推送完整回复。
    """
    if not graph:
        return StreamingResponse(
            _sse_error("系统未就绪"),
            media_type="text/event-stream",
        )

    msg = req.message.strip()
    if not msg:
        return StreamingResponse(
            _sse_error("请输入消息"),
            media_type="text/event-stream",
        )

    msg_lower = msg.lower()

    # 特殊命令直接用非流式处理
    if msg_lower in ('reset', '重置', 'status', '状态'):
        result = await api_chat(req)
        return StreamingResponse(
            _sse_single_reply(result.reply, result.interview_stage or "idle"),
            media_type="text/event-stream",
        )

    config = {"configurable": {"thread_id": req.session_id}}

    async def event_stream():
        from langchain_core.messages import HumanMessage
        import queue as sync_queue
        import threading

        try:
            # 发送开始事件
            yield _sse_event("start", {"message": "正在思考..."})

            # 使用线程安全 Queue 桥接同步 graph.stream() 和异步 SSE
            # 每个 chunk 到达时立即发送，实现真正的流式输出
            chunk_queue = sync_queue.Queue()
            last_ai_content = ""
            producer_error = [None]  # list for mutable capture

            def producer():
                """在后台线程中运行同步 graph.stream()"""
                try:
                    for chunk in graph.stream(
                        {"messages": [HumanMessage(content=msg)]},
                        config,
                        stream_mode="values",
                    ):
                        chunk_queue.put(("chunk", chunk))
                    chunk_queue.put(("done", None))
                except Exception as e:
                    producer_error[0] = str(e)
                    chunk_queue.put(("error", None))

            app.state.session_ctx.set(req.session_id)
            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            # 消费 Queue，逐个发送 SSE 事件
            while True:
                try:
                    item = chunk_queue.get(timeout=0.1)
                except sync_queue.Empty:
                    # 没有新 chunk，让出控制权并继续等待
                    await asyncio.sleep(0.05)
                    continue

                kind = item[0]

                if kind == "error":
                    err = producer_error[0] or "未知错误"
                    logger.error(f"[api/chat/stream] Producer error: {err}")
                    yield _sse_event("error", {"message": err})
                    return

                if kind == "done":
                    break

                if kind == "chunk":
                    chunk = item[1]
                    msgs = chunk.get("messages", [])
                    for m in reversed(msgs):
                        if hasattr(m, 'type') and m.type == 'ai':
                            content = m.content if hasattr(m, 'content') else ""
                            if isinstance(content, list):
                                content = content[0].get('text', '') if content else ''
                            content = str(content)
                            if content and content != last_ai_content:
                                yield _sse_event("chunk", {
                                    "content": content,
                                    "stage": chunk.get("interview_stage", ""),
                                })
                                last_ai_content = content
                                await asyncio.sleep(0)  # flush to client
                            break

            # producer 结束 — 等待线程
            thread.join(timeout=5)
            reply = last_ai_content or "(系统未生成回复)"

            # 获取最终阶段
            try:
                current = await asyncio.to_thread(graph.get_state, config)
                stage = current.values.get("interview_stage", "idle") if current and current.values else "idle"
            except Exception:
                stage = "idle"

            # 持久化
            if chat_store:
                await asyncio.to_thread(chat_store.append_message, req.session_id, "user", msg)
                await asyncio.to_thread(chat_store.append_message, req.session_id, "ai", reply)
                await asyncio.to_thread(chat_store.ensure_session, req.session_id, msg)

            # 发送完成事件
            yield _sse_event("done", {
                "content": reply,
                "stage": stage,
                "session_id": req.session_id,
            })

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[api/chat/stream] Error: {error_msg}")
            if "timeout" in error_msg.lower():
                error_msg = "服务器响应超时，请稍后重试。"
            yield _sse_event("error", {"message": error_msg})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    """构造 SSE 事件字符串"""
    import json as _json
    return f"event: {event}\ndata: {_json.dumps(data, ensure_ascii=False)}\n\n"


async def _sse_error(message: str):
    """SSE 错误流"""
    yield _sse_event("error", {"message": message})


async def _sse_single_reply(content: str, stage: str):
    """SSE 单次回复（用于特殊命令）"""
    yield _sse_event("chunk", {"content": content, "stage": stage})
    yield _sse_event("done", {"content": content, "stage": stage})


# ---- /api/upload ----
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), session_id: str = Form(...)):
    # 类型检查
    filename = file.filename or "resume.pdf"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    # 读取内容大小
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="文件大小超过 10MB 限制")

    # 保存到 temp_uploads/{session_id}/
    upload_dir = Path("temp_uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 处理可能的文件名冲突：添加时间戳
    stem = Path(filename).stem
    safe_name = f"{stem}{ext}"
    dest_path = upload_dir / safe_name
    if dest_path.exists():
        safe_name = f"{stem}_{datetime.now().strftime('%H%M%S')}{ext}"
        dest_path = upload_dir / safe_name

    dest_path.write_bytes(content)

    # 返回绝对路径（统一使用正斜杠）
    abs_path = str(dest_path.resolve()).replace("\\", "/")

    logger.info(f"[api/upload] Saved: {abs_path} ({len(content)} bytes)")

    return {
        "success": True,
        "filename": safe_name,
        "file_path": abs_path,
        "message": "文件上传成功",
    }


# ---- /api/status/{session_id} ----
@app.get("/api/status/{session_id}")
async def api_status(session_id: str):
    if not graph:
        raise HTTPException(status_code=503, detail="系统未就绪")

    config = {"configurable": {"thread_id": session_id}}
    try:
        current = await asyncio.to_thread(graph.get_state, config)
        if current and current.values:
            return build_status(current.values, session_id)
        else:
            return build_status({}, session_id)
    except Exception as e:
        logger.error(f"[api/status] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- /api/reset/{session_id} ----
@app.post("/api/reset/{session_id}")
async def api_reset(session_id: str):
    if not graph:
        raise HTTPException(status_code=503, detail="系统未就绪")

    config = {"configurable": {"thread_id": session_id}}
    try:
        # 1. 清空 LangGraph 状态
        await asyncio.to_thread(
            graph.update_state,
            config,
            {
                "messages": [],
                "resume_data": {},
                "job_matches": [],
                "selected_job": {},
                "interview_questions": [],
                "current_question_idx": 0,
                "answers": [],
                "evaluations": [],
                "final_report": "",
                "overall_score": 0.0,
                "interview_stage": "idle",
                "next_agent": "resume_match",
                "_agent_history": [],
                "_total_steps": 0,
                "_error_count": {},
                "_last_error": "",
                "_asked_for_resume": False,
                "_agent_signal": "",
                "_direct_reply_count": 0,
                "waiting_for_user": False,
            },
        )

        # 2. 清除 Redis 中的用户信息和岗位缓存（防止 resume_match 读到旧数据）
        try:
            from agents.redis_client import get_user_info_store
            store = get_user_info_store()
            store.delete(session_id)  # 同时删除 user info + last_job（同一个 Redis Hash）
            logger.info(f"[api/reset] Redis user info & job cache cleared for {session_id}")
        except Exception as e:
            logger.warning(f"[api/reset] Redis cleanup skipped: {e}")

        # 3. 清除 Redis 中的聊天历史
        if chat_store:
            try:
                await asyncio.to_thread(chat_store.delete_session, session_id)
                logger.info(f"[api/reset] Chat history cleared for {session_id}")
            except Exception as e:
                logger.warning(f"[api/reset] Chat history cleanup skipped: {e}")

        return {"success": True, "message": "会话已重置"}
    except Exception as e:
        logger.error(f"[api/reset] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- /api/sessions ----
@app.get("/api/sessions")
async def api_sessions():
    if not chat_store:
        return {"sessions": []}
    try:
        sessions = await asyncio.to_thread(chat_store.get_sessions)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"[api/sessions] Error: {e}")
        return {"sessions": []}


# ---- /api/session/{session_id}/history ----
@app.get("/api/session/{session_id}/history")
async def api_session_history(session_id: str):
    if not chat_store:
        return {"session_id": session_id, "messages": []}
    try:
        messages = await asyncio.to_thread(chat_store.get_history, session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        logger.error(f"[api/session/{session_id}/history] Error: {e}")
        return {"session_id": session_id, "messages": []}


# ---- /api/session/{session_id}/delete ----
@app.delete("/api/session/{session_id}")
async def api_delete_session(session_id: str):
    if not chat_store:
        raise HTTPException(status_code=503, detail="系统未就绪")
    try:
        await asyncio.to_thread(chat_store.delete_session, session_id)
        return {"success": True, "message": "会话已删除"}
    except Exception as e:
        logger.error(f"[api/session/{session_id}/delete] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 程序入口（直接运行时的启动方式）
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n启动方式：")
    print("  uvicorn web_server:app --reload --host 0.0.0.0 --port 8000")
    print("\n或者直接运行：")
    print("  python -m uvicorn web_server:app --reload --host 0.0.0.0 --port 8000")
    print("\n访问：http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
