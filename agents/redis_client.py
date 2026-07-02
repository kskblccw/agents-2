#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 用户信息持久化存储 (redis_client.py)
=========================================

职责：
1. 简历解析成功后，将个人信息写入 Redis Hash
2. 用户查询个人信息时，从 Redis 读取并返回
3. Redis 不可用时，静默降级（不抛异常、不阻塞主流程）

数据结构：
    Key:   user:info:{thread_id}
    Type:  Hash
    Fields: name, birthday, phone, email, school, skills, summary
    TTL:   7 天（每次更新时刷新）

使用方式：
    from agents.redis_client import get_user_info_store

    store = get_user_info_store()
    store.store("session_1", {"name": "张三", "phone": "138...", ...})
    info = store.get("session_1")           # {"name": "张三", ...}
    info = store.get("session_1", "name")   # {"name": "张三"}
    exists = store.exists("session_1")      # True / False
"""

import os
import logging
from typing import Optional, Dict

logger = logging.getLogger("multi-agent.redis_client")

# =============================================================================
# UserInfoStore — Redis Hash 封装
# =============================================================================


class UserInfoStore:
    """
    用户信息 Redis 存储。

    特性：
    - 延迟初始化（首次调用时连接，不阻塞导入）
    - 连接失败不抛异常（标记不可用，静默降级）
    - 3 秒连接 + 读取超时
    - 每次写入自动刷新 TTL
    """

    DEFAULT_TTL = 604800  # 7 天

    def __init__(self, redis_url: str = None, ttl: int = None):
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL", "redis://localhost:6379"
        )
        self.ttl = ttl or int(os.getenv("USER_INFO_TTL", self.DEFAULT_TTL))
        self._client = None  # None=未初始化, False=不可用, Redis=可用

    # ---- 延迟初始化 ----

    @property
    def client(self):
        """获取 Redis 客户端（延迟连接，失败不抛异常）"""
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis.from_url(
                    self.redis_url,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                    decode_responses=True,
                )
                self._client.ping()
                logger.info(f"[Redis] Connected: {self.redis_url}")
            except Exception as e:
                logger.warning(f"[Redis] Unavailable ({e}) — user info will use fallback")
                self._client = False
        return self._client if self._client is not False else None

    def _key(self, thread_id: str) -> str:
        """构建 Redis Key"""
        return f"user:info:{thread_id}"

    # ---- 写入 ----

    def store(self, thread_id: str, user_info: dict):
        """
        将用户信息写入 Redis Hash。

        参数：
            thread_id: 会话 ID
            user_info: 用户信息字典，只存储非空字符串值

        异常：不抛异常，写入失败静默记录日志
        """
        cli = self.client
        if cli is None:
            return

        # 过滤空值
        data = {k: str(v) for k, v in user_info.items() if v}
        if not data:
            return

        try:
            key = self._key(thread_id)
            # 先删旧数据再写入（确保一致性）
            cli.delete(key)
            cli.hset(key, mapping=data)
            cli.expire(key, self.ttl)
            logger.info(
                f"[Redis] Stored user info for '{thread_id}': "
                f"fields={list(data.keys())}, ttl={self.ttl}s"
            )
        except Exception as e:
            logger.warning(f"[Redis] Store failed for '{thread_id}': {e}")

    # ---- 读取 ----

    def get(self, thread_id: str, field: str = None) -> dict:
        """
        从 Redis 读取用户信息。

        参数：
            thread_id: 会话 ID
            field: 可选，指定字段名（如 'name', 'birthday'）。
                   为 None 时返回全部字段。

        返回：
            dict: 字段名→值的映射。Redis 不可用或无数据时返回空 dict。
        """
        cli = self.client
        if cli is None:
            return {}

        try:
            key = self._key(thread_id)
            if field:
                value = cli.hget(key, field)
                return {field: value} if value else {}
            else:
                return cli.hgetall(key) or {}
        except Exception as e:
            logger.warning(f"[Redis] Get failed for '{thread_id}': {e}")
            return {}

    # ---- 存在性检查 ----

    def exists(self, thread_id: str) -> bool:
        """
        检查 Redis 中是否存在该用户的信息。

        参数：
            thread_id: 会话 ID

        返回：
            bool: Redis 可用且 Key 存在 → True，否则 → False
        """
        cli = self.client
        if cli is None:
            return False

        try:
            return bool(cli.exists(self._key(thread_id)))
        except Exception as e:
            logger.warning(f"[Redis] Exists check failed for '{thread_id}': {e}")
            return False

    # ---- 岗位缓存 ----

    def store_job(self, thread_id: str, job_info: dict):
        """
        将用户最后选择的岗位信息写入 Redis（JSON 编码）。

        参数：
            thread_id: 会话 ID
            job_info: 岗位字典，包含 title, company, match_score 等

        异常：不抛异常，写入失败静默降级
        """
        cli = self.client
        if cli is None:
            return

        try:
            import json
            key = self._key(thread_id)
            cli.hset(key, "last_job", json.dumps(job_info, ensure_ascii=False))
            cli.expire(key, self.ttl)
            logger.info(
                f"[Redis] Stored last_job for '{thread_id}': "
                f"{job_info.get('title', '?')} @ {job_info.get('company', '?')}"
            )
        except Exception as e:
            logger.warning(f"[Redis] Store job failed for '{thread_id}': {e}")

    def get_job(self, thread_id: str) -> dict:
        """
        从 Redis 读取用户最后选择的岗位。

        参数：
            thread_id: 会话 ID

        返回：
            dict: 岗位字典，不存在或解析失败返回空 dict
        """
        cli = self.client
        if cli is None:
            return {}

        try:
            import json
            key = self._key(thread_id)
            raw = cli.hget(key, "last_job")
            if raw:
                return json.loads(raw)
            return {}
        except Exception as e:
            logger.warning(f"[Redis] Get job failed for '{thread_id}': {e}")
            return {}

    # ---- 删除 ----

    def delete(self, thread_id: str):
        """
        删除 Redis 中的用户信息（重新上传简历时调用）。

        参数：
            thread_id: 会话 ID
        """
        cli = self.client
        if cli is None:
            return

        try:
            cli.delete(self._key(thread_id))
            logger.info(f"[Redis] Deleted user info for '{thread_id}'")
        except Exception as e:
            logger.warning(f"[Redis] Delete failed for '{thread_id}': {e}")


# =============================================================================
# 全局单例
# =============================================================================

_user_info_store: Optional[UserInfoStore] = None


def get_user_info_store() -> UserInfoStore:
    """获取 UserInfoStore 全局单例"""
    global _user_info_store
    if _user_info_store is None:
        _user_info_store = UserInfoStore()
    return _user_info_store


# =============================================================================
# 辅助函数
# =============================================================================


def _extract_thread_id(state: dict) -> str:
    """
    从 LangGraph state 中提取 thread_id。

    优先级：
    1. state 中的 config 信息（LangGraph 运行时可能注入）
    2. 降级为 "interview_session"（与 agent_main.py 的硬编码一致）
    """
    # 尝试从 state 中获取 config（LangGraph 运行时注入）
    config = state.get("configurable", {}) or state.get("config", {})
    if isinstance(config, dict):
        tid = config.get("thread_id", "")
        if tid:
            return tid

    # 降级：使用 agent_main.py 中硬编码的默认 thread_id
    return "interview_session"


# =============================================================================
# 自检
# =============================================================================

if __name__ == "__main__":
    print("=== Redis UserInfoStore Self-Check ===\n")

    store = UserInfoStore()

    # Test 1: 连接测试
    print("[1] Testing Redis connection...")
    if store.client:
        print("    [OK] Redis connected")
    else:
        print("    [WARN] Redis not available (check docker start redis-stack)")

    # Test 2: 读写测试
    if store.client:
        print("[2] Testing store/get/delete...")
        test_id = "test_user_check"
        test_data = {
            "name": "测试用户",
            "phone": "13800000000",
            "school": "测试大学",
        }

        # Store
        store.store(test_id, test_data)
        print(f"    Store OK: {test_data}")

        # Get all
        result = store.get(test_id)
        assert result.get("name") == "测试用户", f"Expected name, got {result}"
        print(f"    Get all OK: {result}")

        # Get single field
        result = store.get(test_id, "phone")
        assert result.get("phone") == "13800000000"
        print(f"    Get field OK: {result}")

        # Exists
        assert store.exists(test_id) is True
        print(f"    Exists OK: True")

        # Delete
        store.delete(test_id)
        assert store.exists(test_id) is False
        print(f"    Delete OK: exists={store.exists(test_id)}")

        # Clean up
        store.delete(test_id)

    # Test 3: 不可用降级测试
    print("[3] Testing fallback when unavailable...")
    bad_store = UserInfoStore(redis_url="redis://no-host:9999", ttl=10)
    assert bad_store.client is None, "Should be unavailable"
    assert bad_store.get("any") == {}, "Should return empty dict"
    assert bad_store.exists("any") is False, "Should return False"
    # store/delete 不应抛异常
    bad_store.store("any", {"name": "test"})
    bad_store.delete("any")
    print("    [OK] All fallback methods return safe defaults without exceptions")

    print("\n=== ALL REDIS TESTS PASSED ===")
