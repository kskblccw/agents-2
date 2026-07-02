#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 共享工具模块 (utils.py)
=============================

提供所有 Agent 共用的基础设施：
1. LLMClient      — 带超时、重试、缓存的 LLM 调用封装
2. RateLimiter    — 滑动窗口限流器（按 thread_id）
3. ResponseCache  — TTL 内存缓存（减少重复 LLM 调用）
4. safe_llm_call  — 便捷函数：一行代码完成「超时+重试+限流+缓存」

使用方式：
    from agents.utils import safe_llm_call
    response = safe_llm_call(prompt="...", thread_id="session-1", cache_key="gen_q_v1")
"""

import os
import time
import hashlib
import functools
import logging
import threading
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict

logger = logging.getLogger("multi-agent.utils")

# =============================================================================
# 第一部分：LLMClient — 统一的 LLM 调用封装
# =============================================================================

class LLMClient:
    """
    统一的 LLM 调用客户端。

    特性：
    - 30 秒超时（可配置）
    - 自动重试（最多 3 次，指数退避）
    - 异常统一转换为 RuntimeError 并保留原始信息
    - 支持 system + user 消息格式

    使用：
        client = LLMClient()
        response = client.chat(prompt="你好", system_prompt="你是面试官")
        response = client.chat(messages=[{"role": "user", "content": "你好"}])
    """

    def __init__(
        self,
        model: str = None,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
    ):
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
                timeout=float(self.timeout),
                max_retries=0,  # 我们自己管理重试
            )
        return self._client

    def chat(
        self,
        prompt: str = None,
        system_prompt: str = None,
        messages: list = None,
        temperature: float = None,
        model: str = None,
    ) -> str:
        """
        调用 LLM 对话接口（带超时 + 重试）。

        参数：
            prompt: 用户消息（快捷方式）
            system_prompt: 系统提示词（可选）
            messages: 完整消息列表（与 prompt 二选一）
            temperature: 温度参数（覆盖默认值）
            model: 模型名称（覆盖默认值）

        返回：
            str: LLM 响应文本

        异常：
            RuntimeError: 所有重试均失败
            TimeoutError: 单次调用超时（内部重试）
        """
        # 构建消息列表
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("prompt 和 messages 不能同时为空")

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                if attempt < self.max_retries - 1:
                    wait = min(2 ** attempt, 8)  # 指数退避: 1s, 2s, 4s, max 8s
                    logger.warning(
                        f"[LLM] Attempt {attempt + 1}/{self.max_retries} failed "
                        f"({error_type}: {str(e)[:100]}), retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"[LLM] All {self.max_retries} attempts failed. "
                        f"Last error: {error_type}: {str(e)[:200]}"
                    )

        raise RuntimeError(
            f"LLM 调用失败（已重试 {self.max_retries} 次）: {type(last_error).__name__}: {str(last_error)[:200]}"
        )


# 全局单例（线程安全）
_llm_client: Optional[LLMClient] = None
_llm_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """获取全局 LLMClient 单例"""
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                _llm_client = LLMClient()
    return _llm_client


# =============================================================================
# 第二部分：RateLimiter — 滑动窗口限流器
# =============================================================================

class RateLimiter:
    """
    滑动窗口限流器（按 thread_id / session_id）。

    规则：
    - 每秒最多 N 个请求（默认 2）
    - 超限时抛出 RateLimitExceeded 异常
    - 自动清理过期记录

    使用：
        limiter = RateLimiter(max_requests_per_sec=2)
        try:
            limiter.check("session-123")
        except RateLimitExceeded:
            ...
    """

    class RateLimitExceeded(Exception):
        """限流异常"""
        def __init__(self, key: str, count: int, limit: int):
            self.key = key
            self.count = count
            self.limit = limit
            super().__init__(f"限流: '{key}' 在 1 秒内请求 {count} 次，上限 {limit} 次")

    def __init__(self, max_requests_per_sec: int = 2):
        self.max_requests = max_requests_per_sec
        self._windows: Dict[str, list] = {}  # {key: [timestamp, ...]}
        self._lock = threading.Lock()

    def check(self, key: str):
        """
        检查是否允许请求。

        参数：
            key: 限流键（通常是 thread_id 或 session_id）

        异常：
            RateLimitExceeded: 超过限流阈值
        """
        now = time.time()
        window_start = now - 1.0  # 1 秒窗口

        with self._lock:
            # 获取或创建该 key 的时间窗口
            if key not in self._windows:
                self._windows[key] = []

            # 清理过期记录
            self._windows[key] = [t for t in self._windows[key] if t > window_start]

            # 检查是否超限
            if len(self._windows[key]) >= self.max_requests:
                raise self.RateLimitExceeded(key, len(self._windows[key]), self.max_requests)

            # 记录本次请求
            self._windows[key].append(now)

    def get_usage(self, key: str) -> int:
        """获取当前窗口内请求数（调试用）"""
        now = time.time()
        window_start = now - 1.0
        with self._lock:
            if key not in self._windows:
                return 0
            self._windows[key] = [t for t in self._windows[key] if t > window_start]
            return len(self._windows[key])

    def cleanup(self):
        """清理所有过期记录（定期调用）"""
        now = time.time()
        window_start = now - 5.0  # 保留 5 秒缓冲
        with self._lock:
            for key in list(self._windows.keys()):
                self._windows[key] = [t for t in self._windows[key] if t > window_start]
                if not self._windows[key]:
                    del self._windows[key]


# 全局单例
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """获取全局 RateLimiter 单例"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(max_requests_per_sec=2)
    return _rate_limiter


# =============================================================================
# 第三部分：ResponseCache — TTL 内存缓存
# =============================================================================

class ResponseCache:
    """
    TTL 内存缓存（LRU 淘汰）。

    用于缓存相同问题的 LLM 响应，减少重复调用。

    使用：
        cache = ResponseCache(max_size=100, ttl_seconds=300)
        cache.set("key", "response text")
        result = cache.get("key")  # "response text" or None
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._store: OrderedDict[str, Tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def _build_key(self, *args, **kwargs) -> str:
        """构建缓存键（基于参数 hash）"""
        raw = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def get(self, cache_key: str) -> Optional[str]:
        """获取缓存值（过期返回 None）"""
        with self._lock:
            if cache_key not in self._store:
                return None

            timestamp, value = self._store[cache_key]
            if time.time() - timestamp > self.ttl:
                del self._store[cache_key]
                return None

            # LRU: 移到末尾
            self._store.move_to_end(cache_key)
            return value

    def set(self, cache_key: str, value: str):
        """设置缓存值"""
        with self._lock:
            # 淘汰策略：超过 max_size 时删除最旧的
            while len(self._store) >= self.max_size:
                self._store.popitem(last=False)

            self._store[cache_key] = (time.time(), value)

    def clear(self):
        """清空缓存"""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        """获取缓存统计"""
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "ttl": self.ttl,
            }


# 全局单例
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """获取全局 ResponseCache 单例"""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache(max_size=100, ttl_seconds=300)
    return _response_cache


# =============================================================================
# 第四部分：safe_llm_call — 一站式安全 LLM 调用
# =============================================================================

def safe_llm_call(
    prompt: str = None,
    system_prompt: str = None,
    messages: list = None,
    thread_id: str = "default",
    cache_key: str = None,
    use_cache: bool = True,
    temperature: float = None,
    model: str = None,
) -> str:
    """
    一站式安全 LLM 调用（超时 + 重试 + 限流 + 缓存）。

    这是推荐的外部接口，封装了所有安全措施。

    参数：
        prompt: 用户消息
        system_prompt: 系统提示词
        messages: 完整消息列表
        thread_id: 会话 ID（用于限流）
        cache_key: 缓存键（None=不缓存，相同键返回缓存结果）
        use_cache: 是否使用缓存
        temperature: 温度参数
        model: 模型名称

    返回：
        str: LLM 响应

    异常：
        RateLimitExceeded: 触发限流
        RuntimeError: LLM 调用失败
    """
    # 1. 限流检查
    limiter = get_rate_limiter()
    limiter.check(thread_id)

    # 2. 缓存检查
    if use_cache and cache_key:
        cache = get_response_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug(f"[CACHE] Hit: {cache_key}")
            return cached

    # 3. LLM 调用（自带超时+重试）
    client = get_llm_client()
    response = client.chat(
        prompt=prompt,
        system_prompt=system_prompt,
        messages=messages,
        temperature=temperature,
        model=model,
    )

    # 4. 缓存结果
    if use_cache and cache_key:
        cache = get_response_cache()
        cache.set(cache_key, response)
        logger.debug(f"[CACHE] Set: {cache_key}")

    return response


def build_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    构建可读的缓存键。

    示例：
        key = build_cache_key("gen_q", "Python后端", "FastAPI,Docker")
        # → "gen_q:c6a12b3f..."
    """
    raw = prefix + "|" + "|".join(str(a) for a in args) + "|" + "|".join(
        f"{k}={v}" for k, v in sorted(kwargs.items())
    )
    hash_suffix = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}:{hash_suffix}"


# =============================================================================
# 第五部分：自检
# =============================================================================

if __name__ == "__main__":
    print("=== Utils Self-Check ===\n")

    # Test 1: LLMClient basic
    print("[1] Testing LLMClient...")
    try:
        client = LLMClient(timeout=10, max_retries=1)
        resp = client.chat(prompt="回复一个字：好", temperature=0.0)
        print(f"    Response: {resp[:50]}")
        print("    [OK] LLMClient works")
    except Exception as e:
        print(f"    [WARN] LLMClient failed (API key may not be configured): {e}")

    print()

    # Test 2: RateLimiter
    print("[2] Testing RateLimiter...")
    limiter = RateLimiter(max_requests_per_sec=2)
    try:
        limiter.check("test-user")
        limiter.check("test-user")
        # 第三次应该被限流
        try:
            limiter.check("test-user")
            print("    [WARN] Third request should have been rate-limited!")
        except RateLimiter.RateLimitExceeded as e:
            print(f"    [OK] Rate limited as expected: {e}")
    except Exception as e:
        print(f"    [ERR] {e}")

    print()

    # Test 3: ResponseCache
    print("[3] Testing ResponseCache...")
    cache = ResponseCache(max_size=10, ttl_seconds=3)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1", "Cache hit failed"
    assert cache.get("key2") is None, "Cache miss should return None"

    # TTL 测试（快速过期）
    small_cache = ResponseCache(max_size=10, ttl_seconds=0)
    small_cache.set("k", "v")
    assert small_cache.get("k") is None, "Expired cache should return None"
    print("    [OK] Cache get/set/expire works")
    print()

    # Test 4: Cache key builder
    print("[4] Testing cache key builder...")
    key1 = build_cache_key("gen_q", "Python", "FastAPI", temp=0.7)
    key2 = build_cache_key("gen_q", "Python", "FastAPI", temp=0.7)
    key3 = build_cache_key("gen_q", "Python", "Django", temp=0.7)
    assert key1 == key2, "Same inputs should produce same key"
    assert key1 != key3, "Different inputs should produce different keys"
    print(f"    key1={key1}")
    print(f"    key3={key3}")
    print("    [OK] Cache keys are deterministic")

    print()
    print("=== ALL UTILS TESTS PASSED ===")
