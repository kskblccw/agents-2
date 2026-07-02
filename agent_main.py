#!/usr/bin/env python3
import sys
import agents.resume_match_agent
print("加载的文件路径:", agents.resume_match_agent.__file__)
print("Python 路径:", sys.executable)
print("模块搜索路径:", sys.path)
"""
AI 面试官主程序 — 多 Agent 版
=============================
基于 LangGraph Supervisor 模式的多 Agent 面试系统。

架构：
    Supervisor (路由器)
       ├── ResumeMatch Agent  (简历解析 + 岗位匹配)
       ├── Interview Agent    (面试主持 + 出题)
       ├── Evaluator Agent    (回答评分)
       └── Report Agent       (报告生成)

使用：
    python agent_main.py
"""

import sys
import os

# ==================== 加载环境变量 ====================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ 未设置 DEEPSEEK_API_KEY 环境变量，请在 .env 文件中配置")
    sys.exit(1)

# ==================== 导入多 Agent 系统 ====================
try:
    from agents.supervisor import build_multi_agent_graph, create_checkpointer
    from agents.state import MultiAgentState
    from langchain_core.messages import HumanMessage
    print("✅ 多 Agent 系统导入成功")
except ImportError as e:
    print(f"❌ 多 Agent 系统导入失败: {e}")
    print("请确保 agents/ 目录完整")
    sys.exit(1)


# ==================== 欢迎信息 ====================
def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("🤖 AI 面试官 - 多 Agent 智能面试系统")
    print("=" * 60)
    print(f"  架构：Supervisor + 4 个专业 Agent")
    print(f"  检查点：Redis（持久化会话）")
    print()
    print("欢迎使用！您可以：")
    print("    提供简历文件路径进行解析")
    print("    询问岗位匹配（如：'帮我匹配岗位'）")
    print("    选择职位并开始面试（如：回复数字 1-3 选择岗位）")
    print("    输入 'exit' 或 'quit' 退出程序")
    print("    输入 'status' 查看当前状态")
    print("    输入 'reset' 重置会话")
    print("\n" + "-" * 60 + "\n")


# ==================== 状态查询 ====================
def print_status(state: dict):
    """打印当前会话状态"""
    resume = state.get("resume_data", {})
    job = state.get("selected_job", {})
    questions = state.get("interview_questions", [])
    answers = state.get("answers", [])
    report = state.get("final_report", "")

    print("\n" + "=" * 40)
    print("📊 当前会话状态")
    print("=" * 40)
    print(f"  面试阶段：{state.get('interview_stage', 'idle')}")
    print(f"  简历已解析：{'✅' if resume and resume.get('name') else '❌'}")
    print(f"  岗位已选择：{'✅' if job and job.get('title') else '❌'}")
    print(f"  面试题数：{len(questions)}")
    print(f"  已回答：{len(answers)}/{len(questions)}")
    if answers:
        scores = [a.get('score', '?') for a in answers]
        print(f"  各题得分：{scores}")
    print(f"  报告已生成：{'✅' if report else '❌'}")
    print(f"  总步数：{state.get('_total_steps', 0)}")
    last_error = state.get('_last_error', '')
    if last_error:
        print(f"  最近错误：{last_error[:100]}")
    print("=" * 40 + "\n")


# ==================== 主事件循环 ====================
def main_loop():
    """主事件循环"""
    print_welcome()

    thread_id = "interview_session"

    # ---- 创建检查点（优先 Redis，降级 SQLite） ----
    print("正在初始化检查点...")
    try:
        checkpointer = create_checkpointer("auto")
        cp_type = type(checkpointer).__name__
        print(f"✅ 检查点就绪：{cp_type}")
    except Exception as e:
        print(f"⚠️ 检查点创建失败: {e}，使用内存模式")
        checkpointer = create_checkpointer("memory")

    # ---- 构建多 Agent 图 ----
    print("正在构建多 Agent 图...")
    try:
        graph = build_multi_agent_graph(checkpointer=checkpointer)
        print("✅ 多 Agent 图构建完成")
        print(f"   Supervisor → ResumeMatch | Interview | Evaluator | Report")
    except Exception as e:
        print(f"❌ 图构建失败: {e}")
        sys.exit(1)

    config = {"configurable": {"thread_id": thread_id}}
    print("\n系统就绪！开始对话吧 🚀\n")

    while True:
        try:
            # ---- 获取用户输入 ----
            user_input = input("您: ").strip()

            if not user_input:
                continue

            # ---- 特殊命令 ----
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("\n感谢使用 AI 面试官！再见！👋")
                break

            if user_input.lower() == 'status':
                try:
                    current = graph.get_state(config)
                    if current and current.values:
                        print_status(current.values)
                    else:
                        print("暂无会话状态。")
                except Exception as e:
                    print(f"获取状态失败: {e}")
                continue

            if user_input.lower() == 'reset':
                try:
                    # 重置：用空状态覆盖
                    graph.update_state(config, {
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
                    })
                    # 同时清除 Redis 中的用户信息和岗位缓存
                    try:
                        import redis as _redis
                        _r = _redis.Redis.from_url(
                            os.getenv("REDIS_URL", "redis://localhost:6379"),
                            socket_connect_timeout=3, socket_timeout=3,
                            decode_responses=True,
                        )
                        _r.ping()
                        _r.delete(f"user:info:{thread_id}")
                        print(f"  [RESET] Redis key 'user:info:{thread_id}' deleted")
                        _r.close()
                    except Exception as _e:
                        print(f"  [RESET] Redis cleanup skipped: {_e}")
                    print("✅ 会话已重置。")
                except Exception as e:
                    print(f"重置失败: {e}")
                continue

            # ---- 处理用户输入 ----
            print("⏳", end="", flush=True)

            max_retries = 2
            response_text = None
            last_error = None

            for attempt in range(max_retries):
                try:
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config,
                    )

                    # 提取最后一条 AI 消息
                    msgs = result.get("messages", [])
                    for msg in reversed(msgs):
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            response_text = msg.content
                            break

                    if response_text is None:
                        response_text = "(系统未生成回复，请输入 'status' 查看状态)"

                    break  # 成功

                except Exception as e:
                    last_error = str(e)
                    if "timeout" in last_error.lower() or "timed out" in last_error.lower():
                        print(f"\r⚠️ 请求超时，正在重试 ({attempt + 1}/{max_retries})...")
                        continue
                    else:
                        raise

            if response_text is None:
                print(f"\r❌ 请求失败: {last_error}")
                print("请稍后重试...")
                continue

            # ---- 输出 AI 响应 ----
            print("\r" + "-" * 60)
            print("🤖 AI 面试官:")
            print(response_text)
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！👋")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("\n请重新输入...")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        print("\n请检查依赖是否正确安装！")
        sys.exit(1)
