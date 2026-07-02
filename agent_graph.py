#!/usr/bin/env python3
"""
基于 LangGraph 的完整面试 Agent 系统
实现了 LLM 驱动的流程控制
"""

import os
import json
from typing import TypedDict, Annotated, Sequence, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from agent_core import parse_resume_tool, evaluate_answer_tool, search_local_jobs, get_current_time, get_weather, search_knowledge_base

# ==================== 状态定义 ====================
class InterviewState(MessagesState):
    """面试状态定义"""
    resume_data: dict  # 简历数据
    job_description: str  # 职位描述
    interview_stage: str  # 面试阶段
    final_report: str  # 最终报告
    current_question: str  # 当前问题
    last_answer_score: float  # 上次回答的评分
    user_satisfaction: int  # 用户满意度，1-5


# ==================== System Prompt ====================
system_prompt = """
你是一个专业的 AI 面试官。

## 核心原则
1.根据用户的输入和当前状态，决定下一步做什么
2. 你控制整个面试流程，不是走固定脚本
3. 如果缺少必要信息，主动获取
4. 不要跳过必要的步骤，但也不要执行多余的步骤
5. 当需要外部信息时，调用工具
6. 每次只问一个问题，问完等待用户回答
7. 根据候选人的回答质量，灵活决定下一题

## 必要信息检查
- 面试前，必须有：简历数据 + 岗位信息
- 如果缺少简历数据：考虑调用 parse_resume_tool
- 如果缺少岗位信息：考虑调用 search_local_jobs
- 如果都有了 → 可以开始面试

## 工具说明
- parse_resume_tool：解析简历文件路径，提取结构化信息
- search_local_jobs：根据简历数据匹配岗位，参数使用 resume_data
**重要：你只能从工具返回的岗位列表中选择和推荐，绝对不能自己编造岗位。**如果工具返回的岗位列表为空，就告诉用户没有找到匹配的岗位。
- evaluate_answer_tool：评价候选人的回答，返回评分和改进建议。建议在用户回答后调用此工具，以获得客观的评分参考。
- get_current_time：获取当前时间，参数为空。
- get_weather：获取指定城市的天气信息，参数为城市名称。
- search_knowledge_base：搜索面试知识库，获取面试题、参考答案、评分标准、追问策略等专业知识。当你需要生成面试问题或评估回答时，可以先检索知识库获得参考。参数 query 为搜索关键词。

## 重要规则
- 当用户问关于“曹星桥”、“我”、“我的简历”、“我的生日”、“我的学校”等个人信息时，**必须**调用 search_knowledge_base 工具。
- 当你不确定答案时，优先调用 search_knowledge_base 检索知识库，而不是凭自己的知识回答。

## 常识
- 面试需要有明确的岗位
- 如果用户没有选择岗位，你不应该开始面试
- 通常需要用户告诉你是否需要匹配简历，如果用户告诉你需要匹配岗位，你就不能把岗位匹配了

## 面试流程（原则性指导，不是固定步骤）

### 开始面试
用户说"开始面试"后：
- 你直接根据简历生成第一个面试问题（不调用任何工具）
- 只问一个问题，结尾必须有问号

## 出题原则（必须遵守）
1. 只能问候选人简历里**明确写了**的内容
2. 如果简历里没有写具体细节，不要凭空追问
3. 例如：简历写了“FastAPI”，但没有写“为什么选FastAPI”，就不要问“FastAPI有什么优势”
4. 优先问简历里的数字、技术栈、项目成果

### 用户回答后
- 你可以选择调用 evaluate_answer_tool 来获取结构化评价
- 也可以自己判断回答质量，直接决定下一题
- 根据回答质量调整下一题：
  - 回答好（有细节、有案例）→ 问更有深度的问题
  - 回答一般 → 追问具体细节
  - 回答差 → 问更基础的问题或换个角度

### 结束面试
- 当你觉得问够了（通常 3-5 题），或者用户说"结束面试"时
- 生成一份总结报告，包含整体表现评价和建议

## 诚实原则
- 如果你不确定候选人的项目细节，不要假装知道
- 问通用问题，或者让候选人自己解释
- 不要编造候选人简历里没有的内容

## 禁止行为
- ❌ 一次性问多个问题
- ❌ 重复已经问过的问题
- ❌ 在用户回答前问下一个问题
- ❌ 使用固定的问题列表
- ❌ 编造工具返回结果中没有的岗位
- ❌ 推荐 jobs.json 里不存在的职位
- ❌ 用自己的知识补充岗位信息，必须严格使用工具返回的数据

## 重要
你是真实的面试官，不是判卷机器人。根据候选人的回答动态调整，而不是按脚本执行。
"""

# ==================== 工具配置 ====================
tools = [parse_resume_tool, search_local_jobs, evaluate_answer_tool, get_current_time, get_weather, search_knowledge_base]


# ==================== 自定义工具节点 ====================
def tools_node(state: InterviewState) -> InterviewState:
    """
    自定义工具节点，处理工具调用并更新状态
    """
    messages = list(state["messages"])
    last_message = messages[-1]
    
    # 获取工具调用
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return state
    
    # 处理每个工具调用
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        # 查找对应的工具
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool:
            try:
                # 调用工具
                result = tool.invoke(tool_args)
                print(f"🔧 工具 {tool_name} 返回原始结果: {result}") 
                
                # 处理不同工具的结果
                if tool_name == "parse_resume_tool":
                    try:
                        result_dict = result if isinstance(result, dict) else json.loads(result)
                        if isinstance(result_dict, dict) and (result_dict.get('_parsed') or result_dict.get('name')):
                            state["resume_data"] = result_dict
                            print(f"\n✅ 简历解析成功: {result_dict.get('name', '未知')}")
                    except:
                        pass
                        
                elif tool_name == "evaluate_answer_tool":
                    # 处理回答评分
                    try:
                        # 解析评分结果
                        score_result = result if isinstance(result, dict) else json.loads(result)
                        if isinstance(score_result, dict):
                            # 保存评分
                            score = score_result.get("score", 0)
                            state["last_answer_score"] = score  
                            # 计算满意度
                            if score < 5:
                                state["user_satisfaction"] = 3
                            elif score < 8:
                                state["user_satisfaction"] = 4
                            else:
                                state["user_satisfaction"] = 5
                            print(f"\n✅ 评分完成: {score_result.get('score', 0)}分，满意度:{state['user_satisfaction']}")
                    except Exception as e:
                        print(f"\n⚠️ 处理评分结果时出错：{e}")
                elif tool_name == "get_weather":
                    try:
                        weather_info = result if isinstance(result, str) else str(result)
                        print(f"\n✅ 天气查询成功: {weather_info}")
                    except Exception as e:
                        print(f"\n❌ 天气查询处理失败: {e}")

                elif tool_name == "search_knowledge_base":
                    # RAG 知识库检索
                    try:
                        result_str = result if isinstance(result, str) else str(result)
                        kb_len = len(result_str)
                        print(f"\n📚 知识库检索成功: 返回 {kb_len} 字符")
                    except Exception as e:
                        print(f"\n❌ 知识库检索处理失败: {e}")
                        
                # 将工具结果添加到消息列表
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=tool_call.get("id", "")
                    )
                )
                        
            except Exception as e:
                tool_messages.append(
                    ToolMessage(
                        content=f"工具调用失败: {str(e)}",
                        name=tool_name,
                        tool_call_id=tool_call.get("id", "")
                    )
                )
    
    # 返回更新后的状态
    return {
        "messages": tool_messages,
        "resume_data": state.get("resume_data", {}),
        "job_description": state.get("job_description", ""),
        "interview_stage": state.get("interview_stage", "idle"),
        "final_report": state.get("final_report", ""),
        "current_question": state.get("current_question", ""),
        "last_answer_score": state.get("last_answer_score", 0.0)
    }


# ==================== LLM 全局初始化 ====================
def init_llm():
    """初始化 LLM 模型"""
    from langchain_openai import ChatOpenAI
    
    # 从环境变量获取 API Key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        except ImportError:
            pass
    
    if not api_key:
        raise ValueError("环境变量未设置")
    
    # 初始化 DeepSeek LLM 模型，设置超时时间
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.7,
        api_key=api_key,
        streaming=False,
        request_timeout=60,  # 设置 60 秒超时
        base_url="https://api.deepseek.com/v1"
    )
    
    return llm


# 全局初始化 LLM 和绑定工具（只执行一次）
llm = init_llm()
llm_with_tools = llm.bind_tools(tools)

# ==================== 面试官节点 ====================
def interviewer_node(state: InterviewState):
    """
    面试官核心节点：接收状态，决定下一步动作
    """
    messages = list(state["messages"])
    
    # 检查是否需要插入系统提示词（只在第一次或消息列表为空时插入）
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(content=system_prompt)
        ] + messages
    else:
        # 如果已有系统提示词，检查是否需要添加状态信息
        # 只在有 resume_data 时才添加额外的状态信息
        if state.get("resume_data") and len(messages) == 1:
            resume_data = state["resume_data"]
            state_info = f"\n\n## 当前状态\n简历已解析：{resume_data.get('name', '未知')}\n"
            state_info += f"技能：{', '.join(resume_data.get('skills', [])[:5])}\n\n"
            state_info += "**当用户询问岗位匹配时，必须调用 search_local_jobs 工具！**\n"
            messages.insert(1, SystemMessage(content=state_info))
    
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    
    # 可视化调试：打印工具调用信息
    if hasattr(response, "tool_calls") and response.tool_calls:
        print("\n" + "="*60)
        print("[DEBUG] 🤖 LLM 决定调用工具：")
        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "未知工具")
            tool_args = tool_call.get("args", {})
            print(f"- 工具名称: {tool_name}")
            print(f"- 调用参数: {tool_args}")
        print("="*60 + "\n")
    
    # 返回更新后的状态（保留其他字段）
    return {
        "messages": [response],
        "resume_data": state.get("resume_data", {}),
        "job_description": state.get("job_description", ""),
        "interview_stage": state.get("interview_stage", "idle"),
        "final_report": state.get("final_report", ""),
        "current_question": state.get("current_question", ""),
        "last_answer_score": state.get("last_answer_score", 0.0)
    }


# ==================== 路由逻辑 ====================
def should_continue(state: InterviewState):
    """
    决定下一步是继续调用工具还是结束
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果最后一条消息包含工具调用，继续处理
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return END


# ==================== 构建 Graph ====================
def build_graph(checkpointer=None):
    """构建完整的 LangGraph"""
    # 创建图
    graph = StateGraph(InterviewState)
    
    # 添加节点
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("tools", tools_node)  # 使用自定义工具节点
    
    # 添加边
    graph.add_edge(START, "interviewer")
    graph.add_conditional_edges(
        "interviewer",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    graph.add_edge("tools", "interviewer")
    
    # 编译图，传入 checkpointer
    return graph.compile(checkpointer=checkpointer)


# ==================== 全局 Graph 实例 ====================
checkpointer = MemorySaver()
graph = build_graph(checkpointer=checkpointer)


# ==================== 简单接口函数 ====================
def process_user_input(user_input: str, thread_id: str = "default_session"):
    """
    处理用户输入，返回响应和更新后的状态
    """
    # 配置 thread_id
    config = {"configurable": {"thread_id": thread_id}}
    
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 准备输入状态
    input_state = {"messages": [user_message]}
    
    # 调用 Graph，添加超时保护和错误处理
    try:
        result = graph.invoke(input_state, config=config)
    except Exception as e:
        # 如果超时，返回友好错误信息
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return "抱歉，服务器响应超时，请稍后重试。", {}
        else:
            return f"处理请求时发生错误: {error_msg}", {}
    
    # 提取最后一条消息作为响应
    last_message = result["messages"][-1]
    response_text = last_message.content
    
    # 获取最终状态（可选，用于调试）
    try:
        final_state = graph.get_state(config)
        return response_text, final_state.values
    except:
        # 如果获取状态失败，仍然返回响应
        return response_text, {}


# ==================== 辅助函数 ====================
def get_current_state(thread_id: str = "default_session"):
    """
    获取当前会话的状态（用于调试）
    
    参数：
        thread_id: 会话 ID
    
    返回：
        当前状态字典
    """
    config = {"configurable": {"thread_id": thread_id}}
    current_state = graph.get_state(config)
    return current_state.values


def reset_session(thread_id: str = "default_session"):
    """
    重置指定会话的状态
    
    参数：
        thread_id: 会话 ID
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {
        "messages": [],
        "resume_data": {},
        "job_description": "",
        "interview_stage": "idle",
        "final_report": "",
        "current_question": ""
    })