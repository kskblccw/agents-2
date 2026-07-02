import uuid
from dotenv import load_dotenv; load_dotenv()
from agents import build_multi_agent_graph
from agents.supervisor import create_checkpointer
from langchain_core.messages import HumanMessage

def test_flow():
    checkpointer = create_checkpointer()
    graph = build_multi_agent_graph(checkpointer=checkpointer)
    # 使用唯一 thread_id，避免旧会话状态污染
    config = {"configurable": {"thread_id": f"test_{uuid.uuid4().hex[:8]}"}}
    steps = ["E:\\A2\\简历\\text.docx", "帮我匹配岗位", "1"]
    for user_input in steps:
        print(f"\n=== {user_input} ===")
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        messages = result.get("messages", [])
        if messages and hasattr(messages[-1], 'content'):
            print(f"回复: {messages[-1].content[:200]}")
        print(f"selected_job={bool(result.get('selected_job'))}, job_matches={len(result.get('job_matches', []))}")
    print("\n测试完成，期望 selected_job=True")

if __name__ == "__main__":
    test_flow()
