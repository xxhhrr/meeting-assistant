# agent/agent_builder.py
from langchain.agents import initialize_agent, AgentType
from tools import create_meeting_tool
from llm.local_llm import load_local_chat               # ← 同路径
from prompt_templates import SYSTEM_TEMPLATE

def get_meeting_agent():
    llm = load_local_chat()                             # ← 用新的 ChatLocal
    agent = initialize_agent(
        tools=[create_meeting_tool],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,               # 不变
        system_message=SYSTEM_TEMPLATE,
        verbose=True
    )
    return agent
