# 파일시스템 에이전트
#  - mcp가반 파일도구를 사용하는 에이전트
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

from config import settings
from utils.filesystem import filesystem_tools

_llm = ChatOpenAI(
    model = settings.llm_model,
    temperature=0
)

_agent = create_agent(
    model= _llm,
    tools= filesystem_tools,
    system_prompt= """
        당신은 로컬 downloads/ 폴더를 관리하는 파일시스템 에이전트입니다.
        요청한 작업내용에 맞춰 적절한 도구를 사용해 주세요.
    """
)

def run_filesystem_agent(task:str) -> str:
    result = _agent.invoke({
        "messages" : [
            HumanMessage(content=task)
        ]
    })
    return result["messages"][-1].content