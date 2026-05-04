# 멀티 에이전트 1번
#   리서치 에이전트
#    - 웹 검색도구를 활용하여 주어진 토픽에 대한 자료를 조사하는 에이전트
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from config import settings

# 도구정의
# 1. Tavily를 검색 도구
#  - Langchain Tool인터페이스를 이미 구현하고 있어서 @tool데코레이터 없이 바로 도구로 등록 가능
web_search = TavilySearchResults(
    max_results = 5,
    tavily_api_key = settings.tavily_api_key,
    name = "web_search",
    description = "웹에서 최신정보를 검색하는 도구, 최신 동향이나 뉴스검색이 필요한 경우 사용하세요."
)

_model = ChatOpenAI(
    model = settings.llm_model,
    temperature = 0
)

research_agent = create_agent(
    _model,
    [web_search],
    system_prompt="""
        당신은 전문 리서치 에이전트입니다.
        웹 검색과 내부 법률지식 베이스를 활용하여 주어진 주제를 조사해주세요.
        답변은 한국어로 답변해주세요.
    """
)

# 공개인터페이스
def run_research(topic:str) -> list[str]:
    result = research_agent.invoke({
        "messages" : [{"role":"user", "content":f"다음 주제에 대해 조사해주세요 : {topic}"}]
    })
    final_content = result["messages"][-1].content
    return [final_content]