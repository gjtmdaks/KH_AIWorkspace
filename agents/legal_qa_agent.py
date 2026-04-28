from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from agents.state import LegalQAState
from knowledge_base.retriever import mmr_search, format_docs
from config import settings

# 법률문서 기반의 질의응답을 처리하는 단일 에이전트
"""
┌──────────────────────────────────────┐
│   법률 QA 에이전트                   │
│  (legal_qa_agent.py)                 │
│                                      │
│  START                               │
│    │                                 │
│  [route_node]                        │
│    │                                 │
│┌───┴───┐                             │
││     검색필요 ──▶ [retrieve_node]   │
││                        │            │
││                        │            │
│└─────────────┐          │            │
│              ▼          │            │
│        [generate_node]◀┘            │
│              │                       │
│             END                      │
└──────────────────────────────────────┘
"""
# LLM초기화
llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0
)

# 노드함수
# 1. 라우트노드
#  - 사용자의 질문이 법률관련지식인지 판단하는 라우터 함수.
#  - 상세한 법률조항지식이 필요한 경우 rag로 데이터를 추출할 수 있도록 처리한다.
def route_node(state:LegalQAState) -> LegalQAState:
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
                      당신은 법률 문서 검색 필요 여부를 판단하는 라우터입니다.
                      이 질문은 이미 법률 관련 질문임이 확인되었습니다.
                      아래 기준으로 벡터 스토어에서 특정 법령 조항을 검색해야하는지 판단하세요.
                      [검색 필요 -> yes]
                      ex) 특정 법률의 요건, 예외규정등 정확한 조항확인이 필요한 경우
                      [검색 불필요 -> no]
                      ex) 법원이란? 불법행위 책임이란?
                      """),
                      HumanMessage(content=question)
    ])
    response = llm.invoke(prompt.from_messages())
    need_retrieval = "yes" in response.content.lower()

    return {**state, "need_retrieval":need_retrieval}