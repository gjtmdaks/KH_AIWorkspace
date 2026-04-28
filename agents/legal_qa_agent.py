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
│   법률 QA 에이전트                    │
│  (legal_qa_agent.py)                 │
│                                      │
│  START                               │
│    │                                 │
│  [route_node]                        │
│    │                                 │
│┌───┴───┐                             │
││     검색필요 ──▶ [retrieve_node]    │
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
    response = llm.invoke(prompt.format_messages())
    need_retrieval = "yes" in response.content.lower()

    return {**state, "need_retrieval":need_retrieval}

# rag를 수행하는 노드
def retrieve_node(state:LegalQAState) -> LegalQAState:
    docs = mmr_search(state["question"], k=5, fetch_k=20)
    sources = list(doc.metadata.get("file_name", "unknown") for doc in docs)
    return {**state, "retrieved_docs":docs, "sources":sources}

# 검색된 문서로 답변을 생성하는 노드
def generate_node(state:LegalQAState) -> LegalQAState:
    question = state["question"]
    docs = state.get("retrieved_docs", [])

    if docs:
        context = format_docs(docs)
        system_prompt = (
            """
                당신은 대한민국 법률 전문 ai입니다.
                아래 법률조항을 참고하여 질문에 정확하고 이해하기 쉽게 답변하세요.
                답변 마지막에는 근거조항(출처 및 페이지)을 반드시 명시하세요.

                [참고 법률 조항]\n
            """+context
        )
    else :
        system_prompt = """
                            당신은 대한민국 법률 전문 ai 어시스턴스입니다.
                            일반적인 법률지식을 바탕으로 질문에 답변하세요.
                        """
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)

    return {**state, "answer":response.content}

# 라우팅 함수
def should_retrieve(state:LegalQAState) -> str:
    if state.get("need_retrieval"):
        return "retrieve"
    else :
        return "generate"
    
def build_legal_qa_agent() -> StateGraph:
    graph = StateGraph(LegalQAState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        should_retrieve,
        {
            "retrieve" : "retrieve",
            "generate" : "generate"
        }
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

legal_qa_agent = build_legal_qa_agent()

# 외부공개용 인터페이스
def ask_legal_question(question:str) -> dict:
    initial_state:LegalQAState = {
        "question" : question,
        "retrieved_docs" : [],
        "answer" : "",
        "sources" : [],
        "need_retrieval" : True
    }
    result = legal_qa_agent.invoke(initial_state)
    return {
        "question" : result["question"],
        "answer" : result["answer"],
        "sources" : result["sources"]
    }