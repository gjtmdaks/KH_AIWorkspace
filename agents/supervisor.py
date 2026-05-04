# 감독자 에이전트
#   사용자의 질문을 분석해 어떤 에이전트에게 작업을 분배할지 정하는 마스터 노드
# 사용자 요청
#     │
#     ▼
# [Supervisor: classify_node]
#     LLM이 의도 분류
#     │
#     ├── "legal_qa" ───────────────────────────────────────────────┐
#     │                                                             │
#     └── "report" ──────────────────┐                              │
#                                    │                              │
#                                    ▼                              ▼
#                           [research_node]               [legal_qa_node]
#                           Tavily 웹검색 +                 RAG 검색 +
#                           RAG 내부검색                    LLM 답변 생성
#                                    │                              │
#                                    ▼                              │
#                           [writing_node]                          │
#                           GPT-4o로 구조화된                        │
#                           마크다운 보고서 작성                      │
#                                    │                              │
#                                    ▼                              │
#                           [review_node]                           │
#                           GPT-4o-mini로 검수                       │
#                           (정확성·완전성·가독성)                    │
#                                    │                              │
#                     ┌──────────────┴──────────────┐               │
#                     │ 불합격 & 수정횟수 미초과      │ 합격 or 초과   │
#                     │                             │               │
#                     ▼                             │               │
#              [writing_node]                       │               │
#              (재작성)                              │               │
#                                                   └─────┬─────────┘
#                                                         ▼
#                                                        END
#                                                    최종 응답 반환

import uuid
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import SupervisorState
from agents.legal_qa_agent import ask_legal_question
from agents.research_agent import run_research
from agents.writing_agent import run_writing
from agents.review_agent import run_review
from config import settings
from agents.filesystem_agent import run_filesystem_agent

_llm = ChatOpenAI(
    model = settings.llm_model,
    temperature=0
)

def classity_node(state:SupervisorState) -> dict:
    response = _llm.invoke([
        SystemMessage(content=(
            "당신은 사용자 요청을 분류하는 엄격한 라우터입니다.\n\n"
            "【legal_qa】 — 아래 조건을 모두 만족할 때만 선택:\n"
            "  · 특정 법률·시행령·헌법 조항의 내용, 해석, 적용을 묻는 질문\n"
            "  · 법적 권리·의무·제재·절차에 대한 구체적 질문\n"
            "  · 예: '근로기준법상 연장근로 한도는?', '개인정보보호법 위반 시 과태료 기준은?'\n\n"
            "【report】 — 아래 중 하나라도 해당하면 선택:\n"
            "  · 특정 주제의 조사·분석·정리·요약 요청\n"
            "  · 보고서·문서·글 작성 요청\n"
            "  · 법률 주제라도 '~에 대해 설명해줘', '~를 정리해줘' 같은 서술 요청\n"
            "  · 예: '개인정보보호법 동향 보고서 작성해줘', '환경 규제 현황 분석해줘'\n\n"
            "【unknown】 — 법률·보고서와 무관한 일반 질문 또는 잡담:\n"
            "  · 날씨, 요리, 스포츠, 기술 등 법률과 전혀 관계없는 질문\n"
            "  · 예: '오늘 날씨 어때?', '파이썬 문법 알려줘', '점심 뭐 먹을까?'\n\n"
            "'legal_qa', 'report', 'unknown' 중 하나만 답하세요."
            "모든 답변은 한국어로 답변해주세요."
        )),
        HumanMessage(content=state["question"])
    ])
    res = response.content.lower()
    if "legal_qa" in res:
        intent = "legal_qa"
    elif "report" in res:
        intent = "report"
    else:
        intent = "unknown"
    return {"intent":intent}

# 멀티에이전트 노드들
def legal_qa_node(state:SupervisorState) -> dict:
    result = ask_legal_question(state["question"])
    answer = result["answer"]
    if result["sources"]:
        answer += f"\n\n 출처 : {', '.join(result['sources'])}"
    return {
        "legal_answer" : result["answer"],
        "legal_sources" : result["sources"],
        "final_answer" : answer,
        "messages" : [AIMessage(content=answer)]
    }

def research_node(state:SupervisorState) -> dict:
    topic = state["question"]
    data = run_research(topic)
    return {
        "research_topic" : topic,
        "research_data" : data,
        "messages" : [AIMessage(content=f"수집한 데이터 : {data}")]
    }

def writing_node(state:SupervisorState) -> dict:
    rev = state.get("revision_count", 0) + 1
    draft = run_writing(state["research_topic"], state["research_data"])
    return {
        "draft_report" : draft,
        "revision_count" : rev,
        "messages" : [AIMessage(content=f"보고서 작성 완료 (수정 {rev}회차)")]
    }

def review_node(state:SupervisorState) -> dict:
    result = run_review(state["draft_report"])
    avg = (result.accuracy + result.completeness + result.readability) / 3
    rev = state.get("revision_count", 0)
    max_rev = state.get("max_revisions", 3)

    if result.passed:
        msg = f"검수 합격 (평균 : {avg:.1f}점)"
        final_answer = state["draft_report"]
    elif rev >= max_rev:
        msg = f"최대 수정 횟수 초과. 현재 보고서를 최종 보고서로 출력합니다."
        final_answer = state["draft_report"]
    else:
        msg = f"검수 불합격. (평균 : {avg:.1f}점). 피드백 : {result.feedback}"
        final_answer = ""

    return {
        "review_result" : result.feedback,
        "review_passed" : result.passed,
        "final_answer" : final_answer,
        "messages" : [AIMessage(content=msg)]
    }

def unknown_node(_:SupervisorState) -> dict:
    msg = (
        """
            죄송합니다. 저는 법률 질의응답과 보고서 작성만 지원합니다.
            법률 조항 및 보고서 작성요청을 해주세요.
        """
    )
    return {
        "final_answer" : msg,
        "messages" : [AIMessage(content=msg)]
    }

def save_node(state:SupervisorState) -> dict:
    from datetime import datetime
    from pathlib import Path

    topic = state.get("research_topic", "보고서")
    timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestemp}_{topic}.md"
    filepath = str(Path(__file__).parent.parent / "downloads" / filename)

    task = f"다음 마크다운 보고서를 {filepath} 경로에 저장해줘 \n\n {state['final_answer']}"
    run_filesystem_agent(task)

    return {
        "saved_path" : filepath,
        "mesages" : [AIMessage(content="보고서 작성 완료")]
    }

# 라우팅 함수
def route_by_intent(state:SupervisorState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "legal_qa":
        return "legal_qa"
    elif intent == "report":
        return "report"
    else:
        return "unknown"
    
def route_after_review(state:SupervisorState) -> str:
    if state.get("review_passed"):
        return "save"
    if state.get("revision_count", 0) >= state.get("max_revisions", 3):
        return "save"
    return "writing"

# 그래프 구성
def build_supervisor():
    graph = StateGraph(SupervisorState)

    graph.add_node("classify", classity_node)
    graph.add_node("legal_qa", legal_qa_node)
    graph.add_node("research", research_node)
    graph.add_node("writing", writing_node)
    graph.add_node("review", review_node)
    graph.add_node("unknown", unknown_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "legal_qa" : "legal_qa",
            "report" : "research",
            "unknown" : "unknown"
        }
    )
    graph.add_edge("legal_qa", END)
    graph.add_edge("unknown", END)
    graph.add_edge("research", "writing")
    graph.add_edge("writing", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "writing" : "writing",
            "save" : "save"
        }
    )
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app

supervisor = build_supervisor()

def ask_supervisor(question:str, max_revisions:int = 3, thread_id:str = "") -> dict:
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable":{"thread_id":thread_id}}
    initial_state: SupervisorState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "intent": "",
        "legal_answer": "",
        "legal_sources": [],
        "research_topic": "",
        "research_data": [],
        "draft_report": "",
        "review_result": "",
        "review_passed": False,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "final_answer": "",
        "saved_path": "",
    }

    result = supervisor.invoke(initial_state, config)

    return {
        "status" : "done",
        "thread_id" : thread_id,
        "intent" : result.get("intent"),
        "answer" : result.get("final_answer"),
        "sources" : result.get("legal_sources", []),
        "message" : ""
    }