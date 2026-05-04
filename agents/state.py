# 에이전트들이 사용할 상태값을 정의하는 파일
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class LegalQAState(TypedDict):
    question : str
    retrieved_docs : list[Document] # 벡터스토어에서 검색된 문서들
    answer : str
    sources : list[str]
    need_retrieval : bool

class SupervisorState(TypedDict):
    # 모든 에이전트 공용으로 사용하는 상태값
    messages : Annotated[list[BaseMessage], add_messages]
    question : str
    intent : str # 슈퍼바이저가 분류한 질문의 의도 "legal_qa" | "report" | "unknown"

    # 법률 QA경로 결과
    legal_answer : str # 법률 에이전트가 생성한 답번
    legal_sources : list[str] # 출처목록

    # 보고서 작성 에이전트 상태값
    # 검색 에이전트
    research_topic: str                                    # 조사 주제 (= question)
    research_data: list[str]                                # 수집 자료 (호출마다 초기화)

    # 작성 에이전트
    draft_report: str                                      # 보고서 초안

    # 리뷰 에이전트
    review_result: str                                     # 검수 피드백
    review_passed: bool                                    # 검수 합격 여부
    revision_count: int                                    # 수정 횟수
    max_revisions: int                                     # 최대 수정 횟수

    # ── 최종 출력 ────────────────────────────────────────────────
    final_answer: str       # 사용자에게 반환할 최종 답변 (법률QA 또는 완성된 보고서)

    # ── 파일 저장 ───────────────────────────────────────────────
    saved_path: str          # save_node 가 저장한 보고서 파일 절대 경로