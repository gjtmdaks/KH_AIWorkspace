# 멀티 에이전트3
#   리뷰 에이전트
#    - 작성된 보고서를 평가하는 에이전트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings

class ReviewResult(BaseModel):
    accuracy: int = Field(ge=1, le=10, description="정확성 점수 (1-10): 기술적으로 정확한가?")
    completeness: int = Field(ge=1, le=10, description="완전성 점수 (1-10): 주제를 충분히 다루었는가?")
    readability: int = Field(ge=1, le=10, description="가독성 점수 (1-10): 구조가 명확하고 읽기 쉬운가?")
    passed: bool = Field(description="합격 여부 — 세 점수 평균 7점 이상이면 True")
    feedback: str = Field(description="구체적인 피드백 및 수정 요청 사항")

_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "당신은 전문 리뷰 분석가입니다. 보고서를 다음 기준으로 검수하세요. \n"
        "1. 정확성(1~10) : 기술적으로 정확한가\n"
        "2. 완전성(1~10) : 주제를 충분히 다루었는가\n"
        "3. 가독성(1~10) : 구조가 명확하고 읽기 쉬운가\n"
        "평균7점 이상이면 passed=True, 미만이면 passed=False로 판정하세요."
    )),
    ("human", "다음 보고서를 검수해주세요 : \n{draft_report}")
])

_llm = ChatOpenAI(
    model = settings.llm_mini_model,
    temperature=0
)

_review_chain = _prompt | _llm.with_structured_output(ReviewResult)

def run_review(draft_report:str) -> ReviewResult:
    return _review_chain.invoke({"draft_report":draft_report})