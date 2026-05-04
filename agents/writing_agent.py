# 멀티 에이전트 2
#   보고서 작성 에이전트
#    - 수집된 보고서를 바탕으로 구조화된 마크다운 보고서를 작성
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings

# 출력 스키마 정의
class Report(BaseModel):
    title:str = Field(description="보고서의 제목")
    summary:str = Field(description="요약 내용")
    sections:list[str] = Field(description="본문 섹션 리스트")
    conclusion:str = Field(description="결론")
    references:list[str] = Field(description="참고 자료 목록")

# 프롬프트 템플릿
_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        당신은 전문 기술 작가입니다. 주어진 자료를 바탕으로 구조화된 보고서를 작성하세요.
        반드시 다음 형식을 따르세요.
         - 제목, 요약내용(3줄 이내), 본문(3개 이상의 섹션), 결론, 참고자료
    """),
    ("human",(
        "연구주제 : {topic}\n\n"
        "수집된 자료 : \n{research_data}\n\n"
        "위 자료를 바탕으로 보고서를 작성해 주세요."
    ))
])

# 모델설정
_model = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.3
)

# 체인구성
_writing_chain = _prompt | _model.with_structured_output(Report)

# 공개 인터페이스
def run_writing(topic:str, research_data:list[str]) -> str:
    combined = "\n\n".join(research_data)

    report:Report = _writing_chain.invoke({
        "topic":topic,
        "research_data":combined
    })

    # 마크다운 문자열 반환
    sections_text = "\n\n".join(report.sections)
    refs_text = "\n".join(f"- {r}" for r in report.references)

    return (
        f"# {report.title}\n\n"
        f"## 요약내용 : {report.summary}\n\n"
        f"## 본문\n{sections_text}\n\n"
        f"## 결론 : {report.conclusion}\n\n"
        f"## 참고자료 \n{refs_text}"
    )