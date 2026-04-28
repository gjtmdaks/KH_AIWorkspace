# 랭서브를 활용한 api서버용 모듈
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from agents.legal_qa_agent import ask_legal_question
from config import settings
from knowledge_base.indexing import run_indexing
# from agents.supervisor import ask_supervisor

"""
    LangServe
     - 랭서브는 LCEL체인을 FastAPI기반 REST API로 자동으로 변환해주는 프레임워크
     - 체인을 REST API로 변환하면 어떠한 언어나, 어떠한 환경에서든 http방식으로 현재
       행체인코드를 호출할 수 있다.
     - FasgAPI기반 서버를 직접 구축하는 수고를 덜어주는 역할을 한다.
"""

# 입출력 스키마
class QuestionInput(BaseModel):
    question : str

class AnswerOutput(BaseModel):
    question : str
    answer : str
    sources : list[str] = []

# LangServe 서버 구축
# run_indexing()

# 1. FastAPI앱 생성
app = FastAPI(
    title="법률 rag api",
    description="법률 문서 기반 질의응답 서버",
    version="1.0.0"
)

# 2. 라우트 추가
legal_qa_runnable = RunnableLambda(lambda state:ask_legal_question(state["question"])
                                   ).with_types(
                                        input_type=QuestionInput,
                                        output_type=AnswerOutput
                                    )

add_routes(
    app,
    legal_qa_runnable,
    path="/legal-qa",
    enabled_endpoints=["invoke", "stream", "stream_log", "playground"]
)

# 서버 실행
