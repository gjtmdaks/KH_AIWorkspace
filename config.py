# config.py
#  - 다른 모듈에서 공통으로 사용하는 설정값을 정의

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    # pydantic_settings
    #  - 파이썬 프로젝트의 설정정보를 관리하기 위해
    #    사용하는 라이브러리
    #  - 환경변수 자동 로드 기능과, 타입검증, 타입변화 기능들이 내장

    model_config = SettingsConfigDict(
        env_file=".env", # 설정정보를 읽을 파일명
        env_file_encoding="utf-8",
        extra="ignore" # 현재 클래스에서 정의하지 않은 환경변수는 무시하는 설정
    )

    # api key
    openai_api_key:str = "" # OPENAI API KEY
    tavily_api_key:str = "" # TAVILY API KEY

    # langsmith
    langsmith_tracing:bool = False
    langsmith_endpoint:str = ""
    langsmith_api_key:str = ""
    langsmith_project:str = ""

    # LLM모델 설정
    llm_model:str = "gpt-4o"
    llm_mini_model:str = "gpt-4o-mini"
    embedding_model:str = "text-embedding-3-small"

    # RAG 파라미터
    chunk_size:int = 500
    chunk_overlap:int = 50
    retrieval_k:int = 5
    retrieval_fetch_k:int = 20

    # 경로 설정
    #  - 프로젝트 루트디렉토리 기준 상대경로
    knowledge_base:Path = Path("knowledge_base")
    vector_store:Path = Path("knowledge_base/vector_store")
    documents_dir:Path = Path("knowledge_base/documents")

settings = Settings()