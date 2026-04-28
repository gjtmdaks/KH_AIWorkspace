# indexing.py에서 생성한 벡터 디비를 로드하여
# 다양한 검색함수를 제공하는 모듈
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import settings

# OpenAIEmbeding
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model
    )

# 벡터DB 로드함수
def load_vector_store(vector_dir:Path) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model
    )
    return FAISS.load_local(
        str(vector_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )

# 유사도 검색 함수
def similarity_search(query:str, k:int) -> list[Document]:
    vs = load_vector_store(settings.vector_store)
    return vs.similarity_search(query, k)

# mmr검색 함수
def mmr_search(query:str, k:int, fetch_k:int) -> list[Document]:
    vs = load_vector_store(settings.vector_store)
    return vs.max_marginal_relevance_search(query, k, fetch_k)

# 유사도와 점수를 반환하는 함수
def search_with_score(query:str, k:int) -> list[tuple[Document, float]]:
    vs = load_vector_store(settings.vector_store)
    return vs.similarity_search_with_score(query, k)

# 검색된 Documents리스트를 LLM프롬프트용 문자열로 변환하는 함수
def fomat_docs(docs:list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[{i}] 출처: {file_name} (p.{page})\n{doc.page_content}")
    return "\n\n".join(parts)