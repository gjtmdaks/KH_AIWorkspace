# RAG 검색 에이전트가 사용할 도구 목록.
from langchain_core.tools import tool
from knowledge_base.retriever import similarity_search, mmr_search, format_docs
from config import settings

@tool
def search_legal_docs_tool(query:str, use_mmr:bool = False) -> str:
    try:
        docs = mmr_search(
            query,
            settings.retrieval_k,
            settings.retrieval_fetch_k
            ) if use_mmr else similarity_search(query, settings.retrieval_k)
        if not docs:
            return "관련된 문서가 없습니다."
        return format_docs(docs)
    
    except FileNotFoundError as e:
        return str(e)