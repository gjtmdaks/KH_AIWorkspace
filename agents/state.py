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