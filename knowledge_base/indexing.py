# 인덱싱 파이프라인 설정 모듈
# documents폴더에서 데이터를 로드하여, 스플릿 후 임베딩하여
# 벡터스토어에 저장하는 파이프라인을 작성할 예정
# 1회만 로드하도록 작성할 예정
import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
)
from config import settings
load_dotenv()

# 로드할 데이터별 파일 유형 맵
LOADER_MAP = {
    ".pdf" : PyPDFLoader,
    ".txt" : TextLoader,
    ".csv" : CSVLoader,
    ".doc" : UnstructuredWordDocumentLoader
}

# 인덱싱 처리과정
# 1단계 : 데이터 로드
#  - RAG로 사용할 문서이기 때문에 출처를 반드시 기술할 예정
def load_documents(docs_dir:Path) -> list:
    all_docs = []
    files = list(docs_dir.iterdir())

    if not files:
        print({f"{docs_dir}에 파일이 없습니다."})
        return []
    
    for file_path in files:
        ext = file_path.suffix.lower()
        if ext not in LOADER_MAP:
            print("지원하지 않는 형식의 파일 ", {file_path.name})
            continue

        try:
            loader = LOADER_MAP[ext](str(file_path))
            docs = loader.load()

            for doc in docs:
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_type"] = ext

            all_docs.extend(docs)
            print("데이터 로드 완료 ", {file_path.name})

        except Exception as e:
            print("데이터 로드 실패 ", {file_path.name})
    return all_docs

# 2단계 : 텍스트 스플릿
# 1번 단계에서 전달받은 문서를 분할.
def split_documents(documents:list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = settings.chunk_size,
        chunk_overlap = settings.chunk_overlap,
        separators = ["\n\n", "\n", ". ", " "]
    )
    return splitter.split_documents(documents)

# 3단계 : 벡터디비 생성 및 임베딩
#  - 전달받은 청크를 임베딩하여 벡터 디비에 저장
def build_vector_store(chunks:list, vector_dir:Path) -> FAISS:
    embeddings = OpenAIEmbeddings(
        model = settings.embedding_model
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_dir))
    print("벡터디비 생성 완료", vector_dir)

    return vector_store

# 4단계 : 백터디비 로드


# 인덱싱 파이프라인 실행
def run_indexing():
    print("1단계 : 문서로드")
    documents = load_documents(settings.documents_dir)

    print("2단계 : 문서분할")
    chunks = split_documents(documents)

    print("3단계 : 임베딩 생성 및 벡터DB 구축")
    vector_store = build_vector_store(chunks, settings.vector_store)

    print("4단계 : 벡터DB 로드 및 검색 테스트")
    test_query = "개인정보 보호법 1조"
    results = vector_store.similarity_search(test_query, k=settings.retrieval_k)

    return vector_store

if __name__ == "__main__":
    run_indexing()