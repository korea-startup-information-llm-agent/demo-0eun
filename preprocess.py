import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# 1. 데이터 로드 함수
# -----------------------------
def load_json_files(base_dir):
    texts = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)

                        # raw 문서 구조 (sentences 존재)
                        if "sentences" in data:
                            content = " ".join(data["sentences"])
                            texts.append(content)

                        # 혹시 라벨링 데이터가 들어있을 수도 있으니 대비
                        elif "taskinfo" in data and "input" in data["taskinfo"]:
                            q = data["taskinfo"]["input"]
                            a = data["taskinfo"]["output"]
                            texts.append(f"질문: {q}\n답변: {a}")

                except Exception as e:
                    print(f"[오류] {path}: {e}")
    return texts


# -----------------------------
# 2. 텍스트 분할기 (한글 기준)
# -----------------------------
def chunk_texts(texts, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", "?", "!"]
    )
    docs = splitter.create_documents(texts)
    return docs


# -----------------------------
# 3. 벡터DB 구축
# -----------------------------
def build_vector_db(docs, persist_dir="./chroma_db"):
    model_name = "intfloat/multilingual-e5-large-instruct" 
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    print(f"✅ ChromaDB에 {len(docs)}개 청크 저장 완료: {persist_dir}")


# -----------------------------
# 실행부
# -----------------------------
if __name__ == "__main__":
    dataset_path = "./data/ip_legal_data"

    # 1) 데이터 로드
    texts = load_json_files(dataset_path)
    print(f"총 {len(texts)}개 문서 로드 완료")

    # 2) 청크 분리
    docs = chunk_texts(texts, chunk_size=500, chunk_overlap=50)
    print(f"총 {len(docs)}개 청크 생성 완료")

    # 3) 벡터DB 생성
    build_vector_db(docs, persist_dir="./chroma_db")
