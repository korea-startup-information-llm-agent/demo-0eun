import os
import glob
import json
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------
# 1. ChromaDB 초기화
# ---------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")

legal_collection = chroma_client.get_or_create_collection("legal_collection")
patent_collection = chroma_client.get_or_create_collection("patent_collection")

# ---------------------------
# 2. 한국어 임베딩 모델 로드 (GPU 자동 사용)
# ---------------------------
embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask", device="cuda")

# ---------------------------
# 3. 유틸 함수 (JSON 파일 로딩)
# ---------------------------
def load_json_files(base_path, filter_prefix=None):
    file_list = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".json"):
                if filter_prefix and not f.startswith(filter_prefix):
                    continue
                file_list.append(os.path.join(root, f))
    return file_list

# ---------------------------
# 4. 임베딩 & 저장 함수
# ---------------------------
def embed_and_store(file_list, collection, category_name):
    for fpath in tqdm(file_list, desc=f"Embedding {category_name}"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 문서 내용 추출 (예시: 'content' 키 or 전체 json -> string)
            if isinstance(data, dict):
                text = json.dumps(data, ensure_ascii=False)
            elif isinstance(data, list):
                text = "\n".join([json.dumps(d, ensure_ascii=False) for d in data])
            else:
                text = str(data)

            # 임베딩 생성
            embedding = embed_model.encode([text]).tolist()

            # ChromaDB 저장
            collection.add(
                documents=[text],
                embeddings=embedding,
                metadatas=[{"source": fpath}],
                ids=[fpath]  # 파일 경로를 ID로 사용
            )

        except Exception as e:
            print(f"❌ Error in {fpath}: {e}")

# ---------------------------
# 5. 실행 부분
# ---------------------------
if __name__ == "__main__":
    # ⚖️ 지식재산권 데이터셋 (train/labeled 기준)
    legal_files = load_json_files(
        "./dataset/ip_legal_data/train/labeled"
    )

    # ⚙️ 기술특허 데이터셋 (train/raw 중 kr 파일만)
    patent_files = load_json_files(
        "./dataset/patent_data/train/raw",
        filter_prefix="kr"
    )

    print(f"지식재산권 파일 개수: {len(legal_files)}")
    print(f"기술특허(KR) 파일 개수: {len(patent_files)}")

    # 임베딩 & 저장
    embed_and_store(legal_files, legal_collection, "Legal")
    embed_and_store(patent_files, patent_collection, "Patent")

    print("✅ 모든 데이터 임베딩 및 저장 완료")
