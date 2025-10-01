import os
import json
import glob
import hashlib
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. Qdrant 연결
# ---------------------------
qdrant = QdrantClient(
    host="localhost",  # Docker Qdrant 컨테이너에서 실행 중
    port=6333
)

# ---------------------------
# 2. 컬렉션 설정
# ---------------------------
collections = {
    "legal_collection": {
        "category": "legal"
    },
    "patent_collection": {
        "category": "patent"
    }
}

# 벡터 차원 (모델에 맞게 수정)
VECTOR_DIM = 768

# 컬렉션 생성
for collection_name in collections.keys():
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE
        )
    )

# ---------------------------
# 3. 임베딩 모델 로드
# ---------------------------
embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# ---------------------------
# 4. JSON 데이터 로드 함수
# ---------------------------
def process_json_file(file_path, collection_name, category):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = []
    for i, item in enumerate(data):
        # ---- 구조 파악: QA or 요약 ----
        if "질문" in item and "답변" in item:
            task = "QA"
            question = item.get("질문", "")
            answer = item.get("답변", "")
            content = None
            summary = None
            text_to_embed = f"질문: {question}\n답변: {answer}"
        elif "본문" in item and "요약" in item:
            task = "SUMMARY"
            question, answer = None, None
            content = item.get("본문", "")
            summary = item.get("요약", "")
            text_to_embed = f"본문: {content}\n요약: {summary}"
        else:
            continue  # 스킵

        # ---- 메타데이터 ----
        payload = {
            "category": category,
            "sub_type": os.path.basename(file_path).split("_")[1] if "_" in file_path else "unknown",
            "task": task,
            "question": question,
            "answer": answer,
            "content": content,
            "summary": summary,
            "file": os.path.basename(file_path),
            "lang": "kr" if "kr" in os.path.basename(file_path) else "unknown"
        }

        # ---- 임베딩 ----
        embedding = embed_model.encode(text_to_embed).tolist()

        # ---- 고유 ID ----
        uid = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()

        points.append(
            models.PointStruct(
                id=uid,
                vector=embedding,
                payload=payload
            )
        )

    if points:
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"✅ {file_path} → {len(points)}개 저장 완료 ({collection_name})")

# ---------------------------
# 5. 데이터 경로 설정
# ---------------------------
BASE_DIR = "/mnt/c/Users/Admin/Downloads/dataset"

DATASETS = {
    "legal_collection": os.path.join(BASE_DIR, "ip_legal_data/train/labeled/**/*.json"),
    "patent_collection": os.path.join(BASE_DIR, "patent_data/train/raw/kr*.json")
}

# ---------------------------
# 6. 실행
# ---------------------------
if __name__ == "__main__":
    for collection_name, path_pattern in DATASETS.items():
        category = collections[collection_name]["category"]
        files = glob.glob(path_pattern, recursive=True)

        for file in tqdm(files, desc=f"{collection_name}"):
            process_json_file(file, collection_name, category)
