import os
import json
import glob
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

# -------------------------
# 1. 설정
# -------------------------
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
LEGAL_PATH = "/mnt/c/Users/Admin/Downloads/dataset/ip_legal_data/train/labeled"
PATENT_PATH = "/mnt/c/Users/Admin/Downloads/dataset/patent_data/train/raw"

# Qdrant 서버 접속 (Docker에서 띄운 경우 보통 localhost:6333)
qdrant = QdrantClient(host="localhost", port=6333)

# -------------------------
# 2. 컬렉션 생성 (법률/특허)
# -------------------------
# 한 번만 실행하면 됨 (있으면 덮어쓰지 않음)
qdrant.recreate_collection(
    collection_name="legal_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
qdrant.recreate_collection(
    collection_name="patent_collection",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

# -------------------------
# 3. 임베딩 모델 (GPU 사용)
# -------------------------
embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

# -------------------------
# 4. 업로드 함수
# -------------------------
def upload_to_qdrant(path, collection_name, category):
    files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

    for file in tqdm(files, desc=f"Processing {category}"):
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] {file}: {e}")
                continue

        points = []
        for i, item in enumerate(data):
            if "질문" in item and "답변" in item:
                text = f"질문: {item['질문']}\n답변: {item['답변']}"
            elif "본문" in item and "요약" in item:
                text = f"본문: {item['본문']}\n요약: {item['요약']}"
            else:
                continue

            embedding = embed_model.encode(text).tolist()

            points.append({
                "id": hash(f"{file}_{i}"),
                "vector": embedding,
                "payload": {
                    "file": os.path.basename(file),
                    "type": category,
                    "text": text
                }
            })

        if points:
            qdrant.upsert(collection_name=collection_name, points=points)

# -------------------------
# 5. 실행
# -------------------------
if __name__ == "__main__":
    upload_to_qdrant(LEGAL_PATH, "legal_collection", "legal")

    # 특허 데이터 → kr*.json만
    kr_files = glob.glob(os.path.join(PATENT_PATH, "kr*.json"))
    for file in tqdm(kr_files, desc="Processing patent (kr only)"):
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue

        points = []
        for i, item in enumerate(data):
            text = f"{item.get('title','')} {item.get('abstract','')}".strip()
            if not text:
                continue
            embedding = embed_model.encode(text).tolist()

            points.append({
                "id": hash(f"{file}_{i}"),
                "vector": embedding,
                "payload": {
                    "file": os.path.basename(file),
                    "type": "patent",
                    "lang": "kr",
                    "text": text
                }
            })

        if points:
            qdrant.upsert(collection_name="patent_collection", points=points)

    print("✅ Embedding complete! 데이터가 Qdrant에 저장되었습니다.")
