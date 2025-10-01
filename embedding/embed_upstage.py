import os
import json
import glob
import hashlib
import requests
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------
# 0. 환경변수 로드 (.env)
# ---------------------------
load_dotenv()
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
UPSTAGE_EMBED_URL = "https://api.upstage.ai/v1/embeddings"

# ---------------------------
# 1. Upstage 임베딩 함수
# ---------------------------
def get_embedding(text: str, mode: str = "passage"):
    """
    mode: "passage" (문서 저장용) or "query" (검색용)
    """
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": f"solar-embedding-1-large-{mode}",  # passage / query 구분
        "input": [text]
    }
    resp = requests.post(UPSTAGE_EMBED_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]

# 별도 wrapper 함수 (가독성용)
def embed_passage(text: str):
    return get_embedding(text, mode="passage")

def embed_query(text: str):
    return get_embedding(text, mode="query")

# ---------------------------
# 2. JSON 데이터 처리 → 로컬 저장
# ---------------------------
def process_json_file(file_path, category, save_file):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    for i, item in enumerate(data):
        if "질문" in item and "답변" in item:
            task = "QA"
            question = item.get("질문", "")
            answer = item.get("답변", "")
            content, summary = None, None
            text_to_embed = f"[질문] {question}\n[답변] {answer}"
        elif "본문" in item and "요약" in item:
            task = "SUMMARY"
            question, answer = None, None
            content = item.get("본문", "")
            summary = item.get("요약", "")
            text_to_embed = f"[본문] {content}\n[요약] {summary}"
        else:
            continue

        # passage 임베딩
        embedding = embed_passage(text_to_embed)

        uid = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()

        embeddings.append({
            "id": uid,
            "vector": embedding,
            "metadata": {
                "category": category,
                "task": task,
                "question": question,
                "answer": answer,
                "content": content,
                "summary": summary,
                "file": os.path.basename(file_path)
            }
        })

    if embeddings:
        with open(save_file, "a", encoding="utf-8") as f:
            for e in embeddings:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"✅ {file_path} → {len(embeddings)}개 로컬 저장 완료 ({category})")

# ---------------------------
# 3. 실행 (임베딩 저장)
# ---------------------------
BASE_DIR = "/mnt/c/Users/Admin/Downloads/dataset"
DATASETS = {
    "legal": os.path.join(BASE_DIR, "ip_legal_data/train/labeled/**/*.json"),
    "patent": os.path.join(BASE_DIR, "patent_data/train/raw/kr*.json")
}

if __name__ == "__main__":
    for category, path_pattern in DATASETS.items():
        files = glob.glob(path_pattern, recursive=True)
        for file in tqdm(files, desc=f"{category}"):
            process_json_file(file, category, save_file=f"{category}_embeddings.json")

    # ---------------------------
    # 4. 검색 예시 (query)
    # ---------------------------
    sample_question = "특허 출원일이 언제인가요?"
    q_vec = embed_query(sample_question)
    print(f"\n🔎 Query Example: {sample_question}")
    print(f"임베딩 차원: {len(q_vec)} (expected 1024)")
    print(f"앞 5개 값: {q_vec[:5]}")
