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

# Wrapper 함수
def embed_passage(text: str):
    return get_embedding(text, mode="passage")

def embed_query(text: str):
    return get_embedding(text, mode="query")

# ---------------------------
# 2-1) 레코드 빌더: 법률 데이터
# ---------------------------
def build_legal_record(item: dict):
    # 케이스 1: title + output 구조
    title = item.get("title", "")
    output = item.get("output", "")
    if title or output:
        text = f"[제목] {title}\n[요약] {output}"
        task = "LEGAL_SUMMARY"

    # 케이스 2: 질문/답변 구조
    elif "질문" in item and "답변" in item:
        text = f"[질문] {item.get('질문','')}\n[답변] {item.get('답변','')}"
        task = "LEGAL_QA"

    # 케이스 3: 본문/요약 구조
    elif "본문" in item and "요약" in item:
        text = f"[본문] {item.get('본문','')}\n[요약] {item.get('요약','')}"
        task = "LEGAL_SUMMARY"
    else:
        return None

    meta = {
        "category": "legal",
        "task": task,
        "response_institute": item.get("response_institute"),
        "response_date": item.get("response_date"),
        "title": title or item.get("title"),
        "sentences": item.get("sentences"),
        "file_section": item.get("section"),
    }
    return text, meta

# ---------------------------
# 2-2) 레코드 빌더: 특허 데이터
# ---------------------------
def build_patent_record(item: dict):
    inv = item.get("invention_title", "") or item.get("title", "")
    abstract = item.get("abstract", "")
    kw = item.get("keyword", [])
    if isinstance(kw, list):
        kw_str = ", ".join(map(str, kw))
    else:
        kw_str = str(kw) if kw else ""

    if inv or abstract or kw_str:
        text = f"[발명의명칭] {inv}\n[요약] {abstract}\n[주요키워드] {kw_str}"
        task = "PATENT_ABS"
    else:
        return None

    claims = item.get("claims")
    if isinstance(claims, list):
        claims = "\n".join(map(str, claims))

    meta = {
        "category": "patent",
        "task": task,
        "register_date": item.get("register_date"),
        "open_date": item.get("open_date"),
        "application_date": item.get("application_date"),
        "documentId": item.get("documentId") or item.get("document_id"),
        "title": item.get("title") or inv,
        "claims": claims,
    }
    return text, meta

# ---------------------------
# 3. JSON 데이터 처리 → 로컬 저장
# ---------------------------
def process_json_file(file_path, category, save_file):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):   # ✅ dict 아닌 건 스킵
            continue

        if category == "legal":
            built = build_legal_record(item)
        else:  # "patent"
            built = build_patent_record(item)

        if not built:
            continue

        text_to_embed, metadata = built
        emb = embed_passage(text_to_embed)   # passage 임베딩
        uid = hashlib.md5(f"{file_path}_{i}".encode()).hexdigest()

        embeddings.append({
            "id": uid,
            "vector": emb,
            "metadata": {**metadata, "file": os.path.basename(file_path)}
        })

    if embeddings:
        with open(save_file, "a", encoding="utf-8") as f:
            for e in embeddings:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"✅ {file_path} → {len(embeddings)}개 로컬 저장 완료 ({category})")

# ---------------------------
# 4. 실행
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
    # 5. 검색 예시 (query)
    # ---------------------------
    sample_question = "특허 출원일이 언제인가요?"
    q_vec = embed_query(sample_question)
    print(f"\n🔎 Query Example: {sample_question}")
    print(f"임베딩 차원: {len(q_vec)} (expected 1024)")
    print(f"앞 5개 값: {q_vec[:5]}")
