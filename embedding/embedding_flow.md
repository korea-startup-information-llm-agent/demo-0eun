Upstage 기반 임베딩 파이프라인

개요

이 모듈(embed_upstage.py)은 Upstage Solar 임베딩 API를 활용해 법률/특허
데이터셋을 벡터화하고, 로컬에 저장합니다.
향후 Qdrant 서버 세팅 시, 이 로컬 벡터들을 업로드하여 검색 시스템에
연결할 수 있습니다.

------------------------------------------------------------------------

사용 모델

-   solar-embedding-1-large-passage → 문서(passages) 임베딩용
-   solar-embedding-1-large-query → 사용자 질의(query) 임베딩용
-   차원(Dimension): 1024 (두 모델 동일)

------------------------------------------------------------------------

데이터셋 처리

📂 법률 데이터 (ip_legal_data/train/labeled/)

-   질의응답 데이터

        { "질문": "...", "답변": "..." }

-   요약 데이터

        { "본문": "...", "요약": "..." }

-   기타 구조 (title + output 등)도 대응

📂 특허 데이터 (patent_data/train/raw/)

-   주요 필드:
    -   발명의명칭 (invention_title)
    -   요약 (abstract)
    -   키워드 (keyword)
    -   청구항 (claims)
    -   날짜/식별자 메타데이터 (application_date, register_date,
        documentId 등)

------------------------------------------------------------------------

스키마 (저장 형식)

저장되는 로컬 파일은 .jsonl 형식입니다.
각 줄은 하나의 벡터 + 메타데이터 레코드:

    {
      "id": "해시값",
      "vector": [0.01, -0.02, ...],   // passage 임베딩 (1024차원)
      "metadata": {
        "category": "legal" | "patent",
        "task": "LEGAL_QA" | "LEGAL_SUMMARY" | "PATENT_ABS",
        "question": "...",
        "answer": "...",
        "content": "...",
        "summary": "...",
        "title": "...",
        "claims": "...",
        "file": "원본파일명.json"
      }
    }

------------------------------------------------------------------------

실행 방법

1.  .env 파일 생성

        UPSTAGE_API_KEY=발급받은_API_KEY

2.  스크립트 실행

        python embed_upstage.py

3.  결과 파일 확인

    -   legal_embeddings.json
    -   patent_embeddings.json

------------------------------------------------------------------------

Query 예시

    from embed_upstage import embed_query

    q = "특허 출원일이 언제인가요?"
    vec = embed_query(q)
    print(len(vec))  # 1024
    print(vec[:5])   # 앞 5개 값 출력

------------------------------------------------------------------------

TODO

-   현재는 로컬 저장만 진행
-   추후 Qdrant 서버 세팅 완료 후, 해당 JSONL을 읽어 upsert() 방식으로
    벡터 DB에 업로드 예정
