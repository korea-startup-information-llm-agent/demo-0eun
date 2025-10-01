# demo-0eun


pipe-line 
# 🔎 RAG Workflow with Tools

본 프로젝트는 LLaMA 기반 분류기, 내부 데이터 리트리버, 외부 검색 툴(Tavily)을 활용한 RAG 파이프라인을 구성합니다.  

---

## 📌 Workflow 개요

```text
User Question
     │
     ▼
 [Classifier Tool (LLaMA)]
     │
     ├── 민감 → "답변 불가" 반환
     │
     └── 일반/법률 질문
             │
             ▼
       [Retriever Tool]
             │
             ▼
   관련 문서 + 질문 context
             │
             ▼
   [LLM (답변 생성기)]
             │
             ▼
     최종 응답 (사용자에게)
