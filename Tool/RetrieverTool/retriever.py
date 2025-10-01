class RetrieverTool:
    """
    내부 문서(예: 법률, 특허 데이터셋)에서 관련 문서를 찾아주는 툴
    """

    def __init__(self, vector_db=None):
        # TODO: 실제 벡터DB(Chroma, FAISS 등) 연결
        self.vector_db = vector_db

    def retrieve(self, query: str) -> list:
        """
        Args:
            query (str): 사용자 질문

        Returns:
            list: 관련 문서 리스트
        """
        # TODO: vector_db.similarity_search(query) 로 교체
        dummy_docs = [
            {"id": 1, "content": "지식재산권 관련 법률 문서 예시"},
            {"id": 2, "content": "국가중점기술 관련 특허 문서 예시"}
        ]
        return dummy_docs
