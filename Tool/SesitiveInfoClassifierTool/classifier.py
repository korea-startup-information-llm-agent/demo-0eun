class SensitiveInfoClassifierTool:
    """
    질문이 민감정보 관련인지 여부를 분류하는 툴
    - 민감정보: 개인정보, 금융정보, 보안정보 등
    - 일반/법률 질문: 그대로 다음 단계로 전달
    """

    def __init__(self):
        # TODO: LLaMA 모델 로드 (추후 교체)
        pass

    def classify(self, query: str) -> str:
        """
        Args:
            query (str): 사용자 질문

        Returns:
            str: "sensitive" or "normal"
        """
        sensitive_keywords = ["주민등록번호", "계좌번호", "비밀번호", "전화번호", "신용카드"]
        if any(word in query for word in sensitive_keywords):
            return "sensitive"
        return "normal"
