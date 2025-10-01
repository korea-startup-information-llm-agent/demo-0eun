import requests

class TavilySearchTool:
    """
    Tavily API를 이용한 외부 검색 툴
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str) -> dict:
        """
        Args:
            query (str): 검색 질의

        Returns:
            dict: 검색 결과 JSON
        """
        # TODO: Tavily API 호출 (현재는 dummy response)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"query": query, "num_results": 3}

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
