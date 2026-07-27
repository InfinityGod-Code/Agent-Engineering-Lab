import httpx
from typing import Optional


class RemoteClient:
    def __init__(self, base_url: str = ""):
        self.client = httpx.Client()
        self.base_url = base_url

    def get(self, endpoint: str) -> httpx.Response:
        return self.client.get(self.base_url + endpoint)

    def post(self, endpoint: str, data: Optional[dict] = None) -> httpx.Response:
        return self.client.post(self.base_url + endpoint, json=data)

    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
