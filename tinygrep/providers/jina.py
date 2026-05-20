import time
import httpx
from .base import EmbeddingProvider

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


class JinaProvider(EmbeddingProvider):
    DEFAULT_MODEL = "jina-embeddings-v3"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self._model = model

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        task = "retrieval.passage" if input_type == "document" else "retrieval.query"
        return self._embed_with_retry(texts, task)

    def _embed_with_retry(self, texts: list[str], task: str, max_retries: int = 3) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self._model, "input": texts, "task": task}

        for attempt in range(max_retries):
            try:
                resp = httpx.post(JINA_API_URL, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                # Jina returns data sorted by index
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
                else:
                    raise
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "jina"

    @property
    def batch_size(self) -> int:
        return 64
