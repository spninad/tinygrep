import time
from .base import EmbeddingProvider


class VoyageProvider(EmbeddingProvider):
    DEFAULT_MODEL = "voyage-4-large"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        voyage_input_type = "document" if input_type == "document" else "query"
        result = self._embed_with_retry(texts, voyage_input_type)
        return result.embeddings

    def _embed_with_retry(self, texts: list[str], input_type: str, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return self.client.embed(texts, model=self._model, input_type=input_type)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "voyage"

    @property
    def batch_size(self) -> int:
        return 128

    @property
    def max_tokens_per_batch(self) -> int:
        # Voyage hard limit is 120k; stay well clear of it.
        return 100_000
