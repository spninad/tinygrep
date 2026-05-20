from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        """Embed a batch of texts. input_type is 'document' or 'query'."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    def batch_size(self) -> int:
        return 64
