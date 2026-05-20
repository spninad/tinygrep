from .base import EmbeddingProvider
from .voyage import VoyageProvider
from .jina import JinaProvider


def get_provider(name: str, model: str, api_key: str) -> EmbeddingProvider:
    match name.lower():
        case "voyage":
            return VoyageProvider(api_key=api_key, model=model)
        case "jina":
            return JinaProvider(api_key=api_key, model=model)
        case _:
            raise ValueError(f"Unknown provider '{name}'. Supported: voyage, jina")


__all__ = ["EmbeddingProvider", "VoyageProvider", "JinaProvider", "get_provider"]
