from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderConfig:
    name: str = "voyage"
    model: str = ""  # empty = provider default

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        defaults = {"voyage": "voyage-4-large", "jina": "jina-embeddings-v3"}
        return defaults.get(self.name, "")


@dataclass
class IndexConfig:
    chunk_size: int = 2000
    chunk_overlap: int = 200
    file_extensions: list[str] = field(default_factory=lambda: [".md", ".txt", ".mdx"])


@dataclass
class Config:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    index: IndexConfig = field(default_factory=IndexConfig)

    def api_key(self) -> str:
        env_map = {"voyage": "VOYAGE_API_KEY", "jina": "JINA_API_KEY"}
        env_var = env_map.get(self.provider.name)
        if env_var:
            key = os.environ.get(env_var, "")
            if key:
                return key
        raise ValueError(
            f"API key not found. Set the {env_map.get(self.provider.name, 'PROVIDER_API_KEY')} "
            "environment variable."
        )

    @classmethod
    def load(cls, folder: Path) -> Config:
        cfg = cls()
        # Search order: folder config, then global config
        candidates = [
            folder / ".tinygrep" / "config.toml",
            Path.home() / ".config" / "tinygrep" / "config.toml",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                _apply_toml(cfg, data)
                break

        # Environment variable overrides
        if os.environ.get("TINYGREP_PROVIDER"):
            cfg.provider.name = os.environ["TINYGREP_PROVIDER"]
        if os.environ.get("TINYGREP_MODEL"):
            cfg.provider.model = os.environ["TINYGREP_MODEL"]

        return cfg


def _apply_toml(cfg: Config, data: dict) -> None:
    if "provider" in data:
        p = data["provider"]
        if "name" in p:
            cfg.provider.name = p["name"]
        if "model" in p:
            cfg.provider.model = p["model"]
    if "index" in data:
        idx = data["index"]
        if "chunk_size" in idx:
            cfg.index.chunk_size = int(idx["chunk_size"])
        if "chunk_overlap" in idx:
            cfg.index.chunk_overlap = int(idx["chunk_overlap"])
        if "file_extensions" in idx:
            cfg.index.file_extensions = list(idx["file_extensions"])
