# tinygrep

Semantic search and topic clustering for local folders of markdown notes and articles.

`grep` finds exact strings. `tinygrep` finds *meaning* — surface articles about continuous flow matching even if they never use that phrase, or automatically discover that your notes cluster around three or four recurring themes.

## How it works

1. **Index** — walks your folder, splits each file into overlapping text chunks, and calls an embedding API to turn each chunk into a vector. Vectors are stored in a local SQLite database (`.tinygrep/index.db` inside your folder). Files that haven't changed are skipped on subsequent runs.

2. **Search** — embeds your query with the same model, computes cosine similarity against every stored chunk, and returns the top-N matches with file path, score, and a text snippet.

3. **Cluster** — mean-pools each document's chunk embeddings into a single vector, runs UMAP to reduce dimensionality, then HDBSCAN to find natural topic groups — no need to specify how many clusters you want.

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/you/tinygrep
cd tinygrep
pip install -e .
```

## Quickstart

```bash
# Set your API key (Voyage AI is the default provider)
export VOYAGE_API_KEY=your_key_here

# Index a folder (stores embeddings in ~/notes/.tinygrep/index.db)
tinygrep index ~/notes

# Search
tinygrep search "continuous diffusion models in language" --folder ~/notes

# See what topics your notes cluster around
tinygrep cluster ~/notes

# Check which files are stale or unindexed
tinygrep status ~/notes
```

## Commands

### `tinygrep index [FOLDER]`

Walk the folder, chunk each file, compute embeddings, and store them. Re-running is safe — only changed or new files are re-embedded.

```
Options:
  -p, --provider TEXT   Embedding provider: voyage, jina  [default: voyage]
  -m, --model TEXT      Model name  [default: voyage-4-large]
  -f, --force           Re-embed all files even if unchanged
  -q, --quiet           Suppress progress output
```

### `tinygrep search QUERY`

Search with a natural-language query.

```
Options:
  -d, --folder PATH     Folder to search  [default: current dir]
  -n, --top-n INT       Number of results  [default: 10]
  -o, --format TEXT     Output format: text, json  [default: text]
      --full            Show full chunk text instead of a snippet
  -p, --provider TEXT
  -m, --model TEXT
```

**Example — JSON output for scripting:**
```bash
tinygrep search "flow matching vs score matching" \
  --folder ~/notes --top-n 5 --format json
```

```json
[
  {
    "rank": 1,
    "file": "papers/flow-matching-2023.md",
    "chunk_index": 2,
    "score": 0.9241,
    "snippet": "Flow matching directly learns the vector field that transports..."
  }
]
```

### `tinygrep cluster [FOLDER]`

Group documents by semantic similarity using UMAP + HDBSCAN. No need to pick a number of clusters.

```
Options:
  -s, --min-size INT    Minimum files per cluster  [default: 3]
  -o, --format TEXT     Output format: text, json  [default: text]
  -p, --provider TEXT
  -m, --model TEXT
```

### `tinygrep status [FOLDER]`

Show which files are up-to-date, stale (content changed), unindexed, or deleted.

## Configuration

Create `.tinygrep/config.toml` inside your folder (or `~/.config/tinygrep/config.toml` for a global default):

```toml
[provider]
name  = "voyage"
# voyage-4-large, voyage-4, voyage-4-lite, voyage-4-nano, voyage-code-3
# jina-embeddings-v3
model = "voyage-4-large"

[index]
chunk_size    = 2000   # characters per chunk (~500 tokens)
chunk_overlap = 200    # overlap between consecutive chunks
file_extensions = [".md", ".txt", ".mdx"]
```

See [`config.toml.example`](config.toml.example) for a full annotated example.

### API keys

| Provider | Environment variable |
|----------|----------------------|
| Voyage AI | `VOYAGE_API_KEY` |
| Jina AI  | `JINA_API_KEY`   |

You can also set `TINYGREP_PROVIDER` and `TINYGREP_MODEL` to override the config without editing files.

## Supported embedding providers

| Provider | Models | Notes |
|----------|--------|-------|
| [Voyage AI](https://docs.voyageai.com) | `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-3`, … | Default |
| [Jina AI](https://jina.ai/embeddings/) | `jina-embeddings-v3` | |

### Adding a new provider

1. Create `tinygrep/providers/yourprovider.py` and subclass `EmbeddingProvider`.
2. Add a `case "yourprovider"` branch to `get_provider()` in `tinygrep/providers/__init__.py`.

```python
# tinygrep/providers/myprovider.py
from .base import EmbeddingProvider

class MyProvider(EmbeddingProvider):
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        ...

    @property
    def model_id(self) -> str: return self._model

    @property
    def provider_name(self) -> str: return "myprovider"
```

## Notes on clustering quality

HDBSCAN auto-detects the number of clusters and marks outliers as "noise" (label `-1`). A few things that affect quality:

- **Model choice matters** — `voyage-4-large` will capture domain-specific relationships much better than smaller/generic models for technical note collections.
- **Corpus size** — clustering works best with 20+ documents. With very small collections HDBSCAN may label most docs as noise; try `--min-size 2`.
- **Chunk size** — shorter chunks improve search precision but mean-pooling over many chunks per document can dilute cluster signal for long articles.

## Project layout

```
tinygrep/
├── tinygrep/
│   ├── cli.py          # Typer commands
│   ├── config.py       # Config loading (TOML + env vars)
│   ├── db.py           # SQLite storage layer
│   ├── indexer.py      # File walking, hashing, chunking
│   ├── search.py       # Cosine similarity search
│   ├── cluster.py      # UMAP + HDBSCAN clustering
│   └── providers/
│       ├── base.py     # Abstract EmbeddingProvider
│       ├── voyage.py
│       └── jina.py
└── pyproject.toml
```
