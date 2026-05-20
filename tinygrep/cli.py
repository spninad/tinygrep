from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tinygrep",
    help="Semantic search and clustering for local markdown folders.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Root discovery  (mirrors how git finds .git/)
# ---------------------------------------------------------------------------

def _find_root(start: Path) -> Optional[Path]:
    """Walk up the directory tree from start, return the first dir that contains .tinygrep/."""
    current = start.resolve()
    while True:
        if (current / ".tinygrep").is_dir():
            return current
        parent = current.parent
        if parent == current:   # reached filesystem root
            return None
        current = parent


def _resolve_root(path: Optional[Path]) -> Path:
    """
    Resolve the tinygrep root by walking up from path (or CWD).
    Exits with an error if no .tinygrep/ directory is found.
    """
    start = (path or Path(".")).expanduser().resolve()
    if path and not start.is_dir():
        err_console.print(f"[red]Error:[/red] '{start}' is not a directory.")
        raise typer.Exit(1)
    root = _find_root(start)
    if root is None:
        err_console.print(
            "[red]Error:[/red] No tinygrep root found. "
            "Run [bold]tinygrep init[/bold] in the folder you want to index."
        )
        raise typer.Exit(1)
    return root


def _resolve_dir(path: Optional[Path]) -> Path:
    """Plain directory resolver used only by init (no root walking)."""
    p = (path or Path(".")).expanduser().resolve()
    if not p.is_dir():
        err_console.print(f"[red]Error:[/red] '{p}' is not a directory.")
        raise typer.Exit(1)
    return p


def _db_path(root: Path) -> Path:
    return root / ".tinygrep" / "index.db"


def _load_provider(root: Path, provider_name: Optional[str], model_name: Optional[str]):
    from .config import Config
    from .providers import get_provider

    cfg = Config.load(root)
    if provider_name:
        cfg.provider.name = provider_name
    if model_name:
        cfg.provider.model = model_name

    api_key = cfg.api_key()
    resolved_model = cfg.provider.resolved_model()
    provider = get_provider(cfg.provider.name, resolved_model, api_key)
    return provider, cfg


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    folder: Annotated[Optional[Path], typer.Argument(help="Folder to initialize (default: current dir)")] = None,
):
    """Create a .tinygrep/ directory to mark the root of an index."""
    folder_path = _resolve_dir(folder)
    tg_dir = folder_path / ".tinygrep"

    if tg_dir.exists():
        console.print(f"[yellow]Already initialized:[/yellow] {tg_dir}")
        raise typer.Exit(0)

    tg_dir.mkdir(parents=True)
    console.print(f"[green]Initialized[/green] {tg_dir}")
    console.print(
        f"\nRun [bold]tinygrep index[/bold] from anywhere inside "
        f"[bold]{folder_path}[/bold] to start indexing."
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index(
    path: Annotated[Optional[Path], typer.Argument(help="Root or any subdirectory (default: current dir)")] = None,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p", help="Embedding provider: voyage, jina")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model name")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-embed all files even if unchanged")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
):
    """Index a folder: compute and store embeddings for all markdown files."""
    from .config import Config
    from .db import Database
    from .indexer import index_folder
    from .providers import get_provider

    root = _resolve_root(path)
    cfg = Config.load(root)
    if provider:
        cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    if not quiet:
        console.print(f"Root: [bold]{root}[/bold]")

    api_key = cfg.api_key()
    resolved_model = cfg.provider.resolved_model()
    emb_provider = get_provider(cfg.provider.name, resolved_model, api_key)

    db_file = _db_path(root)
    with Database(db_file) as db:
        stats = index_folder(
            folder=root,
            db=db,
            provider=emb_provider,
            cfg=cfg,
            force=force,
            verbose=not quiet,
        )

    if not quiet:
        console.print(
            f"\n[green]Done.[/green] "
            f"Indexed [bold]{stats['indexed']}[/bold], "
            f"skipped [bold]{stats['skipped']}[/bold] (unchanged), "
            f"failed [bold]{stats['failed']}[/bold] "
            f"out of [bold]{stats['total']}[/bold] files."
        )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    path: Annotated[Optional[Path], typer.Option("--folder", "-d", help="Root or any subdirectory (default: current dir)")] = None,
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of results")] = 10,
    output_format: Annotated[str, typer.Option("--format", "-o", help="Output format: text, json")] = "text",
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    full: Annotated[bool, typer.Option("--full", help="Show full chunk text instead of snippet")] = False,
):
    """Search indexed documents using a natural-language query."""
    from .db import Database
    from .search import search as do_search

    root = _resolve_root(path)
    emb_provider, _ = _load_provider(root, provider, model)

    db_file = _db_path(root)
    if not db_file.exists():
        err_console.print("[red]Error:[/red] Index is empty. Run [bold]tinygrep index[/bold] first.")
        raise typer.Exit(1)

    with Database(db_file) as db:
        results = do_search(query, db, emb_provider, top_k=top_n)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    if output_format == "json":
        output = [
            {
                "rank": i + 1,
                "file": r["file"],
                "chunk_index": r["chunk_index"],
                "score": round(r["score"], 4),
                "snippet": r["text"] if full else r["snippet"],
            }
            for i, r in enumerate(results)
        ]
        print(json.dumps(output, indent=2))
    else:
        _print_search_results(results, full=full)


def _print_search_results(results: list[dict], full: bool = False) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", width=7, justify="right")
    table.add_column("File", style="bold", overflow="fold")
    table.add_column("Snippet", overflow="fold")

    for i, r in enumerate(results, 1):
        score_str = f"{r['score']:.3f}"
        score_color = "green" if r["score"] > 0.8 else "yellow" if r["score"] > 0.6 else "red"
        text = r["text"] if full else r["snippet"]
        table.add_row(
            str(i),
            f"[{score_color}]{score_str}[/{score_color}]",
            r["file"],
            text,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

@app.command()
def cluster(
    path: Annotated[Optional[Path], typer.Argument(help="Root or any subdirectory (default: current dir)")] = None,
    min_cluster_size: Annotated[int, typer.Option("--min-size", "-s", help="Min files per cluster")] = 3,
    output_format: Annotated[str, typer.Option("--format", "-o", help="Output format: text, json")] = "text",
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
):
    """Cluster indexed documents by semantic similarity (UMAP + HDBSCAN)."""
    from .cluster import cluster_documents, group_by_cluster
    from .db import Database

    root = _resolve_root(path)
    emb_provider, _ = _load_provider(root, provider, model)

    db_file = _db_path(root)
    if not db_file.exists():
        err_console.print("[red]Error:[/red] Index is empty. Run [bold]tinygrep index[/bold] first.")
        raise typer.Exit(1)

    with Database(db_file) as db:
        doc_embs = db.get_document_embeddings(emb_provider.provider_name, emb_provider.model_id)

    if not doc_embs:
        console.print("[yellow]No indexed documents found for this provider/model.[/yellow]")
        raise typer.Exit(0)

    if len(doc_embs) < 2:
        console.print("[yellow]Need at least 2 documents to cluster.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Clustering [bold]{len(doc_embs)}[/bold] documents...", highlight=False)

    with console.status("Running UMAP + HDBSCAN..."):
        labels = cluster_documents(doc_embs, min_cluster_size=min_cluster_size)

    groups = group_by_cluster(labels)

    if output_format == "json":
        output = {str(label): files for label, files in groups.items()}
        if "-1" in output:
            output["noise"] = output.pop("-1")
        print(json.dumps(output, indent=2))
    else:
        _print_clusters(groups)


def _print_clusters(groups: dict[int, list[str]]) -> None:
    noise = groups.pop(-1, [])
    total_clusters = len(groups)

    console.print(f"\nFound [bold green]{total_clusters}[/bold green] cluster(s):\n")

    for label, files in sorted(groups.items()):
        console.print(f"[bold cyan]Cluster {label}[/bold cyan] ({len(files)} files)")
        for f in files:
            console.print(f"  [dim]•[/dim] {f}")
        console.print()

    if noise:
        console.print(f"[bold yellow]Unclustered / noise[/bold yellow] ({len(noise)} files)")
        for f in noise:
            console.print(f"  [dim]•[/dim] {f}")
        console.print()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@app.command()
def status(
    path: Annotated[Optional[Path], typer.Argument(help="Root or any subdirectory (default: current dir)")] = None,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
):
    """Show indexing status: which files are up-to-date, stale, or missing."""
    from .config import Config
    from .db import Database
    from .indexer import sha256, walk_folder

    root = _resolve_root(path)
    cfg = Config.load(root)
    if provider:
        cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    db_file = _db_path(root)
    if not db_file.exists():
        console.print(
            f"Root: [bold]{root}[/bold]\n"
            "[yellow]No index yet.[/yellow] Run [bold]tinygrep index[/bold] to build it."
        )
        raise typer.Exit(0)

    with Database(db_file) as db:
        indexed_docs = {d["path"]: d for d in db.get_all_documents()}

    disk_files = walk_folder(root, cfg.index.file_extensions)
    disk_rel = {str(f.relative_to(root)): f for f in disk_files}

    up_to_date: list[str] = []
    stale: list[str] = []
    unindexed: list[str] = []
    deleted: list[str] = []

    for rel, fpath in disk_rel.items():
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        current_hash = sha256(content)
        doc = indexed_docs.get(rel)
        if doc is None:
            unindexed.append(rel)
        elif doc["hash"] != current_hash:
            stale.append(rel)
        else:
            up_to_date.append(rel)

    for rel in indexed_docs:
        if rel not in disk_rel:
            deleted.append(rel)

    console.print(f"\nRoot:  [bold]{root}[/bold]")
    console.print(f"Index: [bold]{db_file}[/bold]\n")
    console.print(f"  [green]✓[/green] Up-to-date:  {len(up_to_date)}")
    console.print(f"  [yellow]~[/yellow] Stale:       {len(stale)}")
    console.print(f"  [blue]+[/blue] Unindexed:   {len(unindexed)}")
    console.print(f"  [red]✗[/red] Deleted:     {len(deleted)}")
    console.print(f"    Total files: {len(disk_rel)}\n")

    if stale:
        console.print("[yellow]Stale files (content changed):[/yellow]")
        for f in stale[:20]:
            console.print(f"  {f}")
        if len(stale) > 20:
            console.print(f"  … and {len(stale) - 20} more")
        console.print()

    if unindexed:
        console.print("[blue]Unindexed files:[/blue]")
        for f in unindexed[:20]:
            console.print(f"  {f}")
        if len(unindexed) > 20:
            console.print(f"  … and {len(unindexed) - 20} more")
        console.print()

    if deleted:
        console.print("[red]In index but no longer on disk:[/red]")
        for f in deleted[:20]:
            console.print(f"  {f}")
        console.print()

    if stale or unindexed:
        console.print("Run [bold]tinygrep index[/bold] to update.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
