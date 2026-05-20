from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="tinygrep",
    help="Semantic search and clustering for local markdown folders.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_folder(folder: Optional[Path]) -> Path:
    p = (folder or Path(".")).expanduser().resolve()
    if not p.is_dir():
        err_console.print(f"[red]Error:[/red] '{p}' is not a directory.")
        raise typer.Exit(1)
    return p


def _db_path(folder: Path) -> Path:
    return folder / ".tinygrep" / "index.db"


def _load_provider(folder: Path, provider_name: Optional[str], model_name: Optional[str]):
    from .config import Config
    from .providers import get_provider

    cfg = Config.load(folder)
    if provider_name:
        cfg.provider.name = provider_name
    if model_name:
        cfg.provider.model = model_name

    api_key = cfg.api_key()
    resolved_model = cfg.provider.resolved_model()
    provider = get_provider(cfg.provider.name, resolved_model, api_key)
    return provider, cfg


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@app.command()
def index(
    folder: Annotated[Optional[Path], typer.Argument(help="Folder to index (default: current dir)")] = None,
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

    folder_path = _resolve_folder(folder)
    cfg = Config.load(folder_path)
    if provider:
        cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    api_key = cfg.api_key()
    resolved_model = cfg.provider.resolved_model()
    emb_provider = get_provider(cfg.provider.name, resolved_model, api_key)

    db_file = _db_path(folder_path)
    with Database(db_file) as db:
        stats = index_folder(
            folder=folder_path,
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
    folder: Annotated[Optional[Path], typer.Option("--folder", "-d", help="Indexed folder (default: current dir)")] = None,
    top_n: Annotated[int, typer.Option("--top-n", "-n", help="Number of results")] = 10,
    output_format: Annotated[str, typer.Option("--format", "-o", help="Output format: text, json")] = "text",
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
    full: Annotated[bool, typer.Option("--full", help="Show full chunk text instead of snippet")] = False,
):
    """Search indexed documents using a natural-language query."""
    from .db import Database
    from .search import search as do_search

    folder_path = _resolve_folder(folder)
    emb_provider, _ = _load_provider(folder_path, provider, model)

    db_file = _db_path(folder_path)
    if not db_file.exists():
        err_console.print("[red]Error:[/red] No index found. Run [bold]tinygrep index[/bold] first.")
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
    folder: Annotated[Optional[Path], typer.Argument(help="Indexed folder (default: current dir)")] = None,
    min_cluster_size: Annotated[int, typer.Option("--min-size", "-s", help="Min files per cluster")] = 3,
    output_format: Annotated[str, typer.Option("--format", "-o", help="Output format: text, json")] = "text",
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
):
    """Cluster indexed documents by semantic similarity (UMAP + HDBSCAN)."""
    from .cluster import cluster_documents, group_by_cluster
    from .db import Database

    folder_path = _resolve_folder(folder)
    emb_provider, _ = _load_provider(folder_path, provider, model)

    db_file = _db_path(folder_path)
    if not db_file.exists():
        err_console.print("[red]Error:[/red] No index found. Run [bold]tinygrep index[/bold] first.")
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
        output = {
            str(label): files
            for label, files in groups.items()
        }
        # Rename -1 to "noise"
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
    folder: Annotated[Optional[Path], typer.Argument(help="Indexed folder (default: current dir)")] = None,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
):
    """Show indexing status: which files are up-to-date, stale, or missing."""
    from .config import Config
    from .db import Database
    from .indexer import sha256, walk_folder

    folder_path = _resolve_folder(folder)
    cfg = Config.load(folder_path)
    if provider:
        cfg.provider.name = provider
    if model:
        cfg.provider.model = model

    db_file = _db_path(folder_path)
    if not db_file.exists():
        console.print("[yellow]No index found.[/yellow] Run [bold]tinygrep index[/bold] to get started.")
        raise typer.Exit(0)

    with Database(db_file) as db:
        indexed_docs = {d["path"]: d for d in db.get_all_documents()}

    disk_files = walk_folder(folder_path, cfg.index.file_extensions)
    disk_rel = {str(f.relative_to(folder_path)): f for f in disk_files}

    up_to_date: list[str] = []
    stale: list[str] = []
    unindexed: list[str] = []
    deleted: list[str] = []

    for rel, path in disk_rel.items():
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
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

    # Summary
    total = len(disk_rel)
    console.print(f"\nFolder: [bold]{folder_path}[/bold]")
    console.print(f"Index:  [bold]{db_file}[/bold]\n")
    console.print(f"  [green]✓[/green] Up-to-date:  {len(up_to_date)}")
    console.print(f"  [yellow]~[/yellow] Stale:       {len(stale)}")
    console.print(f"  [blue]+[/blue] Unindexed:   {len(unindexed)}")
    console.print(f"  [red]✗[/red] Deleted:     {len(deleted)}")
    console.print(f"    Total files: {total}\n")

    if stale:
        console.print("[yellow]Stale files (content changed):[/yellow]")
        for f in stale[:20]:
            console.print(f"  {f}")
        if len(stale) > 20:
            console.print(f"  ... and {len(stale) - 20} more")
        console.print()

    if unindexed:
        console.print("[blue]Unindexed files:[/blue]")
        for f in unindexed[:20]:
            console.print(f"  {f}")
        if len(unindexed) > 20:
            console.print(f"  ... and {len(unindexed) - 20} more")
        console.print()

    if deleted:
        console.print("[red]In index but no longer on disk:[/red]")
        for f in deleted[:20]:
            console.print(f"  {f}")
        console.print()

    if stale or unindexed:
        console.print("Run [bold]tinygrep index[/bold] to update the index.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()
