from __future__ import annotations

import numpy as np


def cluster_documents(
    doc_embeddings: dict[str, np.ndarray],
    min_cluster_size: int = 3,
    umap_components: int = 15,
    umap_neighbors: int = 15,
) -> dict[str, int]:
    """
    Returns {path: cluster_label} where label -1 means noise/unclustered.
    Requires umap-learn and scikit-learn >= 1.3.
    """
    if len(doc_embeddings) < 2:
        return {p: 0 for p in doc_embeddings}

    try:
        import umap as umap_module
    except ImportError:
        raise ImportError(
            "umap-learn is required for clustering. Install it with: pip install umap-learn"
        )

    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        raise ImportError(
            "scikit-learn >= 1.3 is required for HDBSCAN. "
            "Install with: pip install 'scikit-learn>=1.3'"
        )

    from sklearn.preprocessing import normalize

    paths = list(doc_embeddings.keys())
    X = np.stack([doc_embeddings[p] for p in paths]).astype(np.float64)
    X = normalize(X)

    # Reduce dimensions with UMAP before clustering
    n_components = min(umap_components, len(paths) - 1)
    n_neighbors = min(umap_neighbors, len(paths) - 1)

    reducer = umap_module.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    X_reduced = reducer.fit_transform(X)

    clusterer = HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, len(paths) // 5)),
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_reduced)

    return dict(zip(paths, labels.tolist()))


def group_by_cluster(labels: dict[str, int]) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for path, label in labels.items():
        groups.setdefault(label, []).append(path)
    return {k: sorted(v) for k, v in sorted(groups.items())}
