"""Train coarse centroids from fine centroids and assign documents."""
from pathlib import Path
import numpy as np
from .assignment import initial_membership, tfidf_membership


def write_clusters(path, clusters):
    with path.open("w") as output:
        for documents in clusters:
            output.write(" ".join(map(str, documents)) + "\n")


def run(config):
    import faiss

    codes, lengths = [], []
    for shard in range(config.shards):
        c = np.load(config.codes.replace("{shard}", str(shard)), allow_pickle=False)
        n = np.load(config.lengths.replace("{shard}", str(shard)), allow_pickle=False)
        if c.ndim != 1 or n.ndim != 1 or c.dtype.kind not in "iu" or n.dtype.kind not in "iu":
            raise ValueError("codes and lengths must be one-dimensional integer arrays")
        if np.any(n <= 0) or int(n.sum()) != len(c):
            raise ValueError("shard token/code counts differ")
        codes.append(c)
        lengths.append(n)
    codes, lengths = np.concatenate(codes), np.concatenate(lengths)
    centers = np.load(config.fine_centroids, allow_pickle=False).astype(np.float32)
    if centers.ndim != 2 or not np.isfinite(centers).all() or not centers.shape[1]:
        raise ValueError("invalid fine centroid matrix")
    if config.graph_clusters > len(centers):
        raise ValueError("graph_clusters exceeds fine centroid count")
    if codes.size == 0 or codes.min() < 0 or codes.max() >= len(centers):
        raise ValueError("fine code out of range")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    names = ("coarse_centroids.npy", "coarse_cluster_labels.npy", "init_cluster_info.txt", "coarse_cluster_info.txt")
    if any((output / name).exists() for name in names):
        raise FileExistsError("encoding output already exists")
    trainer = faiss.Kmeans(centers.shape[1], config.graph_clusters, niter=config.iterations,
                           gpu=config.gpu, spherical=True, verbose=True)
    trainer.train(centers)
    _, labels = trainer.index.search(centers, 1)
    mapped, initial = initial_membership(codes, lengths, labels, config.graph_clusters)
    assigned = tfidf_membership(mapped, lengths, initial, config.top_r)
    np.save(output / "coarse_centroids.npy", trainer.centroids)
    np.save(output / "coarse_cluster_labels.npy", labels)
    write_clusters(output / "init_cluster_info.txt", initial)
    write_clusters(output / "coarse_cluster_info.txt", assigned)
