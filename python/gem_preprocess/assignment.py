"""Original TF-IDF assignment over coarse codes, with explicit inputs."""
import numpy as np


def initial_membership(codes, lengths, labels, clusters):
    labels = np.asarray(labels).reshape(-1)
    codes = np.asarray(codes).reshape(-1)
    lengths = np.asarray(lengths).reshape(-1)
    if len(lengths) == 0 or np.any(lengths <= 0) or int(lengths.sum()) != len(codes):
        raise ValueError("invalid document lengths")
    if codes.size == 0 or codes.min() < 0 or codes.max() >= len(labels):
        raise ValueError("fine code out of range")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= clusters:
        raise ValueError("coarse label out of range")
    mapped = labels[codes]
    postings = [[] for _ in range(clusters)]
    offset = 0
    for doc, length in enumerate(lengths):
        end = offset + int(length)
        for cluster in np.unique(mapped[offset:end]):
            postings[int(cluster)].append(doc)
        offset = end
    return mapped, postings


def tfidf_membership(mapped, lengths, postings, top_r):
    if top_r <= 0:
        raise ValueError("top_r must be positive")
    result = [[] for _ in postings]
    counts = np.array([len(p) for p in postings])
    offset = 0
    for doc, length in enumerate(lengths):
        end = offset + int(length)
        unique, frequency = np.unique(mapped[offset:end], return_counts=True)
        tf = frequency / len(mapped[offset:end])
        idf = np.array([np.log(len(lengths) / counts[c]) for c in unique])
        selected = unique[np.argsort(-(tf * idf))[:top_r]]
        for cluster in selected:
            result[int(cluster)].append(doc)
        offset = end
    return [sorted(documents, reverse=True) for documents in result]
