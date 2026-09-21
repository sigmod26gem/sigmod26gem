"""Settings for the original GEM preprocessing stages."""
import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    codes: str
    lengths: str
    fine_centroids: str
    output: str
    shards: int = 1
    graph_clusters: int = 1024
    top_r: int = 20
    iterations: int = 1000
    gpu: bool = True


def read_config(filename):
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    with open(filename) as source:
        parser.read_file(source)
    if parser.defaults() or parser.sections() != ["encoding"]:
        raise ValueError("expected a single [encoding] section")
    fields = set(Config.__dataclass_fields__)
    section = parser["encoding"]
    unknown = set(section) - fields
    if unknown:
        raise ValueError(f"unknown encoding fields: {sorted(unknown)}")
    values = {}
    for field in ("codes", "lengths", "fine_centroids", "output"):
        if not section.get(field, "").strip():
            raise ValueError(f"encoding.{field} is required")
        values[field] = str((Path(filename).resolve().parent / section[field]).resolve())
    for field in ("shards", "graph_clusters", "top_r", "iterations"):
        values[field] = section.getint(field, getattr(Config, field))
        if values[field] <= 0:
            raise ValueError(f"encoding.{field} must be positive")
    values["gpu"] = section.getboolean("gpu", True)
    if values["shards"] > 1 and any("{shard}" not in values[k] for k in ("codes", "lengths")):
        raise ValueError("multi-shard paths require {shard}")
    return Config(**values)
