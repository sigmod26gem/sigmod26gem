"""Compare upstream and library search using the same EVQA inputs and graph."""
import configparser
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    repo = Path(__file__).resolve().parents[1]
    build, data, output = map(Path, sys.argv[1:4])
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    output.mkdir(parents=True, exist_ok=True)
    config = configparser.ConfigParser(interpolation=None)
    config.read(repo / "configs/evqa.ini")
    for key in ("vectors", "lengths", "codes", "fine_centroids", "graph_centroids", "clusters"):
        relative = config["data"][key].split("gem_data/evqa/")[1]
        config["data"][key] = str(data / relative)
    for key in ("vectors", "lengths", "qrels"):
        relative = config["query"][key].split("gem_data/evqa/")[1]
        config["query"][key] = str(data / relative)
    config["index"]["path"] = (sys.argv[5] if len(sys.argv) > 5 else
                                str(repo / "example_index/evqaIndex1024_all_24_80/0.bin"))
    config["query"]["limit"] = str(limit)
    reports = {}
    for name, executable, workers in (
        ("reference", "gem_upstream_reference", 1),
        ("library_w1", "gem_run", 1),
        ("library_w8", "gem_run", 8),
    ):
        config["runtime"]["workers"] = str(workers)
        config["benchmark"]["output"] = str(output / f"{name}.tsv")
        cfg_path = output / f"{name}.ini"
        with cfg_path.open("w") as file:
            config.write(file)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run([str(build / executable), str(cfg_path)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        reports[name] = [json.loads(line) for line in (output / f"{name}.log").read_text().splitlines()
                         if line.startswith('{"')]
        print(name, json.dumps(reports[name]), flush=True)
    with (output / "reference.tsv").open() as file:
        reference = list(csv.reader(file, delimiter="\t"))
    for name in ("library_w1", "library_w8"):
        with (output / f"{name}.tsv").open() as file:
            actual = list(csv.reader(file, delimiter="\t"))
        if len(actual) != len(reference):
            raise RuntimeError(f"{name}: result count differs")
        max_error = 0.0
        for a, b in zip(actual, reference):
            if a[:3] != b[:3]:
                raise RuntimeError(f"{name}: query/rank/doc mismatch: {a} vs {b}")
            max_error = max(max_error, abs(float(a[3]) - float(b[3])))
        if max_error > 1e-6:
            raise RuntimeError(f"{name}: score error {max_error}")
        reports[name + "_equivalence"] = {"rows": len(actual), "max_score_error": max_error}
    (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("Equivalent document IDs/ranks; scores within 1e-6.", flush=True)


if __name__ == "__main__":
    main()
