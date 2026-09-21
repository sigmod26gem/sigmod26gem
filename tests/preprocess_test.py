import sys
import configparser
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from gem_preprocess.assignment import initial_membership, tfidf_membership
from gem_preprocess.config import read_config


class AssignmentTests(unittest.TestCase):
    def test_original_formula(self):
        rng = np.random.RandomState(17)
        lengths = rng.randint(1, 30, size=128)
        codes = rng.randint(0, 24, size=int(lengths.sum()))
        labels = rng.randint(0, 7, size=(24, 1))
        mapped, initial = initial_membership(codes, lengths, labels, 8)
        self.assertEqual(initial[-1], [])
        for r in (1, 3, 20):
            expected = [[] for _ in range(8)]
            offset = 0
            for doc, n in enumerate(lengths):
                token_labels = labels[codes[offset:offset + n]].squeeze().reshape(-1)
                unique, counts = np.unique(token_labels, return_counts=True)
                tf = counts / len(token_labels)
                idf = np.array([np.log(len(lengths) / len(initial[w])) for w in unique])
                for cluster in unique[np.argsort(-(tf * idf))[:r]]:
                    expected[cluster].append(doc)
                offset += n
            expected = [sorted(d, reverse=True) for d in expected]
            self.assertEqual(tfidf_membership(mapped, lengths, initial, r), expected)

    def test_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoding.ini"
            content = "[encoding]\ncodes=c.npy\nlengths=l.npy\nfine_centroids=f.npy\noutput=out\n"
            path.write_text(content)
            config = read_config(path)
            self.assertEqual(config.codes, str(Path(directory) / "c.npy"))
            self.assertEqual(config.top_r, 20)
            for suffix in ("typo=1\n", "top_r=0\n", "shards=2\n", "shards=1\nshards=2\n"):
                path.write_text(content + suffix)
                with self.assertRaises((ValueError, configparser.Error)):
                    read_config(path)


if __name__ == "__main__":
    unittest.main()
