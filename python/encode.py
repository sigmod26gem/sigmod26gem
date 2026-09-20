"""Usage: python3 python/encode.py configs/encoding.ini"""
import sys
from gem_preprocess.config import read_config
from gem_preprocess.pipeline import run


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: encode.py config.ini")
    run(read_config(sys.argv[1]))
