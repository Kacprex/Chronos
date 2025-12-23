"""Run supervised preprocessing (GM games -> SL shards).

Note: shard location is intentionally hardcoded in src/config.py.
"""
from src.data.preprocess_rust import preprocess

if __name__ == "__main__":
    preprocess()
