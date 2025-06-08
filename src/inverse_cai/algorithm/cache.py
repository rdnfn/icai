import pandas as pd
import time
from pathlib import Path
from filelock import FileLock
from loguru import logger
import json
import hashlib


class VoteCache:
    def __init__(self, cache_path: Path, lock_timeout: int = 10, verbose: bool = False):
        """Initialize vote cache with path to jsonl file.

        Args:
            cache_path: Path to jsonl file storing votes
            lock_timeout: Maximum time to wait for file lock in seconds
        """
        self.cache_path = Path(cache_path)
        self.hash_path = self.cache_path.with_suffix(".hash.json")
        self.lock_path = self.cache_path.with_suffix(".lock")
        self.lock_timeout = lock_timeout
        self.lock = FileLock(self.lock_path, timeout=lock_timeout)
        self.verbose = verbose

        # Initialize empty cache file if it doesn't exist
        if not self.cache_path.exists():
            self.cache_path.touch()

        if not self.hash_path.exists():
            with open(self.hash_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def get_processed_hashes(self) -> set:
        """Load set of processed hashes."""
        with open(self.hash_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    def check_if_hash_processed(self, hash: str) -> bool:
        """Check if hash has been processed."""
        processed = self.get_processed_hashes()
        return hash in processed

    def _add_processed_hash(self, hash: str):
        """Add hash to processed set."""
        processed = self.get_processed_hashes()
        processed.add(hash)
        with open(self.hash_path, "w", encoding="utf-8") as f:
            json.dump(list(processed), f)

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of hash -> dictionary of votes."""
        with self.lock:
            votes_dict = {}
            if self.cache_path.stat().st_size == 0:
                return votes_dict

            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            hash_key = data["hash"]
                            vote = data["vote"]
                            votes_dict[hash_key] = vote
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(
                                f"Skipping malformed line in cache: {line.strip()}, error: {e}"
                            )
        return votes_dict

    def update_cache(self, hash: str, vote: dict):
        """Update cache with new vote result by appending to file.

        Will retry if lock is not available.
        """
        max_retries = 10
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with self.lock:
                    if self.check_if_hash_processed(hash):
                        if self.verbose:
                            print(
                                f"Cache warning: hash {hash} already processed. Cache not updated."
                            )
                        return

                    # Append new vote as JSON line
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        json_line = json.dumps({"hash": hash, "vote": vote})
                        f.write(json_line + "\n")

                    self._add_processed_hash(hash)
                return
            except TimeoutError as exc:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Lock acquisition failed, retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    raise TimeoutError(
                        f"Failed to acquire lock after {max_retries} retries"
                    ) from exc


def _get_hash(dict_to_hash: dict) -> str:
    string = json.dumps(dict_to_hash, sort_keys=True)
    return hashlib.md5(string.encode("utf-8")).hexdigest()[:8]


def get_vote_hash(
    preferred: str, rejected: str, principle: str, model_name: str
) -> str:
    return _get_hash(
        dict(
            preferred=preferred,
            rejected=rejected,
            principle=principle,
            model_name=model_name,
        )
    )
