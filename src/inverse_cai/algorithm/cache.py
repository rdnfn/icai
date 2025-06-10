import pandas as pd
import time
from pathlib import Path
from filelock import FileLock
from loguru import logger
import json
import hashlib


class VoteCache:
    def __init__(
        self,
        cache_path: Path,
        lock_timeout: int = 10,
        verbose: bool = False,
        max_entries_per_file: int = 10000,
    ):
        """Initialize vote cache with path to jsonl file.

        Args:
            cache_path: Path to jsonl file storing votes (with or without .jsonl suffix)
            lock_timeout: Maximum time to wait for file lock in seconds
            verbose: Whether to print verbose output
            max_entries_per_file: Maximum entries per cache file before rotation
        """
        self.cache_path = Path(cache_path)
        self.cache_dir = self.cache_path.parent
        self.cache_base = self.cache_path.stem
        self.cache_ext = ".jsonl"
        self.lock = FileLock(self.cache_path.with_suffix(".lock"), timeout=lock_timeout)
        self.verbose = verbose
        self.max_entries_per_file = max_entries_per_file

        # In-memory cache of processed hashes for performance
        self._processed_hashes = None

        # Track current file and entry count for rotation
        self._current_file_index = 0
        self._current_file_entries = 0
        self._initialize_file_tracking()

        # Create cache directory and initial file if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        current_cache_path = self._get_cache_file_path(self._current_file_index)
        if not current_cache_path.exists():
            current_cache_path.touch()

    def _get_cache_file_path(self, file_index: int) -> Path:
        """Get path for cache file with given index."""
        return self.cache_dir / f"{self.cache_base}_{file_index}{self.cache_ext}"

    def _initialize_file_tracking(self):
        """Initialize file tracking by finding the current active file."""
        # Find the highest numbered cache file
        file_index = 0
        while self._get_cache_file_path(file_index).exists():
            file_index += 1

        # Use the last existing file or start with 0
        self._current_file_index = max(0, file_index - 1)

        # Count entries in current file
        current_cache_path = self._get_cache_file_path(self._current_file_index)
        if current_cache_path.exists():
            self._current_file_entries = sum(
                1
                for line in open(current_cache_path, "r", encoding="utf-8")
                if line.strip()
            )
        else:
            self._current_file_entries = 0

    def get_processed_hashes(self) -> set:
        """Load set of processed hashes from all cache files (cached in memory)."""
        if self._processed_hashes is None:
            self._processed_hashes = set(self._iter_cache_data(extract_keys_only=True))
        return self._processed_hashes

    def check_if_hash_processed(self, hash: str) -> bool:
        """Check if hash has been processed."""
        return hash in self.get_processed_hashes()

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of hash -> dictionary of votes."""
        with self.lock:
            votes_dict = {}
            for data in self._iter_cache_data():
                votes_dict.update(data)
            return votes_dict

    def _iter_cache_data(self, extract_keys_only: bool = False):
        """Iterate through all cache files and yield data or keys."""
        file_index = 0
        while True:
            cache_file = self._get_cache_file_path(file_index)
            if not cache_file.exists() or cache_file.stat().st_size == 0:
                if not cache_file.exists():
                    break
                file_index += 1
                continue

            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            if extract_keys_only:
                                yield from data.keys()
                            else:
                                yield data
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(
                                f"Skipping malformed line in cache file {cache_file}: {line.strip()}, error: {e}"
                            )
            file_index += 1

    def update_cache(self, hash: str, vote: dict):
        """Update cache with new vote result by appending to file."""
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

                    # Check if we need to rotate to a new file
                    if self._current_file_entries >= self.max_entries_per_file:
                        self._current_file_index += 1
                        self._current_file_entries = 0
                        if self.verbose:
                            print(f"Rotating to cache file {self._current_file_index}")

                    # Append new vote as JSON line to current file
                    current_cache_path = self._get_cache_file_path(
                        self._current_file_index
                    )
                    with open(current_cache_path, "a", encoding="utf-8") as f:
                        json.dump({hash: vote}, f)
                        f.write("\n")

                    # Update in-memory cache
                    if self._processed_hashes is not None:
                        self._processed_hashes.add(hash)

                    self._current_file_entries += 1
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
