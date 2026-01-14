"""
GAIA dataset loader.

This module handles loading the GAIA benchmark dataset from HuggingFace,
including downloading and caching auxiliary files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class GAIADataLoader:
    """
    Loader for the GAIA benchmark dataset.

    The GAIA dataset is hosted on HuggingFace and requires authentication
    to access. It includes auxiliary files (PDFs, images, etc.) that are
    downloaded alongside the main dataset.
    """

    DATASET_ID = "gaia-benchmark/GAIA"
    CONFIGS = [
        "2023_all",
        "2023_level1",
        "2023_level2",
        "2023_level3",
    ]

    def __init__(
        self,
        hf_token: str | None = None,
        data_dir: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        """
        Initialize the GAIA data loader.

        Args:
            hf_token: HuggingFace API token (or set HF_TOKEN env var)
            data_dir: Optional local directory with pre-downloaded data
            cache_dir: Directory for caching downloaded files
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets and huggingface_hub packages not installed. "
                "Install with: pip install datasets huggingface-hub"
            )

        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.data_dir = data_dir
        self.cache_dir = cache_dir or Path.home() / ".cache" / "ai_eval_harness" / "gaia"
        self._dataset_path: Path | None = None

    async def load(
        self,
        split: str = "validation",
        subset: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Load GAIA tasks.

        Args:
            split: Data split ("validation" or "test")
            subset: Optional subset (e.g., "2023_level1")

        Returns:
            List of task dictionaries
        """
        # Determine config name
        config = subset or "2023_all"
        if config not in self.CONFIGS:
            logger.warning(f"Unknown config '{config}', using '2023_all'")
            config = "2023_all"

        # Download dataset if needed
        if self.data_dir:
            self._dataset_path = self.data_dir
        else:
            await self._download_dataset()

        # Load dataset
        logger.info(f"Loading GAIA dataset: config={config}, split={split}")

        try:
            dataset = load_dataset(
                str(self._dataset_path),
                config,
                split=split,
                token=self.hf_token,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Convert to list of dicts
        tasks = []
        for item in dataset:
            task = {
                "task_id": item.get("task_id", ""),
                "Question": item.get("Question", ""),
                "Level": item.get("Level", 1),
                "Final answer": item.get("Final answer"),
                "file_name": item.get("file_name"),
                "file_path": item.get("file_path"),
                "Annotator Metadata": item.get("Annotator Metadata", {}),
            }
            tasks.append(task)

        logger.info(f"Loaded {len(tasks)} tasks")
        return tasks

    async def _download_dataset(self) -> None:
        """Download the dataset from HuggingFace."""
        if self._dataset_path and self._dataset_path.exists():
            return

        logger.info(f"Downloading GAIA dataset to {self.cache_dir}")

        try:
            self._dataset_path = Path(
                snapshot_download(
                    repo_id=self.DATASET_ID,
                    repo_type="dataset",
                    token=self.hf_token,
                    cache_dir=str(self.cache_dir),
                )
            )
            logger.info(f"Dataset downloaded to {self._dataset_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def get_file_path(self, file_name: str | None) -> Path | None:
        """
        Get the full path to an auxiliary file.

        Args:
            file_name: Name of the file (from the dataset)

        Returns:
            Full path to the file, or None if not found
        """
        if not file_name or not self._dataset_path:
            return None

        # Files are typically in 2023/validation/ or 2023/test/
        for subdir in ["2023/validation", "2023/test", "2023"]:
            file_path = self._dataset_path / subdir / file_name
            if file_path.exists():
                return file_path

        # Try direct path
        file_path = self._dataset_path / file_name
        if file_path.exists():
            return file_path

        logger.warning(f"File not found: {file_name}")
        return None

    def get_file_content(self, file_name: str | None) -> bytes | None:
        """
        Get the content of an auxiliary file.

        Args:
            file_name: Name of the file

        Returns:
            File content as bytes, or None if not found
        """
        file_path = self.get_file_path(file_name)
        if file_path:
            return file_path.read_bytes()
        return None
