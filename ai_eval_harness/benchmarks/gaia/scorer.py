"""
GAIA answer scorer.

This module provides flexible answer matching for GAIA benchmark,
handling various answer formats (strings, numbers, lists).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class GAIAScorer:
    """
    Scorer for GAIA benchmark answers.

    Supports flexible matching including:
    - Case-insensitive string comparison
    - Numeric comparison with tolerance
    - List/set comparison (order-independent)
    - Whitespace and punctuation normalization
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.01,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
    ) -> None:
        """
        Initialize the scorer.

        Args:
            numeric_tolerance: Tolerance for numeric comparisons
            ignore_case: Whether to ignore case in string comparisons
            ignore_punctuation: Whether to ignore punctuation
        """
        self.numeric_tolerance = numeric_tolerance
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation

    def score(self, predicted: Any, ground_truth: Any) -> tuple[bool, float]:
        """
        Score a predicted answer against ground truth.

        Args:
            predicted: The agent's answer
            ground_truth: The expected answer

        Returns:
            Tuple of (is_correct, score)
        """
        if predicted is None or ground_truth is None:
            return False, 0.0

        # Convert to strings for processing
        pred_str = str(predicted).strip()
        gt_str = str(ground_truth).strip()

        # Normalize both
        pred_norm = self._normalize(pred_str)
        gt_norm = self._normalize(gt_str)

        # Exact match after normalization
        if pred_norm == gt_norm:
            return True, 1.0

        # Try numeric comparison
        if self._is_numeric(pred_norm) and self._is_numeric(gt_norm):
            try:
                pred_num = self._parse_number(pred_norm)
                gt_num = self._parse_number(gt_norm)
                if abs(pred_num - gt_num) <= self.numeric_tolerance:
                    return True, 1.0
                # Partial credit for close answers
                if abs(pred_num - gt_num) <= self.numeric_tolerance * 10:
                    return False, 0.5
            except ValueError:
                pass

        # Try list comparison
        if self._looks_like_list(pred_str) and self._looks_like_list(gt_str):
            pred_items = self._parse_list(pred_str)
            gt_items = self._parse_list(gt_str)

            if set(pred_items) == set(gt_items):
                return True, 1.0

            # Partial credit for partial matches
            if pred_items and gt_items:
                intersection = set(pred_items) & set(gt_items)
                union = set(pred_items) | set(gt_items)
                if intersection:
                    jaccard = len(intersection) / len(union)
                    return False, jaccard

        # Check if predicted contains ground truth (partial match)
        if gt_norm in pred_norm:
            return False, 0.5

        return False, 0.0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        result = text.strip()

        if self.ignore_case:
            result = result.lower()

        if self.ignore_punctuation:
            # Remove punctuation except for decimal points in numbers
            result = re.sub(r'[^\w\s\.\-]', '', result)

        # Normalize whitespace
        result = ' '.join(result.split())

        return result

    def _is_numeric(self, text: str) -> bool:
        """Check if text looks like a number."""
        # Remove common formatting
        cleaned = text.replace(',', '').replace(' ', '').replace('$', '').replace('%', '')
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _parse_number(self, text: str) -> float:
        """Parse a number from text."""
        # Remove common formatting
        cleaned = text.replace(',', '').replace(' ', '').replace('$', '').replace('%', '')
        return float(cleaned)

    def _looks_like_list(self, text: str) -> bool:
        """Check if text looks like a list."""
        # Check for common list separators
        return ',' in text or ';' in text or '\n' in text or re.search(r'\d+\.\s', text)

    def _parse_list(self, text: str) -> list[str]:
        """Parse a list from text."""
        # Try different separators
        items: list[str] = []

        # First try newlines
        if '\n' in text:
            items = text.split('\n')
        # Then comma
        elif ',' in text:
            items = text.split(',')
        # Then semicolon
        elif ';' in text:
            items = text.split(';')
        else:
            items = [text]

        # Clean up items
        cleaned = []
        for item in items:
            # Remove numbering like "1." or "a)"
            item = re.sub(r'^\s*[\d\w]+[\.\)]\s*', '', item)
            item = item.strip()
            if item:
                cleaned.append(self._normalize(item))

        return cleaned


def extract_final_answer(content: str) -> str | None:
    """
    Extract the final answer from agent response.

    Looks for patterns like "FINAL ANSWER:" or "The answer is:".

    Args:
        content: Agent's response content

    Returns:
        Extracted answer or None
    """
    if not content:
        return None

    # Common patterns for final answer
    patterns = [
        r"FINAL ANSWER:\s*(.+?)(?:\n|$)",
        r"Final Answer:\s*(.+?)(?:\n|$)",
        r"The answer is:?\s*(.+?)(?:\n|$)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"\*\*(.+?)\*\*\s*$",  # Bold at end of response
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up common suffixes
            answer = re.sub(r'[\.\s]+$', '', answer)
            return answer

    # If no pattern found, try to get the last line
    lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        # Skip if it looks like a question or instruction
        if not last_line.endswith('?') and len(last_line) < 200:
            return last_line

    return None
