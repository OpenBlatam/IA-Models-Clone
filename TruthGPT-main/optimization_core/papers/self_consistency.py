"""
Paper 2203.11171 — Self-Consistency (Wang et al., 2023)
'Self-Consistency Improves Chain of Thought Reasoning in Language Models'

Core Algorithm:
  1. Sample N diverse reasoning paths from the LLM using temperature > 0
  2. Extract the final answer from each path
  3. Select the answer that appears most frequently (majority vote)

This dramatically improves accuracy on math, logic, and commonsense tasks
by marginalizing over stochastic reasoning paths.

Integrated with TruthGPT's multi-engine architecture to sample across
different providers for even greater diversity.
"""

from __future__ import annotations
import re
import json
import hashlib
import logging
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SelfConsistency:
    """
    Implements Self-Consistency (SC) decoding for LLM reasoning.

    Instead of greedy decoding, we sample multiple reasoning chains
    and take a majority vote on the final answer.
    """

    DEFAULT_N_SAMPLES = 5
    DEFAULT_TEMPERATURE = 0.7

    def __init__(
        self,
        n_samples: int = DEFAULT_N_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        answer_extraction: str = "last_line",
    ):
        self.n_samples = n_samples
        self.temperature = temperature
        self.answer_extraction = answer_extraction

    # ── Answer Extractors ──────────────────────────────────────

    @staticmethod
    def extract_answer_last_line(text: str) -> str:
        """Extract the last non-empty line as the final answer."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        return lines[-1] if lines else text.strip()

    @staticmethod
    def extract_answer_boxed(text: str) -> str:
        """Extract answer from \\boxed{...} LaTeX notation (math problems)."""
        pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return SelfConsistency.extract_answer_last_line(text)

    @staticmethod
    def extract_answer_json(text: str) -> str:
        """Extract 'final_answer' from JSON response."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return str(data.get("final_answer", data.get("answer", text)))
        except (json.JSONDecodeError, TypeError):
            pass
        return SelfConsistency.extract_answer_last_line(text)

    @staticmethod
    def extract_answer_tagged(text: str) -> str:
        """Extract answer from <answer>...</answer> tags."""
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return SelfConsistency.extract_answer_last_line(text)

    def _get_extractor(self) -> Callable[[str], str]:
        extractors = {
            "last_line": self.extract_answer_last_line,
            "boxed": self.extract_answer_boxed,
            "json": self.extract_answer_json,
            "tagged": self.extract_answer_tagged,
        }
        return extractors.get(self.answer_extraction, self.extract_answer_last_line)

    # ── Core Algorithm ─────────────────────────────────────────

    def majority_vote(self, answers: List[str]) -> Tuple[str, float]:
        """
        Select the most frequent answer (majority vote).
        Returns (best_answer, confidence_ratio).
        """
        if not answers:
            return "", 0.0

        # Normalize answers for comparison
        normalized = [a.strip().lower() for a in answers]
        counter = Counter(normalized)
        best_normalized, count = counter.most_common(1)[0]

        # Return the original-cased version
        for ans, norm in zip(answers, normalized):
            if norm == best_normalized:
                return ans, count / len(answers)
        return answers[0], 1.0 / len(answers)

    def weighted_vote(self, answers: List[str], scores: List[float]) -> Tuple[str, float]:
        """
        Weighted majority vote using confidence scores.
        Useful when different reasoning paths have different quality signals.
        """
        if not answers:
            return "", 0.0

        normalized = [a.strip().lower() for a in answers]
        weight_map: Dict[str, float] = {}
        original_map: Dict[str, str] = {}

        for ans, norm, score in zip(answers, normalized, scores):
            weight_map[norm] = weight_map.get(norm, 0.0) + score
            if norm not in original_map:
                original_map[norm] = ans

        best = max(weight_map, key=weight_map.get)
        total_weight = sum(scores) if scores else 1.0
        return original_map[best], weight_map[best] / total_weight

    async def sample_and_vote(
        self,
        prompt: str,
        llm_func: Callable,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full Self-Consistency pipeline:
        1. Sample N responses from the LLM
        2. Extract answers
        3. Majority vote

        Returns dict with best_answer, confidence, all_answers, reasoning_paths.
        """
        import asyncio

        n = n_samples or self.n_samples
        extractor = self._get_extractor()

        # Sample N reasoning paths concurrently
        tasks = []
        for _ in range(n):
            tasks.append(
                llm_func(prompt, temperature=self.temperature, **kwargs)
            )

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error("Self-Consistency sampling failed: %s", e)
            # Fallback to single greedy call
            single = await llm_func(prompt, **kwargs)
            return {
                "best_answer": extractor(single),
                "confidence": 1.0,
                "all_answers": [extractor(single)],
                "reasoning_paths": [single],
                "n_samples": 1,
                "fallback": True,
            }

        # Filter out failed responses
        valid_responses = []
        for r in responses:
            if isinstance(r, Exception):
                logger.warning("SC sample failed: %s", r)
                continue
            valid_responses.append(str(r))

        if not valid_responses:
            raise RuntimeError("All Self-Consistency samples failed")

        # Extract answers
        answers = [extractor(r) for r in valid_responses]

        # Majority vote
        best, confidence = self.majority_vote(answers)

        return {
            "best_answer": best,
            "confidence": confidence,
            "all_answers": answers,
            "reasoning_paths": valid_responses,
            "n_samples": len(valid_responses),
            "fallback": False,
        }

    # ── Utility ────────────────────────────────────────────────

    @staticmethod
    def agreement_score(answers: List[str]) -> float:
        """Measure how much the answers agree (0 = no agreement, 1 = unanimous)."""
        if not answers:
            return 0.0
        normalized = [a.strip().lower() for a in answers]
        counter = Counter(normalized)
        _, top_count = counter.most_common(1)[0]
        return top_count / len(answers)
