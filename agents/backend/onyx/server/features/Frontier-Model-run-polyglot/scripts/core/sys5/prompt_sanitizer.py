"""
System 5.9 — Prompt Sanitizer.

Intercepts prompts before LLM inference to remove recursive contamination,
deduplicate error blocks, and cap unbounded prompt growth.
"""

import re
import logging
from typing import Optional, Set

logger = logging.getLogger("sys5.prompt_sanitizer")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MAX_PROMPT_CHARS = 12_000
MAX_FINDINGS_DEPTH = 1

# ---------------------------------------------------------------------------
# Compiled patterns (module-level, compiled once)
# ---------------------------------------------------------------------------

_RE_ERROR_BLOCK = re.compile(
    r"\[ERROR DE SISTEMA\]:.*?(?=\[ERROR DE SISTEMA\]|TRUTHgpt:|$)",
    re.IGNORECASE | re.DOTALL,
)

_RE_NESTED_OBJECTIVE = re.compile(
    r"Objective:\s*Previous findings:.*?Objective:",
    re.IGNORECASE | re.DOTALL,
)

_RE_QUOTED_TOPIC = re.compile(r"['\"]([^'\"]+)['\"]")

_RE_OBJECTIVE_TAIL = re.compile(r"Objective:\s*(.+?)(?:\n|$)")


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

class PromptSanitizer:
    """
    Multi-pass prompt cleaner.

    >>> ps = PromptSanitizer()
    >>> clean = ps.sanitize(dirty_prompt)
    """

    def __init__(
        self,
        max_chars: int = MAX_PROMPT_CHARS,
        max_findings_depth: int = MAX_FINDINGS_DEPTH,
    ) -> None:
        self._max_chars = max_chars
        self._max_depth = max_findings_depth
        self._seen_queries: Set[str] = set()
        self._passes: int = 0

    # -- public API ---------------------------------------------------------

    def sanitize(self, prompt: str) -> str:
        """Apply all cleaning passes and return the sanitized prompt."""
        original = len(prompt)

        prompt = self._strip_findings_nesting(prompt)
        prompt = self._dedup_error_blocks(prompt)
        prompt = self._strip_nested_objectives(prompt)
        prompt = self._cap_size(prompt, original)

        if len(prompt) < original:
            self._passes += 1
            logger.info(
                "Sanitized %d→%d chars (−%d bytes, pass #%d)",
                original, len(prompt), original - len(prompt), self._passes,
            )
        return prompt

    def is_duplicate_query(self, query: str) -> bool:
        """Return *True* if this exact query was already attempted."""
        key = query.strip().lower()
        if key in self._seen_queries:
            logger.info("Duplicate query blocked: %.80s", query)
            return True
        self._seen_queries.add(key)
        return False

    def extract_topic(self, contaminated: str) -> Optional[str]:
        """
        Best-effort extraction of the real user topic from a
        recursively contaminated search query.
        """
        quotes = _RE_QUOTED_TOPIC.findall(contaminated)
        if quotes:
            return min(quotes, key=len)

        m = _RE_OBJECTIVE_TAIL.search(contaminated)
        if m:
            return m.group(1).strip()
        return None

    def reset(self) -> None:
        """Clear duplicate-query memory (e.g. new session)."""
        self._seen_queries.clear()

    def get_stats(self) -> dict:
        return {
            "sanitization_passes": self._passes,
            "tracked_queries": len(self._seen_queries),
        }

    def __repr__(self) -> str:
        return (
            f"<PromptSanitizer passes={self._passes} "
            f"queries={len(self._seen_queries)}>"
        )

    # -- internals ----------------------------------------------------------

    def _strip_findings_nesting(self, prompt: str) -> str:
        depth = prompt.count("Previous findings:")
        if depth <= self._max_depth:
            return prompt

        logger.warning(
            "'Previous findings:' nesting depth %d exceeds max %d",
            depth, self._max_depth,
        )
        head, _, tail = prompt.rpartition("Previous findings:")
        clean_head = head.replace("Previous findings:", "").strip()
        return f"{clean_head}\nPrevious findings:{tail}"

    def _dedup_error_blocks(self, prompt: str) -> str:
        matches = list(_RE_ERROR_BLOCK.finditer(prompt))
        if len(matches) <= 1:
            return prompt

        logger.info("Deduplicating %d [ERROR DE SISTEMA] blocks", len(matches) - 1)
        result = prompt
        for m in reversed(matches[:-1]):
            result = result[: m.start()] + result[m.end() :]
        return result

    def _strip_nested_objectives(self, prompt: str) -> str:
        cleaned = _RE_NESTED_OBJECTIVE.sub("Objective:", prompt)
        if cleaned != prompt:
            logger.info("Stripped nested Objective: contamination")
        return cleaned

    def _cap_size(self, prompt: str, original_len: int) -> str:
        if len(prompt) <= self._max_chars:
            return prompt

        head = self._max_chars // 3
        tail = self._max_chars * 2 // 3
        logger.warning("Truncating prompt %d→~%d chars", original_len, self._max_chars)
        return (
            prompt[:head]
            + "\n\n[… PROMPT TRUNCADO …]\n\n"
            + prompt[-tail:]
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

prompt_sanitizer = PromptSanitizer()
