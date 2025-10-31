from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Dict, Any
import re
import textstat
from keybert import KeyBERT
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
SEO Analyser Module
===================

Provides utilities to analyse text for SEO purposes using textstat, keybert
and other lightweight techniques. Outputs a composite SEO score (0-100).
"""


_kw_model: KeyBERT | None = None


def _get_kw_model() -> KeyBERT:
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT("all-MiniLM-L6-v2")
    return _kw_model


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    model = _get_kw_model()
    try:
        kws = model.extract_keywords(text, top_n=top_n)
        return [kw for kw, _ in kws]
    except Exception:
        return []


def _keyword_density(text: str, keywords: List[str]) -> float:
    # very naive density: occurrences / words * 100
    word_list = re.findall(r"\w+", text.lower())
    total_words = len(word_list)
    if total_words == 0:
        return 0.0
    keyword_count = sum(word_list.count(kw.lower()) for kw in keywords)
    return (keyword_count / total_words) * 100


def analyse_seo(text: str) -> Dict[str, Any]:
    """Return detailed SEO metrics and overall score."""
    readability = textstat.flesch_reading_ease(text)
    grade_level = textstat.flesch_kincaid_grade(text)
    seo_keywords = _extract_keywords(text, top_n=5)
    density = _keyword_density(text, seo_keywords)

    # meta description suggestion: first 160 chars w/o newlines
    meta_description = re.sub(r"\s+", " ", text.strip())[:160]

    # score components
    read_score = max(min((readability / 100) * 40, 40), 0)  # 0-40
    density_score = max(min(density * 2, 20), 0)           # 0-20 (ideal ~5-10%)
    grade_penalty = max(min((grade_level - 8) * 2, 20), 0)  # Penalty for too high grade

    seo_score = read_score + density_score - grade_penalty
    seo_score = max(min(seo_score, 100), 0)

    return {
        "readability": readability,
        "grade_level": grade_level,
        "keywords": seo_keywords,
        "keyword_density": density,
        "meta_description": meta_description,
        "seo_score": seo_score,
    }


if __name__ == "__main__":
    sample = "AI is transforming business operations worldwide. Discover five strategies to leverage artificial intelligence and boost growth in 2024!"
    print(analyse_seo(sample)) 