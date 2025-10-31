from __future__ import annotations

from typing import Dict, List, Tuple


try:
    import yake  # type: ignore
    _YAKE = True
except Exception:
    _YAKE = False

try:
    import textstat  # type: ignore
    _TEXTSTAT = True
except Exception:
    _TEXTSTAT = False


class SEOService:
    """SEO helper with optional YAKE and Textstat.

    Falls back to simple heuristics if optional libs are not installed.
    """

    def suggest_keywords(self, topic: str, max_keywords: int = 8) -> List[str]:
        base = [
            topic.lower(),
            f"best {topic}",
            f"{topic} tips",
            f"{topic} guide",
            f"{topic} 2025",
            f"{topic} for beginners",
            f"how to {topic}",
            f"{topic} strategy",
        ]
        return base[:max_keywords]

    def suggest_keywords_yake(self, text: str, max_keywords: int = 10) -> List[str]:
        if not _YAKE:
            return self.suggest_keywords(text, max_keywords)
        extractor = yake.KeywordExtractor(top=max_keywords, stopwords=None)
        scored: List[Tuple[str, float]] = extractor.extract_keywords(text)
        scored.sort(key=lambda x: x[1])
        return [kw for kw, _ in scored][:max_keywords]

    def readability_score(self, text: str) -> float:
        if _TEXTSTAT:
            try:
                # Flesch reading ease ~0-100 (higher easier)
                score = float(textstat.flesch_reading_ease(text))
                return round(score, 2)
            except Exception:
                pass
        words = max(len(text.split()), 1)
        sentences = max(text.count("."), 1)
        avg_words_per_sentence = words / sentences
        score = max(0.0, min(100.0, 100.0 - (avg_words_per_sentence - 12) * 5))
        return round(score, 2)

    def meta(self, title: str, summary: str, keywords: List[str]) -> Dict[str, str]:
        return {
            "title": title[:60],
            "description": summary[:155],
            "keywords": ", ".join(keywords[:10]),
        }

