from __future__ import annotations

from typing import Dict

try:
    import textstat  # type: ignore
    _TEXTSTAT = True
except Exception:
    _TEXTSTAT = False


class AnalyticsService:
    """Content analytics with optional textstat metrics."""

    def content_metrics(self, text: str) -> Dict[str, float]:
        word_count = float(len(text.split()))
        reading_time_min = round(max(1.0, word_count / 200.0), 2)  # 200 wpm
        metrics: Dict[str, float] = {
            "word_count": word_count,
            "reading_time_min": reading_time_min,
        }
        if _TEXTSTAT:
            try:
                metrics.update(
                    {
                        "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
                        "smog_index": float(textstat.smog_index(text)),
                        "coleman_liau_index": float(textstat.coleman_liau_index(text)),
                    }
                )
            except Exception:
                pass
        return metrics


