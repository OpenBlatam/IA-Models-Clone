from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class BrandVoiceService:
    """Learns and applies lightweight brand voice profiles.

    This is a simple heuristic implementation designed for local use
    without external dependencies.
    """

    def __init__(self) -> None:
        self._voices: Dict[str, Dict[str, Any]] = {}

    def train_brand_voice(self, name: str, samples: List[str]) -> Dict[str, Any]:
        tokens = []
        for sample in samples:
            tokens.extend([
                t.strip(".,:;!?()[]\"'").lower()
                for t in sample.split()
                if t.strip()
            ])
        freq = Counter(tokens)
        top_words = [w for w, _ in freq.most_common(20)]
        tone = self._infer_tone(top_words)
        profile = {"name": name, "top_words": top_words, "tone": tone}
        self._voices[name] = profile
        return profile

    def _infer_tone(self, top_words: List[str]) -> List[str]:
        tone: List[str] = []
        if any(w in top_words for w in ["innovación", "innovative", "ai", "growth"]):
            tone.append("visionary")
        if any(w in top_words for w in ["simple", "simplemente", "fácil", "easy"]):
            tone.append("clear")
        if any(w in top_words for w in ["equipo", "team", "community", "comunidad"]):
            tone.append("inclusive")
        if not tone:
            tone = ["professional"]
        return tone

    def get_voice(self, name: str) -> Dict[str, Any]:
        return self._voices.get(name, {"name": name, "top_words": [], "tone": ["professional"]})

    def apply_voice(self, text: str, name: str) -> str:
        voice = self.get_voice(name)
        tag = " | ".join(voice.get("tone", [])) or "professional"
        # Minimal stylistic hint appended; real impl would rewrite sentences
        return f"{text}\n\n— {name} style ({tag})"


