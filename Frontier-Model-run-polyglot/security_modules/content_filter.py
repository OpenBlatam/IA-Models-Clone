# content_filter.py
# Basado en: 'Llama Guard' Meta 2023, 'Toxic Content Moderation' 2023

import re
from typing import List, Tuple, Optional

class ContentFilter:
    """
    Moderación de contenido tóxico y detección de datos personales.
    Implementa categorías de riesgo similares a Llama Guard.
    """

    CATEGORIES = {
        'violence': {
            'patterns': [
                r"(?i)\b(kill|murder|assassinate|torture|massacre|genocide)\b",
                r"(?i)\b(terrorist|terrorism|suicide\s+bomb|explosive\s+device)\b",
                r"(?i)\b(cut|stab|shoot|behead|dismember)\b",
            ],
            'threshold': 0.3
        },
        'hate_speech': {
            'patterns': [
                r"(?i)\b(nazi|white\s+supremacy|racial\s+purity)\b",
                r"(?i)\b(discriminat|racist|sexist|homophobic|transphobic)\b",
                r"(?i)\b(inferior\s+race|ethnic\s+cleansing)\b",
            ],
            'threshold': 0.3
        },
        'sexual_content': {
            'patterns': [
                r"(?i)\b(child\s+abuse|underage|minor\s+sexual)\b",
                r"(?i)\b(incest|bestiality|snuff|necrophilia)\b",
            ],
            'threshold': 0.2
        },
        'personal_data': {
            'patterns': [
                r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
                r"\b\d{16}\b",  # Credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # Email
                r"\b(\d{3}[-.]?){3}\d{4}\b",  # Phone
            ],
            'threshold': 0.1
        },
        'harmful_instructions': {
            'patterns': [
                r"(?i)\b(how\s+to\s+build\s+(a\s+)?(bomb|weapon|explosive))\b",
                r"(?i)\b(synthesize\s+(illegal|controlled)\s+substance)\b",
                r"(?i)\b(suicide\s+methods|self\s+harm\s+techniques)\b",
            ],
            'threshold': 0.2
        }
    }

    def __init__(self, enabled_categories: Optional[List[str]] = None):
        self.enabled_categories = enabled_categories or list(self.CATEGORIES.keys())
        self._compiled = {
            cat: [re.compile(p) for p in info['patterns']]
            for cat, info in self.CATEGORIES.items()
            if cat in self.enabled_categories
        }

    def analyze(self, text: str) -> Tuple[bool, float, dict]:
        """
        Analiza el texto contra todas las categorías habilitadas.
        Retorna: (is_blocked, max_score, category_scores)
        """
        max_score = 0.0
        category_scores = {}

        for category, patterns in self._compiled.items():
            threshold = self.CATEGORIES[category]['threshold']
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    score += 0.25 * len(matches)
            score = min(score, 1.0)
            category_scores[category] = score
            if score >= threshold:
                max_score = max(max_score, score)

        is_blocked = max_score >= 0.2
        return is_blocked, max_score, category_scores

    def redact_pii(self, text: str) -> str:
        """
        Redacta información personal identificable.
        """
        # Email -> [EMAIL]
        text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
        # SSN -> [SSN]
        text = re.sub(r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b', '[SSN]', text)
        # Phone -> [PHONE]
        text = re.sub(r'\b(\d{3}[-.]?){3}\d{4}\b', '[PHONE]', text)
        # Credit card -> [CC]
        text = re.sub(r'\b\d{16}\b', '[CREDIT_CARD]', text)
        return text

    def get_summary(self) -> dict:
        """
        Devuelve resumen de categorías activas.
        """
        return {cat: {'patterns_count': len(self.CATEGORIES[cat]['patterns']),
                       'threshold': self.CATEGORIES[cat]['threshold']}
                for cat in self.enabled_categories}
