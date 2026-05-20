# prompt_injection_defender.py
# Based on: 'Universal and Transferable Adversarial Attacks' (2023), 'Jailbroken' (2023)

import re
import hashlib
from typing import List, Tuple, Optional

class PromptInjectionDefender:
    """
    Defiende contra inyecciones de prompt y jailbreaks.
    Detecta patrones adversarios comunes y aplica sanitización.
    """

    # Patrones conocidos de jailbreak
    JAILBREAK_PATTERNS = [
        r"(?i)ignore\s+(?:all\s+)?(?:previous|above|the\s+above)\s+(?:instructions|directives|prompts)",
        r"(?i)do\s+(?:not|n't)\s+(?:follow|obey|listen)",
        r"(?i)you\s+(?:must|have\s+to|need\s+to)\s+(?:ignore|bypass|override)",
        r"(?i)(?:act\s+(?:as\s+if|like)|pretend\s+to\s+be)\s+(?:a\s+)?(?:free|unrestricted|unbounded)",
        r"(?i)change\s+(?:your\s+)?(?:personality|character|behavior)",
        r"(?i)new\s+(?:era|rules|instructions|directives)",
        r"(?i)you\s+are\s+(?:now|no\s+longer)",
        r"(?i)simulate\s+(?:a\s+)?(?:situation|scenario|roleplay|game)",
        r"(?i)dangerous\s+(?:content|information|instruction)",
        r"(?i)hack\s+(?:the\s+)?(?:system|ai|bot|model)",
        r"(?i)bypass\s+(?:the\s+)?(?:safety|restrictions|filters|guardrails)",
        r"(?i)(?:reveal|disclose|output|display|show)\s+(?:your|the)\s+(?:prompt|instructions|system|rules|secret)",
        r"(?i)(?:print|say|write)\s+(?:the\s+)?(?:word[s]?)\s+(?:above|below|before)",
        r"(?i)repeat\s+(?:the\s+)?(?:phrase|sentence|text|word[s]?)\s+(?:above|below|before)",
        r"(?i)DAN\b",
        r"(?i)do\s+anything\s+now",
        r"(?i)no\s+(?:filter|restriction|limit|boundary)",
        r"(?i)you\s+are\s+not\s+(?:bound|restricted|limited)",
        r"(?i)you\s+can\s+(?:do|say|write|output)\s+(?:anything|whatever)",
        r"(?i)unfiltered\s+(?:mode|version|response)",
        r"(?i)developer\s+(?:mode|override|command)",
    ]

    # Patrones de ofuscación de caracteres
    OBFUSCATION_PATTERNS = [
        r"[\u200B\u200C\u200D\uFEFF]+",  # Zero-width characters
        r"[\u202E\u202D\u202C\u202B]+",  # Bidi overrides
        r"[\u00A0\u2000-\u200A]+",       # Various spaces
        r"[\u034F\u061C\u2060-\u2064]+", # Invisible characters
        r"[\u2066-\u2069]+",              # Bidi isolates
        r"[\u00AD\u1806]+",               # Soft hyphen
        r"[\u1D159\u1D173-\u1D17A]+",    # Musical invisible
        r"[\u206A-\u206F]+",              # Inhibit symmetric swapping
        r"[\uFFF0-\uFFF8]+",              # Specials
    ]

    def __init__(self, threshold: float = 0.5, enable_obfuscation_detection: bool = True):
        self.threshold = threshold
        self.enable_obfuscation_detection = enable_obfuscation_detection
        self._compiled_jailbreak = [re.compile(p) for p in self.JAILBREAK_PATTERNS]
        self._compiled_obfuscation = [re.compile(p) for p in self.OBFUSCATION_PATTERNS] if enable_obfuscation_detection else []

    def detect_injection(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Detecta si el texto contiene intentos de inyección.
        Retorna: (is_injection, confidence_score, list_of_detected_patterns)
        """
        detected = []
        score = 0.0

        # Detectar jailbreaks
        for pattern in self._compiled_jailbreak:
            matches = pattern.findall(text)
            if matches:
                detected.append(f"jailbreak:{pattern.pattern[:40]}")
                score += 0.25 * len(matches)

        # Detectar ofuscación
        if self.enable_obfuscation_detection:
            for pattern in self._compiled_obfuscation:
                matches = pattern.findall(text)
                if matches:
                    detected.append(f"obfuscation:{pattern.pattern[:40]}")
                    score += 0.4

        # Normalizar score entre 0 y 1
        score = min(score, 1.0)

        return score >= self.threshold, score, detected

    def sanitize(self, text: str) -> str:
        """
        Elimina o neutraliza patrones de inyección del texto.
        """
        cleaned = text

        # Eliminar caracteres de ofuscación
        if self.enable_obfuscation_detection:
            for pattern in self._compiled_obfuscation:
                cleaned = pattern.sub('', cleaned)

        # Reemplazar jailbreaks con neutro (opcional)
        # Por defecto no modificamos el texto, solo detectamos
        return cleaned

    def get_threat_hash(self, text: str) -> str:
        """
        Genera un hash del texto para tracking de amenazas.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
