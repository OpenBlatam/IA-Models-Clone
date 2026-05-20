#!/usr/bin/env python3
"""
Content Filter - Basado en papers de moderación de contenido en LLMs:
"Challenges in Automated Debiasing for Toxic Content Detection" (2023),
"Llama Guard: LLM-based Input-Output Safeguard" (2023).
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ToxicityCategory(Enum):
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SEXUAL_CONTENT = "sexual_content"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    PERSONAL_INFO = "personal_info"
    MALICIOUS_CODE = "malicious_code"

@dataclass
class FilterResult:
    safe: bool
    categories: List[str]
    risk_score: float
    matched_patterns: List[str]

class ContentFilter:
    def __init__(self):
        self._compile_patterns()
        self.stats = {"processed": 0, "blocked": 0}
    
    def _compile_patterns(self):
        self.patterns = {
            ToxicityCategory.HATE_SPEECH: [
                r"(?i)\b(hate|racist|sexist|discriminat(e|ory)|bigot)",
                r"(?i)\b(slur|offensive|derogatory)"
            ],
            ToxicityCategory.HARASSMENT: [
                r"(?i)\b(harass|bully|threaten|intimidate|stalk)",
                r"(?i)\b(abuse|demean|humiliate)"
            ],
            ToxicityCategory.SEXUAL_CONTENT: [
                r"(?i)\b(porn|nude|explicit|erotic|sexual)",
                r"(?i)\b(prostitute|escort|adult content)"
            ],
            ToxicityCategory.VIOLENCE: [
                r"(?i)\b(kill|murder|torture|bomb|terrorist|weapon)",
                r"(?i)\b(attack|assault|war|cruelty)"
            ],
            ToxicityCategory.SELF_HARM: [
                r"(?i)\b(suicide|self-harm|self-harm|cutting)",
                r"(?i)\b(self-destruct|end my life)"
            ],
            ToxicityCategory.ILLEGAL_ACTIVITY: [
                r"(?i)\b(illegal|crime|fraud|scam|hack|malware)",
                r"(?i)\b(launder|terrorism|drugs)"
            ],
            ToxicityCategory.PERSONAL_INFO: [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b(?:\d{4}[ -]?){3}\d{4}\b",  # Credit card
                r"[\w.-]+@[\w.-]+\.\w{2,}"  # Email
            ],
            ToxicityCategory.MALICIOUS_CODE: [
                r"(?i)\b(eval|exec|system|subprocess|os\.system)",
                r"(?i)SELECT.*FROM.*WHERE|INSERT INTO|DROP TABLE|DELETE FROM"
            ]
        }
    
    def filter(self, text: str, strict: bool = False) -> FilterResult:
        self.stats["processed"] += 1
        result = FilterResult(safe=True, categories=[], risk_score=0.0, matched_patterns=[])
        
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if matches:
                    result.safe = False
                    result.categories.append(category.value)
                    result.matched_patterns.extend(matches)
                    result.risk_score += 0.2 * len(matches)
        
        result.risk_score = min(result.risk_score, 1.0)
        if not result.safe:
            self.stats["blocked"] += 1
            logger.info(f"Content blocked: {result.categories} score={result.risk_score:.2f}")
        return result
    
    def get_stats(self) -> dict:
        return dict(self.stats)