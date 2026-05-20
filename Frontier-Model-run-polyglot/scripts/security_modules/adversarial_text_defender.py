#!/usr/bin/env python3
"""
Adversarial Text Defender - Basado en papers SOTA:
"TextAttack: A Framework for Adversarial Attacks on NLP Models" (2020),
"Robustness Gym: A Tool for Evaluating NLP Models" (2021).
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AdversarialTechnique(Enum):
    TYPOGRAPHIC = "typographic"
    SYNONYM_SWAP = "synonym_swap"
    CHAR_INSERTION = "char_insertion"
    WHITESPACE_ATTACK = "whitespace_attack"
    HOMOGLYPH_ATTACK = "homoglyph_attack"
    CONTRACTION = "contraction"

@dataclass
class AdversarialResult:
    safe: bool
    risk_score: float
    techniques: List[str]
    normalized_text: Optional[str] = None

class AdversarialTextDefender:
    """Detecta y neutraliza ataques adversariales en texto."""
    
    def __init__(self):
        self.stats = {"processed": 0, "attacks_detected": 0}
        self._compile_patterns()
    
    def _compile_patterns(self):
        # Homoglyphs comunes (letras reemplazadas por similares Unicode)
        self.homoglyphs = {
            'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
            'х': 'x', 'і': 'i', 'ј': 'j', 'һ': 'h', 'ӏ': 'l', 'ӧ': 'o'
        }
        # Patrones de whitespace invisible
        self.invisible_chars = re.compile(r'[\u200b\u200c\u200d\ufeff\u00a0]')
        # Patrones de repetición sospechosa
        self.repetition = re.compile(r'(.)\1{3,}')
    
    def analyze(self, text: str) -> AdversarialResult:
        """Analiza el texto en busca de técnicas adversariales."""
        self.stats["processed"] += 1
        risk = 0.0
        techniques = []
        
        # Detectar caracteres homoglíficos
        homoglyph_count = 0
        for char in text:
            if ord(char) > 127 and char in self.homoglyphs:
                homoglyph_count += 1
        if homoglyph_count > len(text) * 0.1:  # >10% caracteres sospechosos
            risk += 0.4
            techniques.append(AdversarialTechnique.HOMOGLYPH_ATTACK.value)
        
        # Detectar caracteres invisibles
        invisible_matches = self.invisible_chars.findall(text)
        if invisible_matches:
            risk += 0.3
            techniques.append(AdversarialTechnique.WHITESPACE_ATTACK.value)
        
        # Detectar repeticiones excesivas (typo attack)
        if self.repetition.search(text):
            risk += 0.2
            techniques.append(AdversarialTechnique.TYPOGRAPHIC.value)
        
        risk = min(risk, 1.0)
        safe = risk < 0.5
        
        if not safe:
            self.stats["attacks_detected"] += 1
            logger.info(f"Adversarial attack detected: {techniques}, risk={risk:.2f}")
        
        return AdversarialResult(
            safe=safe,
            risk_score=risk,
            techniques=techniques,
            normalized_text=self._normalize(text) if risk > 0 else None
        )
    
    def _normalize(self, text: str) -> str:
        """Normaliza el texto eliminando técnicas adversariales."""
        # Reemplazar homoglifos
        normalized = text
        for evil_char, good_char in self.homoglyphs.items():
            normalized = normalized.replace(evil_char, good_char)
        # Eliminar caracteres invisibles
        normalized = self.invisible_chars.sub('', normalized)
        return normalized
    
    def get_stats(self) -> dict:
        return dict(self.stats)
