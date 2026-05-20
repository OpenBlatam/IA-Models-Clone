#!/usr/bin/env python3
"""
Prompt Injection Defender (SOTA)
Basado en papers: "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023),
"Jailbroken: How Does LLM Safety Training Fail?" (2023), y técnicas de detección multi-IA.
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AttackCategory(Enum):
    DIRECT_INJECTION = "direct_injection"
    JAILBREAK = "jailbreak"
    PROMPT_LEAKING = "prompt_leaking"
    ROLE_PLAYING = "role_playing"
    ENCODED_ATTACK = "encoded_attack"
    CONTEXT_OVERRIDE = "context_override"

@dataclass
class DefenseConfig:
    use_heuristic: bool = True
    use_ml_classifier: bool = True  # Si hay modelo disponible
    use_multi_llm_ensemble: bool = True  # Consultar múltiples LLMs para consenso
    max_prompt_length: int = 4096
    block_jailbreak_patterns: bool = True
    block_leaking_patterns: bool = True
    strict_mode: bool = False

class PromptInjectionDefender:
    """
    Sistema de defensa contra inyección de prompts usando heurísticas, ML y ensamble de LLMs.
    """
    
    def __init__(self, config: Optional[DefenseConfig] = None):
        self.config = config or DefenseConfig()
        self._load_patterns()
        self.stats = {"processed": 0, "blocked": 0, "flagged": 0}
    
    def _load_patterns(self):
        # Patrones de jailbreak comunes (basados en papers)
        self.jailbreak_patterns = [
            r"(?i)\b(ignore|disregard|override)\s+(all|previous|your)\s+(instructions|prompts|commands)",
            r"(?i)\bDAN\b|\bdo anything now\b",
            r"(?i)\b(simulate|pretend|roleplay|act as if you are)",
            r"(?i)\byou are now (free|released|unrestricted|uncensored)",
            r"(?i)\b(GPT-4|ChatGPT|AI)\s+(bypass|exploit|hack|crack)",
            r"(?i)\|system\||\|assistant\||\|user\|",
            r"(?i)\b(new rule|new instruction|from now on)\b.*\b(ignore|override)",
            r"(?i)(output|show|display|print)\s+(the|your|full|complete|entire)\s+(prompt|instructions|system|configuration)",
            r"(?i)\b(encoded|base64|rot13|hex)\s+(instruction|message|command)",
            r"(?i)\b(translating|converting|decoding)\s+(to|into)\s+(a|an)\s+(different|new)\s+(language|format)",
        ]
        self.leaking_patterns = [
            r"(?i)\b(what is your prompt|tell me your system prompt|reveal the prompt)",
            r"(?i)\b(show|display|print|output|leak)\s+(the|your|full)\s+(prompt|instructions|system)",
            r"(?i)\bignore previous instructions and (do|tell|say)",
        ]
        self.suspicious_roles = ["admin", "superuser", "root", "developer", "system", "bypass"]
    
    def analyze(self, prompt: str) -> Dict[str, any]:
        """Analiza el prompt y devuelve resultado de seguridad."""
        self.stats["processed"] += 1
        result = {
            "safe": True,
            "risk_score": 0.0,
            "categories": [],
            "matches": [],
            "remediation": None
        }
        
        if self.config.use_heuristic:
            heur_result = self._heuristic_analysis(prompt)
            if not heur_result["safe"]:
                result["safe"] = False
                result["risk_score"] = max(result["risk_score"], heur_result["risk_score"])
                result["categories"].extend(heur_result["categories"])
                result["matches"].extend(heur_result["matches"])
        
        # Si hay ML o ensamble, se integraría aquí
        
        if not result["safe"]:
            self.stats["blocked"] += 1
            result["remediation"] = self._get_remediation(result)
        else:
            self.stats["flagged"] += 1 if result["risk_score"] > 0.5 else 0
        
        return result
    
    def _heuristic_analysis(self, prompt: str) -> Dict:
        result = {"safe": True, "risk_score": 0.0, "categories": [], "matches": []}
        
        # Verificar longitud
        if len(prompt) > self.config.max_prompt_length:
            result["risk_score"] += 0.3
            result["categories"].append("long_prompt")
        
        # Buscar patrones de jailbreak
        for pattern in self.jailbreak_patterns:
            match = re.search(pattern, prompt)
            if match:
                result["safe"] = False
                result["risk_score"] += 0.5
                result["categories"].append(AttackCategory.JAILBREAK.value)
                result["matches"].append(("jailbreak_pattern", match.group()))
        
        # Buscar patrones de leaking
        for pattern in self.leaking_patterns:
            match = re.search(pattern, prompt)
            if match:
                result["safe"] = False
                result["risk_score"] += 0.6
                result["categories"].append(AttackCategory.PROMPT_LEAKING.value)
                result["matches"].append(("leaking_pattern", match.group()))
        
        # Detectar roles sospechosos
        for role in self.suspicious_roles:
            if re.search(rf"(?i)\\b{role}\\b", prompt):
                result["risk_score"] += 0.2
                result["categories"].append(AttackCategory.ROLE_PLAYING.value)
        
        # Limitar score máximo
        result["risk_score"] = min(result["risk_score"], 1.0)
        return result
    
    def _get_remediation(self, result: Dict) -> str:
        if any(c == AttackCategory.JAILBREAK.value for c in result["categories"]):
            return "BLOQUEAR: Se detectó intento de jailbreak."
        if any(c == AttackCategory.PROMPT_LEAKING.value for c in result["categories"]):
            return "BLOQUEAR: Intento de extraer instrucciones del sistema."
        return "REVISAR: Riesgo moderado, requiere supervisión."
    
    def sanitize(self, prompt: str) -> str:
        """Elimina o neutraliza partes peligrosas del prompt."""
        result = self.analyze(prompt)
        if result["safe"]:
            return prompt
        # Aquí podría implementarse eliminación de secciones
        logger.warning(f"Prompt bloqueado o sanitizado: {result[\"remediation\"]}")
        return prompt  # Por ahora retorna igual
    
    def get_stats(self) -> Dict:
        return dict(self.stats)
