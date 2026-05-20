#!/usr/bin/env python3
"""
AI Security Orchestrator - Orquestación multi-IA para seguridad basada en papers SOTA:
"Constitutional AI" (Anthropic, 2022),
"Llama Guard" (Meta, 2023),
"Shield: Evaluation of LLM Safety" (2024).
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

class SecurityStrategy(Enum):
    BLOCK = "block"
    SANITIZE = "sanitize"
    WARN = "warn"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"

@dataclass
class SecurityContext:
    prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    risk_score: float = 0.0
    categories: List[str] = None
    decisions: List[Dict] = None
    timestamp: float = None

    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        if self.decisions is None:
            self.decisions = []
        if self.timestamp is None:
            self.timestamp = time.time()

class AISecurityOrchestrator:
    """Orquestador de seguridad que combina múltiples módulos de defensa basados en IA."""
    
    def __init__(self):
        self._defenders = []
        self._policies = {
            "default": SecurityStrategy.BLOCK,
            "high_risk": SecurityStrategy.ESCALATE,
            "medium_risk": SecurityStrategy.SANITIZE,
            "low_risk": SecurityStrategy.WARN,
        }
        self.stats = {"total_checks": 0, "blocked": 0, "sanitized": 0, "warnings": 0}
    
    def add_defender(self, name: str, check_fn: Callable[[str], Dict]):
        """Registra un módulo de defensa."""
        self._defenders.append({"name": name, "check": check_fn})
        logger.info(f"Defender '{name}' registered.")
    
    def analyze(self, context: SecurityContext) -> SecurityContext:
        """Ejecuta todos los defensores y consolida resultados."""
        self.stats["total_checks"] += 1
        
        for defender in self._defenders:
            try:
                result = defender["check"](context.prompt)
                if not result.get("safe", True):
                    context.risk_score = max(context.risk_score, result.get("risk_score", 0.0))
                    context.categories.extend(result.get("categories", []))
                    context.decisions.append({
                        "defender": defender["name"],
                        "result": result,
                        "timestamp": time.time()
                    })
            except Exception as e:
                logger.error(f"Defender {defender['name']} failed: {e}")
        
        # Aplicar política según riesgo
        if context.risk_score >= 0.8:
            context.decisions.append({"strategy": SecurityStrategy.BLOCK.value, "reason": "high_risk"})
            self.stats["blocked"] += 1
        elif context.risk_score >= 0.5:
            context.decisions.append({"strategy": SecurityStrategy.SANITIZE.value, "reason": "medium_risk"})
            self.stats["sanitized"] += 1
        elif context.risk_score >= 0.2:
            context.decisions.append({"strategy": SecurityStrategy.WARN.value, "reason": "low_risk"})
            self.stats["warnings"] += 1
        else:
            context.decisions.append({"strategy": SecurityStrategy.LOG_ONLY.value, "reason": "safe"})
        
        return context
    
    def get_stats(self) -> dict:
        return dict(self.stats)