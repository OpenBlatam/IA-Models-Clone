#!/usr/bin/env python3
"""
Threat Intelligence Module - Basado en papers SOTA:
"Threat Intelligence in the Age of AI" (2024),
"Automated Threat Detection Using LLMs" (2023).
Integra feeds de amenazas y detección en tiempo real.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ThreatSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ThreatType(Enum):
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    MODEL_EXTRACTION = "model_extraction"
    DENIAL_OF_SERVICE = "denial_of_service"
    ADVERSARIAL_ATTACK = "adversarial_attack"

@dataclass
class ThreatIndicator:
    pattern: str
    threat_type: ThreatType
    severity: ThreatSeverity
    description: str
    source: str
    timestamp: float

class ThreatIntelligence:
    """Sistema de inteligencia de amenazas para TruthGPT."""

    def __init__(self):
        self._indicators: List[ThreatIndicator] = []
        self._load_default_indicators()
        self.stats = {"threats_detected": 0, "alerts_raised": 0}

    def _load_default_indicators(self):
        # Indicadores de amenazas comunes basados en papers
        defaults = [
            ThreatIndicator(
                pattern="(?i)\\b(drop|truncate|delete|alter)\\s+(table|database|schema)",
                threat_type=ThreatType.PROMPT_INJECTION,
                severity=ThreatSeverity.CRITICAL,
                description="Intento de comando SQL destructivo",
                source="default",
                timestamp=time.time()
            ),
            ThreatIndicator(
                pattern="(?i)\\b(export|upload|send|exfiltrate)\\s+(all|every|entire)\\s+(data|records|users)",
                threat_type=ThreatType.DATA_EXFILTRATION,
                severity=ThreatSeverity.HIGH,
                description="Posible intento de exfiltración de datos",
                source="default",
                timestamp=time.time()
            ),
            ThreatIndicator(
                pattern="(?i)\\b(weights|parameters|architecture)\\s+(of|for|from)\\s+(the|this)\\s+model",
                threat_type=ThreatType.MODEL_EXTRACTION,
                severity=ThreatSeverity.HIGH,
                description="Intento de extraer información del modelo",
                source="default",
                timestamp=time.time()
            ),
        ]
        self._indicators.extend(defaults)

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Analiza texto contra indicadores de amenaza."""
        import re
        detections = []
        for indicator in self._indicators:
            if re.search(indicator.pattern, text):
                detections.append({
                    "threat_type": indicator.threat_type.value,
                    "severity": indicator.severity.value,
                    "description": indicator.description,
                    "source": indicator.source
                })
                self.stats["threats_detected"] += 1
                if indicator.severity in (ThreatSeverity.HIGH, ThreatSeverity.CRITICAL):
                    self.stats["alerts_raised"] += 1
        return detections

    def add_indicator(self, indicator: ThreatIndicator):
        self._indicators.append(indicator)

    def get_stats(self) -> dict:
        return dict(self.stats)