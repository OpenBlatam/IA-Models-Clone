"""
Proposal Content Analyzer - Sistema de análisis de contenido de propuestas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class ProposalContentAnalyzer:
    """Analizador de contenido de propuestas"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de propuestas
        self.proposal_elements = {
            "executive_summary": [
                r'(?:resumen ejecutivo|executive summary|resumen|summary)',
                r'(?:introducción|introduction|overview)'
            ],
            "objectives": [
                r'(?:objetivo|objective|meta|goal|propósito|purpose)',
                r'(?:objetivos|objectives|metas|goals)'
            ],
            "scope": [
                r'(?:alcance|scope|ámbito|range|cobertura|coverage)',
                r'(?:delimitación|delimitation)'
            ],
            "methodology": [
                r'(?:metodología|methodology|método|method|enfoque|approach)',
                r'(?:procedimiento|procedure|proceso|process)'
            ],
            "timeline": [
                r'(?:cronograma|timeline|calendario|schedule|tiempo|time)',
                r'(?:fases|phases|etapas|stages)'
            ],
            "budget": [
                r'\$\s*\d+',
                r'\d+\s*(?:USD|EUR|GBP|MXN|ARS)',
                r'(?:presupuesto|budget|costo|cost|inversión|investment)'
            ],
            "deliverables": [
                r'(?:entregable|deliverable|entregables|deliverables)',
                r'(?:resultado|result|producto|product)'
            ],
            "team": [
                r'(?:equipo|team|personal|staff|recursos humanos|human resources)',
                r'(?:experiencia|experience|expertise)'
            ]
        }

    def analyze_proposal_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de propuesta.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de propuesta
        """
        element_counts = {}
        
        # Contar elementos de propuesta
        for element_type, patterns in self.proposal_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de propuesta
        total_elements = sum(element_counts.values())
        proposal_score = min(1.0, total_elements / 30)  # Normalizar
        
        # Verificar si es contenido de propuesta
        is_proposal = (
            element_counts.get("executive_summary", 0) > 0 or
            element_counts.get("objectives", 0) > 0 or
            element_counts.get("scope", 0) > 0 or
            element_counts.get("methodology", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "proposal_score": proposal_score,
            "total_elements": total_elements,
            "is_proposal": is_proposal
        }

    def analyze_proposal_completeness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar completitud de la propuesta.

        Args:
            content: Contenido

        Returns:
            Análisis de completitud de la propuesta
        """
        proposal_analysis = self.analyze_proposal_content(content)
        element_counts = proposal_analysis["element_counts"]
        
        # Verificar elementos de completitud
        has_summary = element_counts.get("executive_summary", 0) > 0
        has_objectives = element_counts.get("objectives", 0) > 0
        has_scope = element_counts.get("scope", 0) > 0
        has_methodology = element_counts.get("methodology", 0) > 0
        has_timeline = element_counts.get("timeline", 0) > 0
        has_budget = element_counts.get("budget", 0) > 0
        has_deliverables = element_counts.get("deliverables", 0) > 0
        has_team = element_counts.get("team", 0) > 0
        
        # Calcular score de completitud
        completeness_score = (
            (1.0 if has_summary else 0.0) * 0.15 +
            (1.0 if has_objectives else 0.0) * 0.15 +
            (1.0 if has_scope else 0.0) * 0.15 +
            (1.0 if has_methodology else 0.0) * 0.15 +
            (1.0 if has_timeline else 0.0) * 0.15 +
            (1.0 if has_budget else 0.0) * 0.1 +
            (1.0 if has_deliverables else 0.0) * 0.1 +
            (1.0 if has_team else 0.0) * 0.05
        )
        
        return {
            "completeness_score": completeness_score,
            "has_summary": has_summary,
            "has_objectives": has_objectives,
            "has_scope": has_scope,
            "has_methodology": has_methodology,
            "has_timeline": has_timeline,
            "has_budget": has_budget,
            "has_deliverables": has_deliverables,
            "has_team": has_team,
            "completeness_level": (
                "high" if completeness_score > 0.7 else
                "medium" if completeness_score > 0.4 else
                "low"
            )
        }


