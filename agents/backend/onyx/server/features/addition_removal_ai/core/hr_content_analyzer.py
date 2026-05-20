"""
HR Content Analyzer - Sistema de análisis de contenido de recursos humanos
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class HRContentAnalyzer:
    """Analizador de contenido de recursos humanos"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de HR
        self.hr_elements = {
            "hr_terms": {
                "es": ["empleado", "trabajador", "colaborador", "reclutamiento", "selección", "contratación", "evaluación", "desarrollo"],
                "en": ["employee", "worker", "staff", "recruitment", "selection", "hiring", "evaluation", "development"]
            },
            "job_requirements": [
                r'(?:requisito|requirement|calificación|qualification|experiencia|experience)',
                r'(?:educación|education|título|degree|certificación|certification)'
            ],
            "benefits": [
                r'(?:beneficio|benefit|prestación|perk|compensación|compensation)',
                r'(?:salario|salary|sueldo|wage|vacaciones|vacation|seguro|insurance)'
            ],
            "responsibilities": [
                r'(?:responsabilidad|responsibility|función|function|tarea|task)',
                r'(?:deber|duty|obligación|obligation)'
            ],
            "skills": [
                r'(?:habilidad|skill|competencia|competency|capacidad|ability)',
                r'(?:conocimiento|knowledge|experiencia|experience)'
            ],
            "culture": [
                r'(?:cultura|culture|valores|values|ambiente|environment)',
                r'(?:equipo|team|colaboración|collaboration)'
            ]
        }

    def analyze_hr_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de recursos humanos.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de HR
        """
        content_lower = content.lower()
        element_counts = {}
        
        # Contar términos de HR
        hr_terms_count = 0
        for lang_words in self.hr_elements["hr_terms"].values():
            hr_terms_count += sum(1 for word in lang_words if word in content_lower)
        element_counts["hr_terms"] = hr_terms_count
        
        # Contar otros elementos de HR
        for element_type, patterns in self.hr_elements.items():
            if element_type != "hr_terms":
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)
                element_counts[element_type] = count
        
        # Calcular score de HR
        total_elements = sum(element_counts.values())
        hr_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de HR
        is_hr = (
            element_counts.get("hr_terms", 0) > 3 or
            element_counts.get("job_requirements", 0) > 0 or
            element_counts.get("benefits", 0) > 0 or
            element_counts.get("responsibilities", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "hr_score": hr_score,
            "total_elements": total_elements,
            "is_hr": is_hr
        }

    def analyze_hr_completeness(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar completitud del contenido de HR.

        Args:
            content: Contenido

        Returns:
            Análisis de completitud de HR
        """
        hr_analysis = self.analyze_hr_content(content)
        element_counts = hr_analysis["element_counts"]
        
        # Verificar elementos de completitud
        has_requirements = element_counts.get("job_requirements", 0) > 0
        has_benefits = element_counts.get("benefits", 0) > 0
        has_responsibilities = element_counts.get("responsibilities", 0) > 0
        has_skills = element_counts.get("skills", 0) > 0
        has_culture = element_counts.get("culture", 0) > 0
        
        # Calcular score de completitud
        completeness_score = (
            (1.0 if has_requirements else 0.0) * 0.25 +
            (1.0 if has_benefits else 0.0) * 0.2 +
            (1.0 if has_responsibilities else 0.0) * 0.25 +
            (1.0 if has_skills else 0.0) * 0.2 +
            (1.0 if has_culture else 0.0) * 0.1
        )
        
        return {
            "completeness_score": completeness_score,
            "has_requirements": has_requirements,
            "has_benefits": has_benefits,
            "has_responsibilities": has_responsibilities,
            "has_skills": has_skills,
            "has_culture": has_culture,
            "completeness_level": (
                "high" if completeness_score > 0.7 else
                "medium" if completeness_score > 0.4 else
                "low"
            )
        }






