"""
Medical Content Analyzer - Sistema de análisis de contenido médico
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class MedicalContentAnalyzer:
    """Analizador de contenido médico"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos médicos
        self.medical_elements = {
            "medical_terms": {
                "es": ["diagnóstico", "tratamiento", "síntoma", "enfermedad", "medicamento", "dosis", "paciente", "clínico"],
                "en": ["diagnosis", "treatment", "symptom", "disease", "medication", "dose", "patient", "clinical"]
            },
            "anatomy": {
                "es": ["órgano", "sistema", "tejido", "célula", "músculo", "hueso", "nervio"],
                "en": ["organ", "system", "tissue", "cell", "muscle", "bone", "nerve"]
            },
            "conditions": [
                r'(?:enfermedad|disease|condición|condition|síndrome|syndrome)',
                r'(?:infección|infection|inflamación|inflammation)'
            ],
            "medications": [
                r'(?:medicamento|medication|fármaco|drug|medicina|medicine)',
                r'(?:mg|ml|dosis|dose)'
            ],
            "warnings": [
                r'(?:advertencia|warning|precaución|caution|contraindicación|contraindication)',
                r'(?:efectos secundarios|side effects|reacción|reaction)'
            ],
            "references": [
                r'(?:consulte|consult|referencia|reference|fuente|source)',
                r'(?:estudio|study|investigación|research|ensayo|trial)'
            ]
        }

    def analyze_medical_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido médico.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido médico
        """
        content_lower = content.lower()
        element_counts = {}
        
        # Contar términos médicos
        medical_terms_count = 0
        for lang_words in self.medical_elements["medical_terms"].values():
            medical_terms_count += sum(1 for word in lang_words if word in content_lower)
        element_counts["medical_terms"] = medical_terms_count
        
        # Contar términos de anatomía
        anatomy_count = 0
        for lang_words in self.medical_elements["anatomy"].values():
            anatomy_count += sum(1 for word in lang_words if word in content_lower)
        element_counts["anatomy"] = anatomy_count
        
        # Contar otros elementos médicos
        for element_type, patterns in self.medical_elements.items():
            if element_type not in ["medical_terms", "anatomy"]:
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)
                element_counts[element_type] = count
        
        # Calcular score médico
        total_elements = sum(element_counts.values())
        medical_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido médico
        is_medical = (
            element_counts.get("medical_terms", 0) > 3 or
            element_counts.get("anatomy", 0) > 2 or
            element_counts.get("conditions", 0) > 0 or
            element_counts.get("medications", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "medical_score": medical_score,
            "total_elements": total_elements,
            "is_medical": is_medical
        }

    def analyze_medical_safety(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar seguridad médica del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de seguridad médica
        """
        medical_analysis = self.analyze_medical_content(content)
        element_counts = medical_analysis["element_counts"]
        
        # Verificar elementos de seguridad
        has_warnings = element_counts.get("warnings", 0) > 0
        has_references = element_counts.get("references", 0) > 0
        has_medications = element_counts.get("medications", 0) > 0
        
        # Calcular score de seguridad
        safety_score = (
            (1.0 if has_warnings else 0.0) * 0.4 +
            (1.0 if has_references else 0.0) * 0.4 +
            (0.5 if has_medications else 1.0) * 0.2  # Medicamentos requieren más cuidado
        )
        
        return {
            "safety_score": safety_score,
            "has_warnings": has_warnings,
            "has_references": has_references,
            "has_medications": has_medications,
            "safety_level": (
                "high" if safety_score > 0.7 else
                "medium" if safety_score > 0.4 else
                "low"
            )
        }






