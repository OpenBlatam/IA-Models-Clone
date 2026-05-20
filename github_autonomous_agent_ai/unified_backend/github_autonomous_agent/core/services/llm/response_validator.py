"""
Response Validator - Validación y evaluación de respuestas de LLMs.

Sigue principios de calidad y confiabilidad.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import re

from config.logging_config import get_logger

logger = get_logger(__name__)


class ValidationLevel(str, Enum):
    """Nivel de validación."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


@dataclass
class ValidationResult:
    """Resultado de validación."""
    is_valid: bool
    score: float  # 0.0 - 1.0
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]


class ResponseValidator:
    """
    Validador de respuestas de LLMs.
    
    Características:
    - Validación de formato
    - Detección de contenido problemático
    - Evaluación de calidad
    - Sugerencias de mejora
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.BASIC):
        """
        Inicializar ResponseValidator.
        
        Args:
            validation_level: Nivel de validación
        """
        self.validation_level = validation_level
    
    def validate(
        self,
        response: str,
        expected_format: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        required_keywords: Optional[List[str]] = None,
        forbidden_keywords: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validar una respuesta.
        
        Args:
            response: Respuesta a validar
            expected_format: Formato esperado (json, code, markdown, etc.)
            min_length: Longitud mínima
            max_length: Longitud máxima
            required_keywords: Palabras clave que deben estar presentes
            forbidden_keywords: Palabras clave que no deben estar presentes
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        suggestions = []
        score = 1.0
        
        # Validación básica
        if not response or not response.strip():
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Respuesta vacía"],
                warnings=[],
                suggestions=[]
            )
        
        # Validar longitud
        length = len(response)
        if min_length and length < min_length:
            issues.append(f"Respuesta muy corta: {length} caracteres (mínimo: {min_length})")
            score -= 0.3
        elif max_length and length > max_length:
            warnings.append(f"Respuesta muy larga: {length} caracteres (máximo: {max_length})")
            score -= 0.1
        
        # Validar formato
        if expected_format:
            format_valid = self._validate_format(response, expected_format)
            if not format_valid:
                issues.append(f"Formato inválido: se esperaba {expected_format}")
                score -= 0.2
        
        # Validar palabras clave requeridas
        if required_keywords:
            missing = [kw for kw in required_keywords if kw.lower() not in response.lower()]
            if missing:
                issues.append(f"Palabras clave faltantes: {', '.join(missing)}")
                score -= 0.2 * len(missing) / len(required_keywords)
        
        # Validar palabras clave prohibidas
        if forbidden_keywords:
            found = [kw for kw in forbidden_keywords if kw.lower() in response.lower()]
            if found:
                issues.append(f"Palabras clave prohibidas encontradas: {', '.join(found)}")
                score -= 0.3
        
        # Validaciones adicionales según nivel
        if self.validation_level == ValidationLevel.STRICT:
            # Detectar contenido problemático
            if self._has_incomplete_code(response):
                warnings.append("Código posiblemente incompleto")
                score -= 0.1
            
            if self._has_placeholders(response):
                warnings.append("Posibles placeholders no reemplazados")
                score -= 0.1
            
            if self._has_repetition(response):
                warnings.append("Posible repetición excesiva")
                score -= 0.1
        
        # Sugerencias
        if length < 100:
            suggestions.append("Considera agregar más detalles")
        if not response.endswith(('.', '!', '?', '`')):
            suggestions.append("Considera terminar con puntuación apropiada")
        
        score = max(0.0, min(1.0, score))
        is_valid = len(issues) == 0 and score >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_format(self, response: str, format_type: str) -> bool:
        """Validar formato específico."""
        if format_type == "json":
            try:
                import json
                json.loads(response)
                return True
            except:
                return False
        
        elif format_type == "code":
            # Verificar que tenga bloques de código
            return "```" in response or re.search(r'\b(def|class|function|const|let|var)\b', response)
        
        elif format_type == "markdown":
            # Verificar elementos markdown básicos
            return bool(re.search(r'[#*_`\[\]]', response))
        
        return True
    
    def _has_incomplete_code(self, response: str) -> bool:
        """Detectar código incompleto."""
        # Buscar patrones de código incompleto
        incomplete_patterns = [
            r'def\s+\w+\s*\([^)]*$',  # Función sin cerrar
            r'class\s+\w+[^:]*$',  # Clase sin dos puntos
            r'\{[^}]*$',  # Llave sin cerrar
            r'\[[^\]]*$',  # Corchete sin cerrar
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, response, re.MULTILINE):
                return True
        
        return False
    
    def _has_placeholders(self, response: str) -> bool:
        """Detectar placeholders no reemplazados."""
        placeholder_patterns = [
            r'\{\{.*?\}\}',
            r'\[.*?\]',
            r'<.*?>',
            r'TODO',
            r'FIXME',
            r'XXX'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        
        return False
    
    def _has_repetition(self, response: str) -> bool:
        """Detectar repetición excesiva."""
        sentences = re.split(r'[.!?]\s+', response)
        if len(sentences) < 3:
            return False
        
        # Verificar si hay muchas oraciones similares
        unique_sentences = set(s.lower().strip() for s in sentences)
        repetition_ratio = 1 - (len(unique_sentences) / len(sentences))
        
        return repetition_ratio > 0.5
    
    def evaluate_quality(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluar calidad general de una respuesta.
        
        Args:
            response: Respuesta a evaluar
            context: Contexto adicional (opcional)
            
        Returns:
            Diccionario con métricas de calidad
        """
        metrics = {
            "length": len(response),
            "word_count": len(response.split()),
            "sentence_count": len(re.split(r'[.!?]+', response)),
            "has_code_blocks": "```" in response,
            "has_structure": bool(re.search(r'[#*\-]', response)),  # Headers, lists
            "readability_score": self._calculate_readability(response)
        }
        
        # Validación básica
        validation = self.validate(response)
        metrics["validation_score"] = validation.score
        metrics["validation_issues"] = len(validation.issues)
        metrics["validation_warnings"] = len(validation.warnings)
        
        return metrics
    
    def _calculate_readability(self, text: str) -> float:
        """Calcular score de legibilidad simple."""
        # Score basado en:
        # - Longitud promedio de oraciones
        # - Longitud promedio de palabras
        # - Uso de puntuación
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(w) for w in words) / len(words)
        
        # Score más alto = más legible
        # Penalizar oraciones muy largas o palabras muy largas
        score = 1.0
        if avg_sentence_length > 25:
            score -= 0.2
        if avg_word_length > 6:
            score -= 0.1
        
        return max(0.0, min(1.0, score))


# Instancia global
_validator = ResponseValidator()


def get_validator(validation_level: ValidationLevel = ValidationLevel.BASIC) -> ResponseValidator:
    """Obtener instancia de validador."""
    if validation_level != _validator.validation_level:
        return ResponseValidator(validation_level)
    return _validator



