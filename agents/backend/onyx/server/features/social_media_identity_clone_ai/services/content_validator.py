"""
Sistema de validación de contenido generado
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..core.models import GeneratedContent, Platform, ContentType

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación"""
    is_valid: bool
    score: float  # 0.0 - 1.0
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]


class ContentValidator:
    """Validador de contenido generado"""
    
    def __init__(self):
        self.platform_limits = {
            Platform.INSTAGRAM: {
                "max_length": 2200,
                "min_length": 1,
                "max_hashtags": 30,
                "max_mentions": 20
            },
            Platform.TIKTOK: {
                "max_length": 2200,
                "min_length": 1,
                "max_hashtags": 100,
                "max_mentions": 0  # TikTok no tiene menciones en captions
            },
            Platform.YOUTUBE: {
                "max_length": 5000,
                "min_length": 1,
                "max_hashtags": 15,
                "max_mentions": 0
            }
        }
    
    def validate(self, content: GeneratedContent) -> ValidationResult:
        """
        Valida contenido generado
        
        Args:
            content: Contenido a validar
            
        Returns:
            ValidationResult
        """
        issues = []
        warnings = []
        suggestions = []
        score = 1.0
        
        platform = content.platform
        limits = self.platform_limits.get(platform, {})
        
        # Validar longitud
        content_length = len(content.content)
        if limits.get("max_length") and content_length > limits["max_length"]:
            issues.append(f"Contenido muy largo ({content_length} chars, máximo: {limits['max_length']})")
            score -= 0.3
        elif limits.get("min_length") and content_length < limits["min_length"]:
            issues.append(f"Contenido muy corto ({content_length} chars, mínimo: {limits['min_length']})")
            score -= 0.2
        
        # Validar hashtags
        hashtag_count = len(content.hashtags)
        max_hashtags = limits.get("max_hashtags", 0)
        if max_hashtags > 0:
            if hashtag_count > max_hashtags:
                issues.append(f"Demasiados hashtags ({hashtag_count}, máximo: {max_hashtags})")
                score -= 0.2
            elif hashtag_count == 0:
                warnings.append("No hay hashtags. Considera agregar algunos para mejor alcance")
                score -= 0.1
            elif hashtag_count < 5:
                suggestions.append(f"Considera agregar más hashtags (actual: {hashtag_count}, recomendado: 5-{max_hashtags})")
        
        # Validar menciones
        mentions = self._extract_mentions(content.content)
        mention_count = len(mentions)
        max_mentions = limits.get("max_mentions", 0)
        if max_mentions > 0 and mention_count > max_mentions:
            issues.append(f"Demasiadas menciones ({mention_count}, máximo: {max_mentions})")
            score -= 0.1
        
        # Validar caracteres especiales
        if self._has_invalid_characters(content.content):
            warnings.append("Contenido contiene caracteres especiales que podrían no renderizarse correctamente")
            score -= 0.05
        
        # Validar estructura
        if platform == Platform.INSTAGRAM:
            if not self._has_line_breaks(content.content):
                suggestions.append("Considera agregar saltos de línea para mejor legibilidad en Instagram")
        
        # Validar engagement hooks
        if not self._has_engagement_hook(content.content):
            suggestions.append("Considera agregar un hook o pregunta para aumentar engagement")
        
        # Validar emojis
        emoji_count = self._count_emojis(content.content)
        if emoji_count == 0:
            suggestions.append("Considera agregar emojis para hacer el contenido más atractivo")
        elif emoji_count > 10:
            warnings.append(f"Muchos emojis ({emoji_count}), podría verse spam")
            score -= 0.05
        
        # Asegurar score entre 0 y 1
        score = max(0.0, min(1.0, score))
        
        is_valid = len(issues) == 0 and score >= 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extrae menciones del texto"""
        mentions = re.findall(r'@\w+', text)
        return list(set(mentions))
    
    def _has_invalid_characters(self, text: str) -> bool:
        """Verifica caracteres inválidos"""
        # Caracteres de control (excepto newlines y tabs)
        invalid_pattern = r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]'
        return bool(re.search(invalid_pattern, text))
    
    def _has_line_breaks(self, text: str) -> bool:
        """Verifica si tiene saltos de línea"""
        return '\n' in text or '\r' in text
    
    def _has_engagement_hook(self, text: str) -> bool:
        """Verifica si tiene hook de engagement"""
        hooks = [
            r'\?',  # Preguntas
            r'¿',   # Preguntas en español
            r'!',   # Exclamaciones
            r'comenta', r'comenta',
            r'dime', r'dime',
            r'cuéntame', r'cuentame',
            r'qué opinas', r'que opinas'
        ]
        text_lower = text.lower()
        return any(re.search(hook, text_lower) for hook in hooks)
    
    def _count_emojis(self, text: str) -> int:
        """Cuenta emojis en el texto"""
        # Patrón simple para emojis comunes
        emoji_pattern = r'[😀-🙏🌀-🗿]'
        return len(re.findall(emoji_pattern, text))
    
    def get_validation_report(self, content: GeneratedContent) -> Dict[str, Any]:
        """Obtiene reporte completo de validación"""
        result = self.validate(content)
        
        return {
            "content_id": content.content_id,
            "platform": content.platform.value,
            "content_type": content.content_type.value,
            "validation": {
                "is_valid": result.is_valid,
                "score": result.score,
                "issues": result.issues,
                "warnings": result.warnings,
                "suggestions": result.suggestions
            },
            "stats": {
                "content_length": len(content.content),
                "hashtag_count": len(content.hashtags),
                "word_count": len(content.content.split()),
                "emoji_count": self._count_emojis(content.content)
            }
        }




