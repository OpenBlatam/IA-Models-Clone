"""
Token Manager - Gestión avanzada de tokens para LLMs.

Sigue principios de eficiencia y optimización de recursos.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re

from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TokenInfo:
    """Información sobre tokens."""
    estimated_tokens: int
    actual_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class TokenManager:
    """
    Gestor de tokens para LLMs.
    
    Características:
    - Estimación precisa de tokens
    - Validación de límites
    - Optimización de prompts
    - Tracking de uso
    """
    
    # Estimaciones por tipo de contenido
    CHARS_PER_TOKEN_TEXT = 4  # Texto general
    CHARS_PER_TOKEN_CODE = 3  # Código (más denso)
    CHARS_PER_TOKEN_WHITESPACE = 6  # Espacios en blanco
    
    # Límites comunes por modelo (pueden ser configurados)
    MODEL_LIMITS: Dict[str, Dict[str, int]] = {
        "gpt-4o": {"max_tokens": 128000, "max_output": 16384},
        "gpt-4o-mini": {"max_tokens": 128000, "max_output": 16384},
        "gpt-4": {"max_tokens": 8192, "max_output": 4096},
        "gpt-3.5-turbo": {"max_tokens": 16385, "max_output": 4096},
        "claude-3.5-sonnet": {"max_tokens": 200000, "max_output": 8192},
        "claude-3-opus": {"max_tokens": 200000, "max_output": 8192},
        "gemini-pro-1.5": {"max_tokens": 1000000, "max_output": 8192},
    }
    
    def __init__(self):
        """Inicializar TokenManager."""
        self.usage_stats: Dict[str, List[TokenInfo]] = defaultdict(list)
    
    def estimate_tokens(
        self,
        text: str,
        content_type: str = "text"
    ) -> int:
        """
        Estimar número de tokens en un texto.
        
        Args:
            text: Texto a analizar
            content_type: Tipo de contenido (text, code, mixed)
            
        Returns:
            Número estimado de tokens
        """
        if not text:
            return 0
        
        # Ajustar estimación según tipo de contenido
        if content_type == "code":
            chars_per_token = self.CHARS_PER_TOKEN_CODE
        elif content_type == "mixed":
            # Mezcla: calcular proporción
            code_ratio = len(re.findall(r'[{}();=]', text)) / max(len(text), 1)
            chars_per_token = (
                self.CHARS_PER_TOKEN_CODE * code_ratio +
                self.CHARS_PER_TOKEN_TEXT * (1 - code_ratio)
            )
        else:
            chars_per_token = self.CHARS_PER_TOKEN_TEXT
        
        # Contar espacios (menos tokens)
        whitespace_count = len(re.findall(r'\s+', text))
        text_chars = len(text) - whitespace_count
        
        # Calcular estimación
        base_tokens = text_chars / chars_per_token
        whitespace_tokens = whitespace_count / self.CHARS_PER_TOKEN_WHITESPACE
        
        return int(base_tokens + whitespace_tokens)
    
    def estimate_messages_tokens(
        self,
        messages: List[Dict[str, str]]
    ) -> int:
        """
        Estimar tokens en una lista de mensajes.
        
        Args:
            messages: Lista de mensajes
            
        Returns:
            Número estimado de tokens
        """
        total = 0
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Tokens de formato (role, separadores, etc.)
            format_tokens = 4  # Aproximado
            
            # Tokens del contenido
            content_type = "code" if "```" in content else "text"
            content_tokens = self.estimate_tokens(content, content_type)
            
            total += format_tokens + content_tokens
        
        return total
    
    def get_model_limits(
        self,
        model: str
    ) -> Dict[str, int]:
        """
        Obtener límites de tokens para un modelo.
        
        Args:
            model: Nombre del modelo
            
        Returns:
            Diccionario con límites
        """
        # Buscar modelo en límites conocidos
        for model_key, limits in self.MODEL_LIMITS.items():
            if model_key in model.lower():
                return limits
        
        # Límites por defecto conservadores
        return {
            "max_tokens": 4096,
            "max_output": 2048
        }
    
    def validate_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None
    ) -> Tuple[bool, Optional[str], TokenInfo]:
        """
        Validar request antes de enviarlo.
        
        Args:
            model: Modelo a usar
            messages: Mensajes del request
            max_tokens: Máximo de tokens de salida solicitados
            
        Returns:
            Tupla (es_válido, mensaje_error, token_info)
        """
        limits = self.get_model_limits(model)
        estimated_prompt = self.estimate_messages_tokens(messages)
        
        # Validar prompt no exceda límites
        if estimated_prompt > limits["max_tokens"]:
            return (
                False,
                f"Prompt muy grande: {estimated_prompt} tokens (límite: {limits['max_tokens']})",
                TokenInfo(estimated_tokens=estimated_prompt)
            )
        
        # Validar max_tokens si se proporciona
        if max_tokens:
            if max_tokens > limits["max_output"]:
                return (
                    False,
                    f"max_tokens ({max_tokens}) excede límite del modelo ({limits['max_output']})",
                    TokenInfo(estimated_tokens=estimated_prompt)
                )
            
            # Validar que prompt + max_tokens no exceda límite total
            if estimated_prompt + max_tokens > limits["max_tokens"]:
                suggested_max = limits["max_tokens"] - estimated_prompt - 100  # Buffer
                return (
                    False,
                    f"Prompt ({estimated_prompt}) + max_tokens ({max_tokens}) excede límite. Sugerencia: max_tokens={suggested_max}",
                    TokenInfo(estimated_tokens=estimated_prompt)
                )
        
        return (
            True,
            None,
            TokenInfo(estimated_tokens=estimated_prompt)
        )
    
    def optimize_prompt(
        self,
        prompt: str,
        max_tokens: int,
        target_reduction: float = 0.2
    ) -> str:
        """
        Optimizar prompt reduciendo tokens si es necesario.
        
        Args:
            prompt: Prompt original
            max_tokens: Límite de tokens
            target_reduction: Porcentaje de reducción objetivo
            
        Returns:
            Prompt optimizado
        """
        current_tokens = self.estimate_tokens(prompt)
        
        if current_tokens <= max_tokens:
            return prompt
        
        # Estrategias de optimización
        optimized = prompt
        
        # 1. Eliminar espacios múltiples
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # 2. Eliminar líneas vacías múltiples
        optimized = re.sub(r'\n\s*\n\s*\n', '\n\n', optimized)
        
        # 3. Acortar comentarios largos
        optimized = re.sub(
            r'# (.{50,})',
            lambda m: f"# {m.group(1)[:50]}...",
            optimized
        )
        
        new_tokens = self.estimate_tokens(optimized)
        reduction = (current_tokens - new_tokens) / current_tokens
        
        if reduction >= target_reduction:
            logger.info(f"Prompt optimizado: {current_tokens} -> {new_tokens} tokens ({reduction*100:.1f}% reducción)")
            return optimized
        
        return prompt
    
    def record_usage(
        self,
        model: str,
        token_info: TokenInfo
    ) -> None:
        """Registrar uso de tokens para estadísticas."""
        self.usage_stats[model].append(token_info)
        
        # Mantener solo últimos 1000 registros por modelo
        if len(self.usage_stats[model]) > 1000:
            self.usage_stats[model] = self.usage_stats[model][-1000:]
    
    def get_usage_stats(
        self,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener estadísticas de uso de tokens.
        
        Args:
            model: Modelo específico (opcional)
            
        Returns:
            Estadísticas de uso
        """
        if model:
            stats = self.usage_stats.get(model, [])
        else:
            stats = [s for stats_list in self.usage_stats.values() for s in stats_list]
        
        if not stats:
            return {
                "total_requests": 0,
                "average_tokens": 0,
                "total_tokens": 0
            }
        
        total_tokens = sum(s.actual_tokens or s.estimated_tokens for s in stats)
        avg_tokens = total_tokens / len(stats)
        
        return {
            "total_requests": len(stats),
            "average_tokens": int(avg_tokens),
            "total_tokens": total_tokens,
            "min_tokens": min(s.actual_tokens or s.estimated_tokens for s in stats),
            "max_tokens": max(s.actual_tokens or s.estimated_tokens for s in stats)
        }


# Instancia global
_token_manager = TokenManager()


def get_token_manager() -> TokenManager:
    """Obtener instancia global de TokenManager."""
    return _token_manager



