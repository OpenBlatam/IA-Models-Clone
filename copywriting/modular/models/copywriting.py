from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import uuid
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from .enums import ToneType, LanguageType, UseCaseType
from typing import Any, List, Dict, Optional
import logging
import asyncio
# -*- coding: utf-8 -*-
"""
Copywriting Models - Modelos de datos para copywriting
======================================================

Modelos de datos estructurados para requests y responses.
"""



@dataclass
class CopywritingRequest:
    """
    Request de copywriting con validación completa
    
    Attributes:
        prompt: Texto del prompt principal
        tone: Tono del contenido (profesional, casual, etc.)
        language: Idioma del contenido
        use_case: Caso de uso específico
        target_length: Longitud objetivo en palabras (opcional)
        keywords: Lista de palabras clave a incluir
        use_cache: Si usar cache para esta request
        request_id: ID único de la request
        timestamp: Timestamp de creación
    """
    prompt: str
    tone: str = ToneType.PROFESSIONAL.value
    language: str = LanguageType.SPANISH.value
    use_case: str = UseCaseType.GENERAL.value
    target_length: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    use_cache: bool = True
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> Any:
        """Validación de datos post-inicialización"""
        self._validate_prompt()
        self._validate_tone()
        self._validate_language()
        self._validate_target_length()
    
    def _validate_prompt(self) -> bool:
        """Validar que el prompt no esté vacío"""
        if not self.prompt or len(self.prompt.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        
        if len(self.prompt) > 1000:
            raise ValueError("Prompt too long (max 1000 characters)")
    
    def _validate_tone(self) -> bool:
        """Validar que el tono sea válido"""
        valid_tones = [tone.value for tone in ToneType]
        if self.tone not in valid_tones:
            raise ValueError(f"Invalid tone '{self.tone}'. Must be one of: {valid_tones}")
    
    def _validate_language(self) -> bool:
        """Validar que el idioma sea válido"""
        valid_languages = [lang.value for lang in LanguageType]
        if self.language not in valid_languages:
            raise ValueError(f"Invalid language '{self.language}'. Must be one of: {valid_languages}")
    
    def _validate_target_length(self) -> Optional[Dict[str, Any]]:
        """Validar longitud objetivo"""
        if self.target_length is not None:
            if self.target_length <= 0:
                raise ValueError("Target length must be positive")
            if self.target_length > 1000:
                raise ValueError("Target length too large (max 1000 words)")
    
    def to_cache_key(self) -> str:
        """Generar clave de cache única para esta request"""
        key_components = [
            self.prompt,
            self.tone,
            self.language,
            self.use_case,
            str(self.target_length) if self.target_length else "none",
            "|".join(sorted(self.keywords)) if self.keywords else "none"
        ]
        return "|".join(key_components)

@dataclass 
class CopywritingResponse:
    """
    Response de copywriting con metadata completa
    
    Attributes:
        content: Contenido generado
        request_id: ID de la request asociada
        generation_time_ms: Tiempo de generación en milisegundos
        cache_hit: Si el resultado vino del cache
        optimization_score: Score de optimización del sistema
        performance_tier: Tier de performance actual
        word_count: Número de palabras del contenido
        character_count: Número de caracteres del contenido
        compression_ratio: Ratio de compresión (opcional)
        timestamp: Timestamp de generación
    """
    content: str
    request_id: str
    generation_time_ms: float
    cache_hit: bool
    optimization_score: float
    performance_tier: str
    word_count: int
    character_count: int
    compression_ratio: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> Any:
        """Calcular métricas automáticamente"""
        if self.word_count == 0:
            self.word_count = len(self.content.split())
        if self.character_count == 0:
            self.character_count = len(self.content)
    
    def get_summary(self) -> dict:
        """Obtener resumen de la response"""
        return {
            "request_id": self.request_id,
            "content_preview": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "generation_time_ms": self.generation_time_ms,
            "cache_hit": self.cache_hit,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "optimization_score": self.optimization_score,
            "performance_tier": self.performance_tier
        } 