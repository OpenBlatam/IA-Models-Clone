"""
Contract Models - Modelos de contratos MCP
===========================================
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator


class FrameMetadata(BaseModel):
    """Metadata de un frame de contexto"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(..., description="Fuente del contexto")
    version: str = Field(default="1.0", description="Versión del frame")
    token_count: Optional[int] = Field(None, description="Conteo de tokens")
    encoding: str = Field(default="utf-8", description="Encoding utilizado")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata adicional")


class ContextFrame(BaseModel):
    """
    Frame de contexto estandarizado
    
    Define la estructura de contexto que se pasa entre componentes MCP.
    """
    
    frame_id: str = Field(..., description="ID único del frame")
    content: str = Field(..., description="Contenido del contexto")
    context_type: str = Field(..., description="Tipo de contexto (text, code, data, etc.)")
    
    # Límites y validación
    max_tokens: int = Field(default=4096, description="Límite máximo de tokens")
    token_count: Optional[int] = Field(None, description="Conteo actual de tokens")
    
    # Metadata
    metadata: FrameMetadata = Field(..., description="Metadata del frame")
    
    # Referencias y relaciones
    parent_frame_id: Optional[str] = Field(None, description="ID del frame padre")
    related_frames: List[str] = Field(default_factory=list, description="IDs de frames relacionados")
    
    # Campos adicionales
    fields: Dict[str, Any] = Field(default_factory=dict, description="Campos adicionales")
    
    @validator("content")
    def validate_content(cls, v):
        """Valida que content no esté vacío"""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    @validator("max_tokens")
    def validate_max_tokens(cls, v):
        """Valida límite de tokens"""
        if v < 1 or v > 100000:
            raise ValueError("max_tokens must be between 1 and 100000")
        return v
    
    def estimate_tokens(self) -> int:
        """
        Estima número de tokens (aproximación simple)
        
        Returns:
            Número estimado de tokens
        """
        # Aproximación: 1 token ≈ 4 caracteres
        return len(self.content) // 4
    
    def is_within_limits(self) -> bool:
        """
        Verifica si el frame está dentro de los límites
        
        Returns:
            True si está dentro de límites
        """
        tokens = self.token_count or self.estimate_tokens()
        return tokens <= self.max_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte frame a diccionario"""
        return self.dict(exclude_none=True)


class PromptFrame(BaseModel):
    """
    Frame de prompt estandarizado
    
    Define la estructura de prompts que se envían al modelo.
    """
    
    prompt_id: str = Field(..., description="ID único del prompt")
    system_prompt: Optional[str] = Field(None, description="Prompt del sistema")
    user_prompt: str = Field(..., description="Prompt del usuario")
    
    # Contexto asociado
    context_frames: List[ContextFrame] = Field(
        default_factory=list,
        description="Frames de contexto asociados"
    )
    
    # Configuración
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature")
    max_tokens: int = Field(default=2048, description="Máximo de tokens en respuesta")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    
    # Metadata
    metadata: FrameMetadata = Field(..., description="Metadata del prompt")
    
    # Campos adicionales
    fields: Dict[str, Any] = Field(default_factory=dict, description="Campos adicionales")
    
    @validator("user_prompt")
    def validate_user_prompt(cls, v):
        """Valida que user_prompt no esté vacío"""
        if not v or not v.strip():
            raise ValueError("user_prompt cannot be empty")
        return v.strip()
    
    def get_total_context_tokens(self) -> int:
        """
        Calcula total de tokens en contexto
        
        Returns:
            Total de tokens estimados
        """
        total = 0
        for frame in self.context_frames:
            total += frame.token_count or frame.estimate_tokens()
        return total
    
    def is_within_limits(self) -> bool:
        """
        Verifica si el prompt está dentro de los límites
        
        Returns:
            True si está dentro de límites
        """
        context_tokens = self.get_total_context_tokens()
        return context_tokens + self.max_tokens <= 8192  # Límite típico
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte prompt frame a diccionario"""
        return self.dict(exclude_none=True)

