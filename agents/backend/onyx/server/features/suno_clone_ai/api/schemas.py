"""
Schemas Pydantic para validación de requests y responses
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

try:
    from utils.validators import InputValidators
    from config.settings import settings
except ImportError:
    from .utils.validators import InputValidators
    from .config.settings import settings


class ChatMessage(BaseModel):
    """Mensaje de chat del usuario"""
    message: str = Field(..., min_length=1, max_length=500, description="Mensaje del usuario")
    user_id: Optional[str] = Field(None, max_length=100, description="ID del usuario")
    chat_history: Optional[List[Dict[str, Any]]] = Field(None, max_length=50, description="Historial de conversación")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        return InputValidators.validate_prompt(v, max_length=500)


class SongGenerationRequest(BaseModel):
    """Request para generación de canción"""
    prompt: str = Field(..., min_length=1, max_length=500, description="Descripción de la canción")
    duration: Optional[int] = Field(None, ge=1, le=300, description="Duración en segundos")
    genre: Optional[str] = Field(None, max_length=50, description="Género musical")
    mood: Optional[str] = Field(None, max_length=50, description="Estado de ánimo")
    user_id: Optional[str] = Field(None, max_length=100, description="ID del usuario")
    
    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        return InputValidators.validate_prompt(v, max_length=500)
    
    @field_validator('duration')
    @classmethod
    def validate_duration(cls, v: Optional[int]) -> Optional[int]:
        if v is not None:
            return InputValidators.validate_duration(v, max_duration=settings.max_audio_length)
        return v
    
    @field_validator('genre')
    @classmethod
    def validate_genre(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return InputValidators.validate_genre(v)
        return v


class SongResponse(BaseModel):
    """Response con información de la canción generada"""
    song_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioEditRequest(BaseModel):
    """Request para edición de audio"""
    song_id: str = Field(..., min_length=1, description="ID de la canción a editar")
    operations: List[Dict[str, Any]] = Field(default_factory=list, max_length=20, description="Operaciones a aplicar")
    fade_in: Optional[float] = Field(None, ge=0, le=10, description="Fade in en segundos")
    fade_out: Optional[float] = Field(None, ge=0, le=10, description="Fade out en segundos")
    normalize: Optional[bool] = Field(True, description="Normalizar audio")
    trim_silence: Optional[bool] = Field(False, description="Eliminar silencio")
    
    @field_validator('song_id')
    @classmethod
    def validate_song_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Song ID cannot be empty")
        return v.strip()
    
    @field_validator('fade_in', 'fade_out')
    @classmethod
    def validate_fade(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            return InputValidators.validate_fade_time(v, max_time=10.0)
        return v


class AudioMixRequest(BaseModel):
    """Request para mezclar múltiples canciones"""
    song_ids: List[str] = Field(..., min_length=1, max_length=10, description="IDs de canciones a mezclar")
    volumes: Optional[List[float]] = Field(None, description="Volúmenes para cada canción")
    
    @field_validator('song_ids')
    @classmethod
    def validate_song_ids(cls, v: List[str]) -> List[str]:
        return InputValidators.validate_song_ids(v, max_count=10)
    
    @field_validator('volumes')
    @classmethod
    def validate_volumes(cls, v: Optional[List[float]], info) -> Optional[List[float]]:
        if v is not None:
            # Obtener song_ids del contexto de validación
            song_ids = info.data.get('song_ids', []) if hasattr(info, 'data') else []
            if song_ids:
                return InputValidators.validate_volumes(v, len(song_ids))
        return v


class SongListResponse(BaseModel):
    """Response para lista de canciones con paginación"""
    songs: List[Dict[str, Any]]
    total: int
    limit: Optional[int] = Field(None, description="Límite de resultados")
    offset: Optional[int] = Field(None, description="Offset para paginación")
    has_more: Optional[bool] = Field(None, description="Indica si hay más resultados disponibles")


class SongAnalysisResponse(BaseModel):
    """Response para análisis de canción"""
    song_id: str
    analysis: Dict[str, Any]
    metadata: Dict[str, Any]


class StatusResponse(BaseModel):
    """Response para estado de generación"""
    status: str
    song_id: str
    message: str
    progress: Optional[Dict[str, Any]] = Field(None, description="Información de progreso opcional")


class BatchStatusResponse(BaseModel):
    """Response para estado de múltiples generaciones"""
    total_requested: int
    found: int
    not_found: int
    statuses: Dict[str, Dict[str, Any]]


class ChatResponse(BaseModel):
    """Response para mensaje de chat"""
    message: str
    user_id: Optional[str] = None
    response: str
    song_info: Optional[Dict[str, Any]] = None
    should_generate: bool = False
