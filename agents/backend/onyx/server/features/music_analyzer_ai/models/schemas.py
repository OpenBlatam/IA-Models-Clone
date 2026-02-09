"""
Esquemas de datos para el análisis musical
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class TrackSearchRequest(BaseModel):
    """Request para buscar una canción"""
    query: str = Field(..., description="Nombre de la canción o artista")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Número máximo de resultados")


class TrackAnalysisRequest(BaseModel):
    """Request para analizar una canción"""
    track_id: Optional[str] = Field(None, description="ID de la canción en Spotify")
    track_name: Optional[str] = Field(None, description="Nombre de la canción para buscar")
    include_coaching: bool = Field(True, description="Incluir análisis de coaching")


class TrackBasicInfo(BaseModel):
    """Información básica de una canción"""
    name: str
    artists: List[str]
    album: str
    duration_ms: int
    duration_seconds: float
    popularity: int
    release_date: str
    external_urls: Dict[str, str]
    preview_url: Optional[str] = None


class MusicalAnalysis(BaseModel):
    """Análisis musical"""
    key_signature: str
    root_note: str
    mode: str
    tempo: Dict[str, Any]
    time_signature: str
    key_changes: List[Dict[str, Any]]
    tempo_changes: List[Dict[str, Any]]
    structure: Dict[str, Any]
    scale: Dict[str, Any]


class TechnicalAnalysis(BaseModel):
    """Análisis técnico"""
    energy: Dict[str, Any]
    danceability: Dict[str, Any]
    valence: Dict[str, Any]
    acousticness: Dict[str, Any]
    instrumentalness: Dict[str, Any]
    liveness: Dict[str, Any]
    speechiness: Dict[str, Any]
    loudness: Dict[str, Any]
    rhythm_structure: Dict[str, Any]


class CompositionAnalysis(BaseModel):
    """Análisis de composición"""
    structure: List[Dict[str, Any]]
    harmonic_progressions: List[Dict[str, Any]]
    composition_style: str
    complexity: Dict[str, Any]


class PerformanceAnalysis(BaseModel):
    """Análisis de interpretación"""
    timbre_analysis: List[Dict[str, Any]]
    dynamic_range: Dict[str, Any]
    performance_characteristics: List[str]


class EducationalInsights(BaseModel):
    """Insights educativos"""
    key_analysis: Dict[str, Any]
    tempo_analysis: Dict[str, Any]
    learning_points: List[str]
    practice_suggestions: List[str]


class MusicAnalysisResponse(BaseModel):
    """Respuesta completa de análisis musical"""
    success: bool
    track_basic_info: TrackBasicInfo
    musical_analysis: MusicalAnalysis
    technical_analysis: TechnicalAnalysis
    composition_analysis: CompositionAnalysis
    performance_analysis: PerformanceAnalysis
    educational_insights: EducationalInsights
    timestamp: datetime = Field(default_factory=datetime.now)


class CoachingOverview(BaseModel):
    """Resumen de coaching"""
    summary: str
    key_findings: List[str]
    difficulty_level: str
    suitable_for: List[str]


class LearningPathStep(BaseModel):
    """Paso en la ruta de aprendizaje"""
    step: int
    title: str
    description: str
    duration: str


class PracticeExercise(BaseModel):
    """Ejercicio de práctica"""
    type: str
    title: str
    description: str
    tempo: int
    repetitions: int


class CoachingAnalysisResponse(BaseModel):
    """Respuesta de análisis de coaching"""
    success: bool
    overview: CoachingOverview
    technical_breakdown: Dict[str, Any]
    learning_path: List[LearningPathStep]
    practice_exercises: List[PracticeExercise]
    composition_insights: Dict[str, Any]
    performance_tips: List[str]
    recommendations: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Respuesta de error"""
    success: bool = False
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


