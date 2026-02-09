"""
API Schemas - Esquemas Pydantic para la API
===========================================
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field


class PaperUploadRequest(BaseModel):
    """Request para subir PDF"""
    pass  # Se maneja con FileUpload


class PaperLinkRequest(BaseModel):
    """Request para procesar link"""
    url: HttpUrl = Field(..., description="URL del paper")
    download: bool = Field(True, description="Descargar PDF automáticamente")


class PaperResponse(BaseModel):
    """Response con información del paper"""
    source: str
    title: str
    authors: List[str]
    abstract: str
    sections_count: int
    content_length: int
    metadata: Dict[str, Any]


class TrainingRequest(BaseModel):
    """Request para entrenar modelo"""
    paper_ids: Optional[List[str]] = Field(None, description="IDs de papers a usar")
    model_name: str = Field("gpt-4", description="Nombre del modelo base")
    epochs: int = Field(3, ge=1, le=10, description="Número de épocas")
    use_all_papers: bool = Field(True, description="Usar todos los papers disponibles")


class TrainingResponse(BaseModel):
    """Response del entrenamiento"""
    model_id: str
    status: str
    papers_count: int
    training_examples: int
    epochs: int
    model_path: str


class CodeImproveRequest(BaseModel):
    """Request para mejorar código"""
    github_repo: str = Field(..., description="Repositorio en formato 'owner/repo'")
    file_path: str = Field(..., description="Ruta al archivo en el repositorio")
    branch: Optional[str] = Field(None, description="Rama del repositorio")
    model_id: Optional[str] = Field(None, description="ID del modelo a usar")


class CodeImproveResponse(BaseModel):
    """Response con código mejorado"""
    original_code: str
    improved_code: str
    suggestions: List[Dict[str, Any]]
    repo: str
    file_path: str
    improvements_applied: int


class RepositoryAnalyzeRequest(BaseModel):
    """Request para analizar repositorio completo"""
    github_repo: str = Field(..., description="Repositorio en formato 'owner/repo'")
    branch: Optional[str] = Field(None, description="Rama del repositorio")
    model_id: Optional[str] = Field(None, description="ID del modelo a usar")
    max_files: int = Field(10, ge=1, le=50, description="Máximo número de archivos a analizar")


class RepositoryAnalyzeResponse(BaseModel):
    """Response del análisis de repositorio"""
    repo: str
    files_analyzed: int
    total_improvements: int
    improvements: List[Dict[str, Any]]


class ModelStatusResponse(BaseModel):
    """Response del estado del modelo"""
    model_id: str
    status: str
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None




