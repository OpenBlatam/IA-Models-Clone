"""
Modelos de datos para el procesador de documentos AI
===================================================

Define las estructuras de datos utilizadas en el sistema de procesamiento de documentos.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class DocumentType(str, Enum):
    """Tipos de documentos soportados"""
    MARKDOWN = "markdown"
    PDF = "pdf"
    WORD = "word"
    TEXT = "text"
    UNKNOWN = "unknown"

class DocumentArea(str, Enum):
    """Áreas de conocimiento de documentos"""
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    ACADEMIC = "academic"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCE = "finance"
    MARKETING = "marketing"
    EDUCATION = "education"
    GENERAL = "general"

class DocumentCategory(str, Enum):
    """Categorías específicas de documentos"""
    # Business
    BUSINESS_PLAN = "business_plan"
    CONSULTANCY_REPORT = "consultancy_report"
    MARKETING_STRATEGY = "marketing_strategy"
    FINANCIAL_ANALYSIS = "financial_analysis"
    
    # Technology
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    SOFTWARE_SPECIFICATION = "software_specification"
    SYSTEM_ARCHITECTURE = "system_architecture"
    API_DOCUMENTATION = "api_documentation"
    
    # Academic
    RESEARCH_PAPER = "research_paper"
    THESIS = "thesis"
    ACADEMIC_REPORT = "academic_report"
    LITERATURE_REVIEW = "literature_review"
    
    # Legal
    CONTRACT = "contract"
    LEGAL_OPINION = "legal_opinion"
    POLICY_DOCUMENT = "policy_document"
    COMPLIANCE_REPORT = "compliance_report"
    
    # General
    MANUAL = "manual"
    GUIDE = "guide"
    PRESENTATION = "presentation"
    PROPOSAL = "proposal"

class ProfessionalFormat(str, Enum):
    """Formatos de documentos profesionales"""
    CONSULTANCY = "consultancy"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    COMMERCIAL = "commercial"
    LEGAL = "legal"

class DocumentAnalysis(BaseModel):
    """Análisis completo de un documento"""
    filename: str
    document_type: DocumentType
    area: DocumentArea
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza en la clasificación")
    language: str = "es"
    word_count: int = 0
    page_count: int = 0
    key_topics: List[str] = []
    summary: str = ""
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)

class ProfessionalDocument(BaseModel):
    """Documento profesional transformado"""
    title: str
    format: ProfessionalFormat
    language: str
    content: str
    structure: Dict[str, Any] = {}
    sections: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    original_analysis: Optional[DocumentAnalysis] = None
    created_at: datetime = Field(default_factory=datetime.now)

class DocumentProcessingRequest(BaseModel):
    """Solicitud de procesamiento de documento"""
    filename: str
    target_format: ProfessionalFormat = ProfessionalFormat.CONSULTANCY
    language: str = "es"
    include_analysis: bool = True
    custom_instructions: Optional[str] = None

class DocumentProcessingResponse(BaseModel):
    """Respuesta del procesamiento de documento"""
    success: bool
    message: str
    analysis: Optional[DocumentAnalysis] = None
    professional_document: Optional[ProfessionalDocument] = None
    processing_time: float = 0.0
    errors: List[str] = []

class FileInfo(BaseModel):
    """Información básica de un archivo"""
    filename: str
    size: int
    content_type: str
    extension: str
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

class TextExtractionResult(BaseModel):
    """Resultado de extracción de texto"""
    text: str
    metadata: Dict[str, Any] = {}
    extraction_method: str
    confidence: float = Field(ge=0.0, le=1.0)
    language_detected: Optional[str] = None
    word_count: int = 0
    character_count: int = 0

class ClassificationResult(BaseModel):
    """Resultado de clasificación de documento"""
    document_type: DocumentType
    area: DocumentArea
    category: DocumentCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    alternative_classifications: List[Dict[str, Any]] = []

class TransformationResult(BaseModel):
    """Resultado de transformación profesional"""
    success: bool
    professional_document: ProfessionalDocument
    transformation_method: str
    quality_score: float = Field(ge=0.0, le=1.0)
    improvements_made: List[str] = []
    warnings: List[str] = []

class TemplateInfo(BaseModel):
    """Información de plantilla profesional"""
    name: str
    format: ProfessionalFormat
    description: str
    sections: List[str]
    required_fields: List[str]
    optional_fields: List[str]
    example_content: str
    language: str = "es"

class ProcessingStats(BaseModel):
    """Estadísticas de procesamiento"""
    total_documents_processed: int = 0
    successful_transformations: int = 0
    failed_transformations: int = 0
    average_processing_time: float = 0.0
    most_common_formats: List[Dict[str, Any]] = []
    most_common_areas: List[Dict[str, Any]] = []
    last_updated: datetime = Field(default_factory=datetime.now)


