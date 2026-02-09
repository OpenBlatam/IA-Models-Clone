"""
Paper Model - Modelos de datos para papers
===========================================
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PaperSection(BaseModel):
    """Modelo de sección de paper"""
    title: str
    content: str
    type: str  # abstract, introduction, methodology, results, conclusion, etc.
    page: Optional[int] = None


class Paper(BaseModel):
    """Modelo de paper completo"""
    id: Optional[str] = None
    source: str  # pdf, link
    path: Optional[str] = None
    url: Optional[str] = None
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str = ""
    content: str = ""
    sections: List[PaperSection] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




