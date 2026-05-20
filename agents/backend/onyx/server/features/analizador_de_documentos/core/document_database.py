"""
Document Database - Integración con Bases de Datos
===================================================

Integración con bases de datos para persistencia de análisis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Registro de análisis en base de datos."""
    document_id: str
    analysis_type: str
    result: Dict[str, Any]
    quality_score: Optional[float] = None
    grammar_score: Optional[float] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseAdapter(ABC):
    """Adapter base para bases de datos."""
    
    @abstractmethod
    async def save_analysis(self, record: AnalysisRecord) -> bool:
        """Guardar análisis."""
        pass
    
    @abstractmethod
    async def get_analysis(self, document_id: str, analysis_type: Optional[str] = None) -> List[AnalysisRecord]:
        """Obtener análisis."""
        pass
    
    @abstractmethod
    async def delete_analysis(self, document_id: str) -> bool:
        """Eliminar análisis."""
        pass


class InMemoryDatabase(DatabaseAdapter):
    """Base de datos en memoria (para desarrollo/testing)."""
    
    def __init__(self):
        """Inicializar base de datos en memoria."""
        self.storage: Dict[str, List[AnalysisRecord]] = {}
    
    async def save_analysis(self, record: AnalysisRecord) -> bool:
        """Guardar análisis."""
        if record.document_id not in self.storage:
            self.storage[record.document_id] = []
        
        self.storage[record.document_id].append(record)
        return True
    
    async def get_analysis(self, document_id: str, analysis_type: Optional[str] = None) -> List[AnalysisRecord]:
        """Obtener análisis."""
        if document_id not in self.storage:
            return []
        
        records = self.storage[document_id]
        if analysis_type:
            records = [r for r in records if r.analysis_type == analysis_type]
        
        return records
    
    async def delete_analysis(self, document_id: str) -> bool:
        """Eliminar análisis."""
        if document_id in self.storage:
            del self.storage[document_id]
            return True
        return False


class DocumentDatabase:
    """Gestor de base de datos para documentos."""
    
    def __init__(self, analyzer, adapter: Optional[DatabaseAdapter] = None):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.adapter = adapter or InMemoryDatabase()
    
    async def save_analysis_result(
        self,
        document_id: str,
        analysis_result: Any,
        analysis_type: str = "full",
        quality_score: Optional[float] = None,
        grammar_score: Optional[float] = None,
        processing_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Guardar resultado de análisis."""
        # Convertir resultado a dict
        if hasattr(analysis_result, '__dict__'):
            result_dict = analysis_result.__dict__
        elif isinstance(analysis_result, dict):
            result_dict = analysis_result
        else:
            result_dict = {"result": str(analysis_result)}
        
        record = AnalysisRecord(
            document_id=document_id,
            analysis_type=analysis_type,
            result=result_dict,
            quality_score=quality_score,
            grammar_score=grammar_score,
            processing_time=processing_time,
            metadata=metadata or {}
        )
        
        return await self.adapter.save_analysis(record)
    
    async def get_analysis_history(
        self,
        document_id: str,
        analysis_type: Optional[str] = None
    ) -> List[AnalysisRecord]:
        """Obtener historial de análisis."""
        return await self.adapter.get_analysis(document_id, analysis_type)
    
    async def get_latest_analysis(
        self,
        document_id: str,
        analysis_type: Optional[str] = None
    ) -> Optional[AnalysisRecord]:
        """Obtener último análisis."""
        records = await self.get_analysis_history(document_id, analysis_type)
        if records:
            return max(records, key=lambda r: r.timestamp)
        return None
    
    async def delete_document_analyses(self, document_id: str) -> bool:
        """Eliminar todos los análisis de un documento."""
        return await self.adapter.delete_analysis(document_id)
    
    async def search_analyses(
        self,
        query: Dict[str, Any],
        limit: int = 100
    ) -> List[AnalysisRecord]:
        """Buscar análisis (implementación básica)."""
        # En producción, usar query real de base de datos
        # Por ahora retornar lista vacía
        return []


# Funciones helper para adaptadores comunes
def create_sqlite_adapter(db_path: str) -> DatabaseAdapter:
    """Crear adapter SQLite (placeholder)."""
    # En producción, implementar adapter real
    logger.warning("SQLite adapter no implementado, usando in-memory")
    return InMemoryDatabase()


def create_postgresql_adapter(connection_string: str) -> DatabaseAdapter:
    """Crear adapter PostgreSQL (placeholder)."""
    # En producción, implementar adapter real
    logger.warning("PostgreSQL adapter no implementado, usando in-memory")
    return InMemoryDatabase()


def create_mongodb_adapter(connection_string: str) -> DatabaseAdapter:
    """Crear adapter MongoDB (placeholder)."""
    # En producción, implementar adapter real
    logger.warning("MongoDB adapter no implementado, usando in-memory")
    return InMemoryDatabase()


__all__ = [
    "DocumentDatabase",
    "DatabaseAdapter",
    "InMemoryDatabase",
    "AnalysisRecord",
    "create_sqlite_adapter",
    "create_postgresql_adapter",
    "create_mongodb_adapter"
]

