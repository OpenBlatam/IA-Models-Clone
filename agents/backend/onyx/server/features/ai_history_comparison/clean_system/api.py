"""
API simple y funcional
=====================

Solo los endpoints esenciales para el sistema.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import logging

from .models import HistoryEntry, ComparisonResult, AnalysisJob
from .services import ContentAnalyzer, ModelComparator, QualityAssessor
from .repositories import HistoryRepository, ComparisonRepository

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar servicios
content_analyzer = ContentAnalyzer()
model_comparator = ModelComparator()
quality_assessor = QualityAssessor()
history_repo = HistoryRepository()
comparison_repo = ComparisonRepository()


def create_app() -> FastAPI:
    """Crear aplicación FastAPI."""
    app = FastAPI(
        title="AI History Comparison API",
        description="API simple para comparar historial de IA",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Endpoints
    setup_endpoints(app)
    
    return app


def setup_endpoints(app: FastAPI):
    """Configurar endpoints."""
    
    @app.get("/")
    async def root():
        """Endpoint raíz."""
        return {
            "message": "AI History Comparison API",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "healthy"}
    
    # Análisis de contenido
    @app.post("/analyze")
    async def analyze_content(
        content: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Analizar contenido."""
        try:
            # Analizar contenido
            metrics = content_analyzer.analyze(content)
            
            # Crear entrada de historial
            entry = HistoryEntry.create(
                content=content,
                model_version=model_version,
                metadata=metadata or {},
                **metrics
            )
            
            # Guardar en repositorio
            saved_entry = history_repo.save(entry)
            
            return {
                "success": True,
                "entry": saved_entry.to_dict(),
                "analysis": metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Obtener entrada por ID
    @app.get("/entries/{entry_id}")
    async def get_entry(entry_id: str = Path(..., description="ID de entrada")):
        """Obtener entrada por ID."""
        entry = history_repo.find_by_id(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entrada no encontrada")
        
        return {"entry": entry.to_dict()}
    
    # Buscar entradas
    @app.get("/entries")
    async def search_entries(
        model_version: Optional[str] = Query(None, description="Filtrar por modelo"),
        days: Optional[int] = Query(7, description="Días recientes"),
        limit: Optional[int] = Query(100, description="Límite de resultados"),
        query: Optional[str] = Query(None, description="Buscar en contenido")
    ):
        """Buscar entradas."""
        try:
            if query:
                entries = history_repo.search(query, limit)
            elif model_version:
                entries = history_repo.find_by_model_version(model_version, limit)
            else:
                entries = history_repo.find_recent(days, limit)
            
            return {
                "entries": [entry.to_dict() for entry in entries],
                "total": len(entries)
            }
            
        except Exception as e:
            logger.error(f"Error searching entries: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Comparar modelos
    @app.post("/compare")
    async def compare_models(
        entry1_id: str,
        entry2_id: str
    ):
        """Comparar dos entradas."""
        try:
            # Obtener entradas
            entry1 = history_repo.find_by_id(entry1_id)
            entry2 = history_repo.find_by_id(entry2_id)
            
            if not entry1 or not entry2:
                raise HTTPException(status_code=404, detail="Una o ambas entradas no encontradas")
            
            # Comparar
            result = model_comparator.compare(entry1, entry2)
            
            # Guardar resultado
            saved_result = comparison_repo.save(result)
            
            return {
                "success": True,
                "comparison": saved_result.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Obtener comparaciones
    @app.get("/comparisons")
    async def get_comparisons(
        model_a: Optional[str] = Query(None, description="Modelo A"),
        model_b: Optional[str] = Query(None, description="Modelo B"),
        days: Optional[int] = Query(7, description="Días recientes"),
        limit: Optional[int] = Query(50, description="Límite de resultados")
    ):
        """Obtener comparaciones."""
        try:
            if model_a and model_b:
                comparisons = comparison_repo.find_by_models(model_a, model_b, limit)
            else:
                comparisons = comparison_repo.find_recent(days, limit)
            
            return {
                "comparisons": [comp.to_dict() for comp in comparisons],
                "total": len(comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error getting comparisons: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Evaluar calidad
    @app.get("/entries/{entry_id}/quality")
    async def assess_quality(entry_id: str = Path(..., description="ID de entrada")):
        """Evaluar calidad de entrada."""
        entry = history_repo.find_by_id(entry_id)
        
        if not entry:
            raise HTTPException(status_code=404, detail="Entrada no encontrada")
        
        assessment = quality_assessor.assess(entry)
        
        return {
            "entry_id": entry_id,
            "assessment": assessment
        }
    
    # Estadísticas
    @app.get("/stats")
    async def get_stats():
        """Obtener estadísticas del sistema."""
        try:
            # Estadísticas de entradas por modelo
            entry_counts = history_repo.count_by_model()
            
            # Estadísticas de comparaciones
            comparison_stats = comparison_repo.get_model_stats()
            
            return {
                "entries_by_model": entry_counts,
                "comparison_stats": comparison_stats,
                "total_entries": sum(entry_counts.values()),
                "total_comparisons": len(comparison_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Eliminar entrada
    @app.delete("/entries/{entry_id}")
    async def delete_entry(entry_id: str = Path(..., description="ID de entrada")):
        """Eliminar entrada."""
        deleted = history_repo.delete(entry_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Entrada no encontrada")
        
        return {"message": "Entrada eliminada exitosamente"}


# Crear aplicación
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




