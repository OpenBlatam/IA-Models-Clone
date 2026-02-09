"""
API Routes - Endpoints de la API
=================================
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from .schemas import (
    PaperLinkRequest,
    PaperResponse,
    TrainingRequest,
    TrainingResponse,
    CodeImproveRequest,
    CodeImproveResponse,
    RepositoryAnalyzeRequest,
    RepositoryAnalyzeResponse,
    ModelStatusResponse,
)
from .decorators import handle_api_errors
from .helpers import resolve_model_path, create_code_improver

from ..core.paper_extractor import PaperExtractor
from ..core.model_trainer import ModelTrainer
from ..core.vector_store import VectorStore
from ..core.core_utils import get_paper_storage

# Verificar disponibilidad de módulos
try:
    from ..core.rag_engine import RAGEngine
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from ..core.cache_manager import CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from ..core.code_analyzer import CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/research-paper-code-improver", tags=["Research Paper Code Improver"])

# Instancias globales (en producción usar dependency injection)
paper_extractor = PaperExtractor()
model_trainer = ModelTrainer()
vector_store = VectorStore()
paper_storage = get_paper_storage()


@router.post("/papers/upload", response_model=PaperResponse)
@handle_api_errors()
async def upload_paper(file: UploadFile = File(...)):
    """
    Sube y procesa un PDF de paper.
    
    Args:
        file: Archivo PDF
        
    Returns:
        Información extraída del paper
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # Extraer información del PDF
        paper_data = paper_extractor.extract_from_pdf(tmp_path, save_to_storage=True)
        
        # Indexar en vector store
        paper_id = paper_data.get("id")
        if paper_id:
            vector_store.add_paper(paper_id, paper_data)
        
        return PaperResponse(
            source=paper_data.get("source", "pdf"),
            title=paper_data.get("title", ""),
            authors=paper_data.get("authors", []),
            abstract=paper_data.get("abstract", ""),
            sections_count=len(paper_data.get("sections", [])),
            content_length=len(paper_data.get("content", "")),
            metadata=paper_data.get("metadata", {})
        )
    finally:
        # Limpiar archivo temporal
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/papers/link", response_model=PaperResponse)
@handle_api_errors()
async def process_link(request: PaperLinkRequest):
    """
    Procesa un paper desde una URL.
    
    Args:
        request: Request con URL del paper
        
    Returns:
        Información extraída del paper
    """
    # Extraer información del link
    paper_data = paper_extractor.extract_from_link(str(request.url))
    
    # Guardar en almacenamiento
    try:
        paper_id = paper_storage.save_paper(paper_data)
        paper_data["id"] = paper_id
        
        # Indexar en vector store
        vector_store.add_paper(paper_id, paper_data)
    except Exception as e:
        logger.warning(f"No se pudo guardar paper: {e}")
    
    return PaperResponse(
        source=paper_data.get("source", "link"),
        title=paper_data.get("title", ""),
        authors=paper_data.get("authors", []),
        abstract=paper_data.get("abstract", ""),
        sections_count=len(paper_data.get("sections", [])),
        content_length=len(paper_data.get("content", "")),
        metadata=paper_data.get("metadata", {})
    )


@router.get("/papers")
@handle_api_errors()
async def list_papers(limit: int = 50):
    """Lista todos los papers almacenados"""
    papers = paper_storage.list_papers(limit=limit)
    return {
        "papers": papers,
        "total": len(papers),
        "statistics": paper_storage.get_statistics()
    }


@router.get("/papers/{paper_id}")
@handle_api_errors()
async def get_paper(paper_id: str):
    """Obtiene un paper por ID"""
    paper = paper_storage.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper no encontrado")
    return paper


@router.post("/training/train", response_model=TrainingResponse)
@handle_api_errors()
async def train_model(request: TrainingRequest):
    """
    Entrena un modelo basado en papers.
    
    Args:
        request: Configuración del entrenamiento
        
    Returns:
        Información del modelo entrenado
    """
    # Obtener papers desde almacenamiento
    if request.use_all_papers:
        papers_data = paper_storage.list_papers()
        papers = [paper_storage.get_paper(p["id"]) for p in papers_data]
        papers = [p for p in papers if p]  # Filtrar None
    else:
        papers = []
        if request.paper_ids:
            for paper_id in request.paper_ids:
                paper = paper_storage.get_paper(paper_id)
                if paper:
                    papers.append(paper)
    
    if not papers:
        raise HTTPException(
            status_code=400,
            detail="No hay papers disponibles para entrenar. Sube papers primero."
        )
    
    # Entrenar modelo
    model_path = model_trainer.train_from_papers(
        papers=papers,
        epochs=request.epochs
    )
    
    # Obtener estado del modelo
    status = model_trainer.get_model_status(model_path)
    
    return TrainingResponse(
        model_id=model_path.split("/")[-1],
        status=status.get("status", "ready"),
        papers_count=len(papers),
        training_examples=status.get("config", {}).get("training_examples", 0),
        epochs=request.epochs,
        model_path=model_path
    )


@router.get("/models/{model_id}/status", response_model=ModelStatusResponse)
@handle_api_errors()
async def get_model_status(model_id: str):
    """
    Obtiene el estado de un modelo.
    
    Args:
        model_id: ID del modelo
        
    Returns:
        Estado del modelo
    """
    model_path = resolve_model_path(model_id)
    if not model_path:
        raise HTTPException(status_code=400, detail="model_id es requerido")
    
    status = model_trainer.get_model_status(model_path)
    
    return ModelStatusResponse(
        model_id=model_id,
        status=status.get("status", "unknown"),
        config=status.get("config"),
        error=status.get("error")
    )


@router.post("/code/improve", response_model=CodeImproveResponse)
@handle_api_errors()
async def improve_code(request: CodeImproveRequest):
    """
    Mejora código de un repositorio de GitHub.
    
    Args:
        request: Información del código a mejorar
        
    Returns:
        Código mejorado y sugerencias
    """
    code_improver = create_code_improver(
        model_id=request.model_id,
        vector_store=vector_store,
        use_rag=True
    )
    
    # Mejorar código
    result = code_improver.improve_code(
        github_repo=request.github_repo,
        file_path=request.file_path,
        branch=request.branch
    )
    
    return CodeImproveResponse(
        original_code=result.get("original_code", ""),
        improved_code=result.get("improved_code", ""),
        suggestions=result.get("suggestions", []),
        repo=result.get("repo", ""),
        file_path=result.get("file_path", ""),
        improvements_applied=result.get("improvements_applied", 0)
    )


@router.post("/code/improve-text", response_model=CodeImproveResponse)
@handle_api_errors()
async def improve_code_text(code: str, context: str = None, model_id: str = None):
    """
    Mejora código directamente desde texto.
    
    Args:
        code: Código a mejorar
        context: Contexto adicional (opcional)
        model_id: ID del modelo a usar (opcional)
        
    Returns:
        Código mejorado y sugerencias
    """
    code_improver = create_code_improver(
        model_id=model_id,
        vector_store=vector_store,
        use_rag=True
    )
    
    # Mejorar código
    result = code_improver.improve_code_from_text(code, context)
    
    return CodeImproveResponse(
        original_code=result.get("original_code", ""),
        improved_code=result.get("improved_code", ""),
        suggestions=result.get("suggestions", []),
        repo="",
        file_path="",
        improvements_applied=result.get("improvements_applied", 0)
    )


@router.post("/repository/analyze", response_model=RepositoryAnalyzeResponse)
@handle_api_errors()
async def analyze_repository(request: RepositoryAnalyzeRequest):
    """
    Analiza un repositorio completo y sugiere mejoras.
    
    Args:
        request: Información del repositorio
        
    Returns:
        Análisis completo del repositorio
    """
    code_improver = create_code_improver(
        model_id=request.model_id,
        vector_store=vector_store,
        use_rag=True
    )
    
    # Analizar repositorio
    result = code_improver.analyze_repository(
        github_repo=request.github_repo,
        branch=request.branch
    )
    
    return RepositoryAnalyzeResponse(
        repo=result.get("repo", ""),
        files_analyzed=result.get("files_analyzed", 0),
        total_improvements=result.get("total_improvements", 0),
        improvements=result.get("improvements", [])
    )


@router.get("/vector-store/stats")
@handle_api_errors()
async def vector_store_stats():
    """Obtiene estadísticas del vector store"""
    return {
        "papers_indexed": vector_store.get_paper_count(),
        "collection_name": vector_store.collection_name
    }


@router.post("/batch/improve")
@handle_api_errors()
async def batch_improve(files: List[Dict[str, Any]], model_id: Optional[str] = None):
    """
    Procesa múltiples archivos en lote.
    
    Args:
        files: Lista de archivos [{"repo": "...", "path": "...", "branch": "..."}]
        model_id: ID del modelo a usar (opcional)
        
    Returns:
        Resultados del procesamiento en lote
    """
    from ..core.batch_processor import BatchProcessor
    
    code_improver = create_code_improver(
        model_id=model_id,
        vector_store=vector_store,
        use_rag=True,
        use_cache=True,
        use_analyzer=True
    )
    
    # Procesar en lote
    processor = BatchProcessor(max_workers=4)
    results = processor.process_files(files, code_improver)
    
    # Generar resumen
    summary = processor.generate_summary(results)
    
    return {
        "summary": summary,
        "results": results
    }


@router.post("/export")
@handle_api_errors()
async def export_results(
    results: Dict[str, Any],
    format: str = "json"
):
    """
    Exporta resultados en diferentes formatos.
    
    Args:
        results: Resultados a exportar
        format: Formato (json, markdown, html)
        
    Returns:
        Ruta al archivo exportado
    """
    from ..utils.exporters import ResultExporter
    
    exporter = ResultExporter()
    
    if format == "json":
        filepath = exporter.export_json(results)
    elif format == "markdown":
        filepath = exporter.export_markdown(results)
    elif format == "html":
        filepath = exporter.export_html(results)
    else:
        raise HTTPException(status_code=400, detail=f"Formato no soportado: {format}")
    
    return {
        "filepath": filepath,
        "format": format,
        "download_url": f"/api/research-paper-code-improver/download/{Path(filepath).name}"
    }


@router.get("/cache/stats")
@handle_api_errors()
async def cache_stats():
    """Obtiene estadísticas del cache"""
    from ..core.cache_manager import CacheManager
    
    cache_manager = CacheManager()
    return cache_manager.get_cache_stats()


@router.post("/cache/clear")
@handle_api_errors()
async def clear_cache(older_than_hours: Optional[int] = None):
    """Limpia el cache"""
    from ..core.cache_manager import CacheManager
    
    cache_manager = CacheManager()
    deleted = cache_manager.clear_cache(older_than_hours)
    
    return {
        "deleted_files": deleted,
        "message": f"Cache limpiado: {deleted} archivos eliminados"
    }


@router.post("/analyze/code")
@handle_api_errors()
async def analyze_code(code: str, language: str = "python"):
    """
    Analiza código sin mejorarlo.
    
    Args:
        code: Código a analizar
        language: Lenguaje de programación
        
    Returns:
        Análisis del código
    """
    from ..core.code_analyzer import CodeAnalyzer
    
    analyzer = CodeAnalyzer()
    return analyzer.analyze_code(code, language)


@router.post("/compare/code")
@handle_api_errors()
async def compare_code(original: str, improved: str, language: str = "python"):
    """
    Compara código original y mejorado.
    
    Args:
        original: Código original
        improved: Código mejorado
        language: Lenguaje de programación
        
    Returns:
        Comparación detallada
    """
    from ..core.code_analyzer import CodeAnalyzer
    
    analyzer = CodeAnalyzer()
    return analyzer.compare_code(original, improved, language)


@router.get("/metrics/stats")
@handle_api_errors()
async def get_metrics_stats(hours: int = 24):
    """Obtiene estadísticas de métricas"""
    from ..core.metrics_collector import MetricsCollector
    
    collector = MetricsCollector()
    return collector.get_statistics(hours=hours)


@router.post("/tests/generate")
@handle_api_errors()
async def generate_tests(code: str, language: str = "python", framework: str = "pytest"):
    """
    Genera tests automáticos para código.
    
    Args:
        code: Código a testear
        language: Lenguaje de programación
        framework: Framework de testing
        
    Returns:
        Tests generados
    """
    from ..core.test_generator import TestGenerator
    
    generator = TestGenerator()
    return generator.generate_tests(code, language, framework)


@router.post("/git/apply")
@handle_api_errors()
async def apply_to_git(
    repo_url: str,
    improvements: List[Dict[str, Any]],
    branch_name: str = "code-improvements",
    commit_message: str = "Apply code improvements from research papers"
):
    """
    Aplica mejoras a un repositorio Git.
    
    Args:
        repo_url: URL del repositorio
        improvements: Lista de mejoras
        branch_name: Nombre de la rama
        commit_message: Mensaje de commit
        
    Returns:
        Resultado de la operación
    """
    from ..core.git_integration import GitIntegration
    
    git = GitIntegration()
    
    # Clonar repositorio
    repo_path = git.clone_repository(repo_url)
    
    # Crear rama
    git.create_branch(repo_path, branch_name)
    
    # Aplicar mejoras
    result = git.apply_improvements(repo_path, improvements, commit_message)
    
    # Generar info de PR
    pr_info = git.create_pull_request_info(
        repo_path,
        branch_name,
        "Code Improvements from Research Papers",
        commit_message
    )
    
    result["pr_info"] = pr_info
    
    return result


@router.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "service": "Research Paper Code Improver",
        "version": "1.0.0",
        "vector_store": {
            "papers_indexed": vector_store.get_paper_count(),
            "available": vector_store.collection is not None or hasattr(vector_store, "_documents")
        },
        "paper_storage": {
            "total_papers": paper_storage.get_statistics().get("total_papers", 0)
        },
        "features": {
            "rag": RAG_AVAILABLE,
            "cache": CACHE_AVAILABLE,
            "analyzer": ANALYZER_AVAILABLE
        }
    }
