from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
from datetime import datetime
import json
from official_docs_reference import OfficialDocsReference
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
FastAPI Integration - Official Documentation Reference System
============================================================

Integración de FastAPI con el sistema de referencias de documentación oficial
para desarrollo de APIs escalables.
"""


# Importar el sistema de referencias

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI app
app = FastAPI(
    title="Official Docs Reference API",
    description="API para acceder a referencias de documentación oficial de ML libraries",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar sistema de referencias
ref_system = OfficialDocsReference()

# Modelos Pydantic
class LibraryInfoRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")

class APIReferenceRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")
    api_name: str = Field(..., description="Nombre de la API")

class BestPracticesRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")
    category: Optional[str] = Field(None, description="Categoría de mejores prácticas")

class VersionCompatibilityRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")
    version: str = Field(..., description="Versión a verificar")

class CodeValidationRequest(BaseModel):
    code: str = Field(..., description="Código a validar")
    library_name: str = Field(..., description="Librería para validación")

class PerformanceRecommendationsRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")

class MigrationGuideRequest(BaseModel):
    library_name: str = Field(..., description="Nombre de la librería")
    from_version: str = Field(..., description="Versión de origen")
    to_version: str = Field(..., description="Versión de destino")

# Dependencias
async def get_ref_system() -> OfficialDocsReference:
    """Dependency para obtener el sistema de referencias."""
    return ref_system

# Endpoints principales
@app.get("/")
async def root():
    """Endpoint raíz con información del API."""
    return {
        "message": "Official Documentation Reference API",
        "version": "1.0.0",
        "docs": "/docs",
        "available_libraries": ["pytorch", "transformers", "diffusers", "gradio"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "Official Docs Reference API"
    }

@app.post("/library/info")
async def get_library_info(
    request: LibraryInfoRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Obtener información de una librería."""
    try:
        lib_info = ref_system.get_library_info(request.library_name)
        if not lib_info:
            raise HTTPException(status_code=404, detail=f"Librería '{request.library_name}' no encontrada")
        
        return {
            "success": True,
            "library": {
                "name": lib_info.name,
                "current_version": lib_info.current_version,
                "min_supported_version": lib_info.min_supported_version,
                "documentation_url": lib_info.documentation_url,
                "github_url": lib_info.github_url,
                "pip_package": lib_info.pip_package,
                "conda_package": lib_info.conda_package
            }
        }
    except Exception as e:
        logger.error(f"Error getting library info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reference")
async def get_api_reference(
    request: APIReferenceRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Obtener referencia de una API específica."""
    try:
        api_ref = ref_system.get_api_reference(request.library_name, request.api_name)
        if not api_ref:
            raise HTTPException(
                status_code=404, 
                detail=f"API '{request.api_name}' no encontrada en librería '{request.library_name}'"
            )
        
        return {
            "success": True,
            "api_reference": {
                "name": api_ref.name,
                "description": api_ref.description,
                "official_docs_url": api_ref.official_docs_url,
                "code_example": api_ref.code_example,
                "best_practices": api_ref.best_practices,
                "performance_tips": api_ref.performance_tips,
                "deprecation_warning": api_ref.deprecation_warning,
                "migration_guide": api_ref.migration_guide
            }
        }
    except Exception as e:
        logger.error(f"Error getting API reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/best-practices")
async def get_best_practices(
    request: BestPracticesRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Obtener mejores prácticas de una librería."""
    try:
        practices = ref_system.get_best_practices(
            request.library_name, 
            request.category
        )
        
        return {
            "success": True,
            "library": request.library_name,
            "category": request.category,
            "practices": [
                {
                    "title": practice.title,
                    "description": practice.description,
                    "source": practice.source,
                    "code_example": practice.code_example,
                    "category": practice.category,
                    "importance": practice.importance
                }
                for practice in practices
            ],
            "count": len(practices)
        }
    except Exception as e:
        logger.error(f"Error getting best practices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/version/compatibility")
async def check_version_compatibility(
    request: VersionCompatibilityRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Verificar compatibilidad de versiones."""
    try:
        compat = ref_system.check_version_compatibility(
            request.library_name, 
            request.version
        )
        
        return {
            "success": True,
            "library": request.library_name,
            "requested_version": request.version,
            "compatibility": compat
        }
    except Exception as e:
        logger.error(f"Error checking version compatibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/validate")
async def validate_code(
    request: CodeValidationRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Validar código contra mejores prácticas."""
    try:
        validation = ref_system.validate_code_snippet(
            request.code, 
            request.library_name
        )
        
        return {
            "success": True,
            "library": request.library_name,
            "validation": validation
        }
    except Exception as e:
        logger.error(f"Error validating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/performance/recommendations")
async def get_performance_recommendations(
    request: PerformanceRecommendationsRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Obtener recomendaciones de rendimiento."""
    try:
        recommendations = ref_system.get_performance_recommendations(
            request.library_name
        )
        
        return {
            "success": True,
            "library": request.library_name,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error getting performance recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/migration/guide")
async def generate_migration_guide(
    request: MigrationGuideRequest,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Generar guía de migración."""
    try:
        guide = ref_system.generate_migration_guide(
            request.library_name,
            request.from_version,
            request.to_version
        )
        
        return {
            "success": True,
            "migration_guide": guide
        }
    except Exception as e:
        logger.error(f"Error generating migration guide: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de utilidad
@app.get("/libraries")
async def list_libraries(ref_system: OfficialDocsReference = Depends(get_ref_system)):
    """Listar todas las librerías disponibles."""
    try:
        libraries = list(ref_system.libraries.keys())
        return {
            "success": True,
            "libraries": libraries,
            "count": len(libraries)
        }
    except Exception as e:
        logger.error(f"Error listing libraries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/apis/{library_name}")
async def list_apis(
    library_name: str,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Listar todas las APIs disponibles para una librería."""
    try:
        refs = getattr(ref_system, f"{library_name.lower()}_refs", {})
        apis = list(refs.keys())
        
        return {
            "success": True,
            "library": library_name,
            "apis": apis,
            "count": len(apis)
        }
    except Exception as e:
        logger.error(f"Error listing APIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/references")
async def export_references(
    background_tasks: BackgroundTasks,
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Exportar todas las referencias a archivos."""
    try:
        def export_task():
            """Tarea en background para exportar referencias."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Exportar a JSON
            json_file = f"references_{timestamp}.json"
            ref_system.export_references(json_file, "json")
            
            # Exportar a YAML
            yaml_file = f"references_{timestamp}.yaml"
            ref_system.export_references(yaml_file, "yaml")
            
            logger.info(f"References exported to {json_file} and {yaml_file}")
        
        # Ejecutar tarea en background
        background_tasks.add_task(export_task)
        
        return {
            "success": True,
            "message": "Export started in background",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting references: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de análisis
@app.post("/analyze/project")
async def analyze_project(
    libraries: List[str],
    ref_system: OfficialDocsReference = Depends(get_ref_system)
):
    """Analizar un proyecto con múltiples librerías."""
    try:
        analysis = {
            "project_analysis": {
                "libraries": [],
                "total_recommendations": 0,
                "compatibility_issues": [],
                "performance_opportunities": []
            }
        }
        
        for lib in libraries:
            lib_info = ref_system.get_library_info(lib)
            if lib_info:
                # Obtener recomendaciones
                recommendations = ref_system.get_performance_recommendations(lib)
                
                # Obtener mejores prácticas
                practices = ref_system.get_best_practices(lib)
                
                lib_analysis = {
                    "name": lib,
                    "current_version": lib_info.current_version,
                    "recommendations_count": len(recommendations),
                    "best_practices_count": len(practices),
                    "recommendations": recommendations[:5],  # Top 5
                    "critical_practices": [
                        p for p in practices if p.importance == "critical"
                    ]
                }
                
                analysis["project_analysis"]["libraries"].append(lib_analysis)
                analysis["project_analysis"]["total_recommendations"] += len(recommendations)
        
        return {
            "success": True,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Error analyzing project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Middleware para logging
@app.middleware("http")
async async def log_requests(request, call_next) -> Any:
    """Middleware para logging de requests."""
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async async def http_exception_handler(request, exc) -> Any:
    """Handler para excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc) -> Any:
    """Handler para excepciones generales."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Función para ejecutar el servidor
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Ejecutar el servidor FastAPI."""
    uvicorn.run(
        "fastapi_integration:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

match __name__:
    case "__main__":
    run_server() 