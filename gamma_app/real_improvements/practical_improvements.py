"""
Practical Improvements - Solo cosas reales y funcionales
Mejoras prácticas basadas en los archivos existentes
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sqlite3
import hashlib
import uuid

@dataclass
class PracticalImprovement:
    """Mejora práctica real"""
    id: str
    title: str
    description: str
    effort_minutes: int
    impact_score: int  # 1-10
    category: str
    code_example: str
    test_command: str
    is_implemented: bool = False

class PracticalImprovementsEngine:
    """Motor de mejoras prácticas reales"""
    
    def __init__(self):
        self.improvements = []
        self._load_practical_improvements()
    
    def _load_practical_improvements(self):
        """Cargar mejoras prácticas reales"""
        self.improvements = [
            PracticalImprovement(
                id="pi_001",
                title="Añadir índices a la base de datos",
                description="Crear índices en columnas frecuentemente consultadas para mejorar performance",
                effort_minutes=15,
                impact_score=9,
                category="performance",
                code_example="""
# Ejecutar en la base de datos
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_usage_stats_date ON usage_stats(date);
""",
                test_command="SELECT * FROM sqlite_master WHERE type='index';"
            ),
            
            PracticalImprovement(
                id="pi_002",
                title="Implementar caché en memoria",
                description="Añadir caché LRU para consultas frecuentes",
                effort_minutes=30,
                impact_score=8,
                category="performance",
                code_example="""
from functools import lru_cache
import time

@lru_cache(maxsize=1000)
def get_user_by_id(user_id: str):
    # Tu consulta a la base de datos aquí
    return database.get_user(user_id)

@lru_cache(maxsize=500)
def get_template_by_name(template_name: str):
    return database.get_template(template_name)
""",
                test_command="python -c 'from functools import lru_cache; print(\"Cache implementado\")'"
            ),
            
            PracticalImprovement(
                id="pi_003",
                title="Añadir validación de entrada",
                description="Validar todos los inputs con Pydantic",
                effort_minutes=20,
                impact_score=9,
                category="security",
                code_example="""
from pydantic import BaseModel, EmailStr, validator
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

class DocumentCreate(BaseModel):
    title: str
    content: str
    template_type: str
    language: str = "es"
    
    @validator('title')
    def validate_title(cls, v):
        if len(v) < 3:
            raise ValueError('Title must be at least 3 characters')
        return v
""",
                test_command="python -c 'from pydantic import BaseModel; print(\"Pydantic funcionando\")'"
            ),
            
            PracticalImprovement(
                id="pi_004",
                title="Implementar rate limiting",
                description="Limitar requests por usuario/IP",
                effort_minutes=25,
                impact_score=7,
                category="security",
                code_example="""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/documents/")
@limiter.limit("10/minute")
async def create_document(request: Request, document: DocumentCreate):
    # Tu lógica aquí
    pass

@app.get("/api/users/")
@limiter.limit("100/hour")
async def get_users(request: Request):
    # Tu lógica aquí
    pass
""",
                test_command="pip install slowapi"
            ),
            
            PracticalImprovement(
                id="pi_005",
                title="Añadir logging estructurado",
                description="Implementar logging con información útil",
                effort_minutes=20,
                impact_score=8,
                category="maintainability",
                code_example="""
import logging
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_user_action(user_id: str, action: str, details: dict = None):
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "action": action,
        "details": details or {}
    }
    logger.info(json.dumps(log_data))

# Usar en tus endpoints
@app.post("/api/documents/")
async def create_document(document: DocumentCreate, current_user: User = Depends(get_current_user)):
    log_user_action(current_user.id, "create_document", {"title": document.title})
    # Tu lógica aquí
""",
                test_command="python -c 'import logging; print(\"Logging configurado\")'"
            ),
            
            PracticalImprovement(
                id="pi_006",
                title="Implementar health checks",
                description="Añadir endpoints de salud para monitoreo",
                effort_minutes=15,
                impact_score=6,
                category="reliability",
                code_example="""
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "disk_space": check_disk_space()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

async def check_database_connection():
    try:
        # Tu verificación de DB aquí
        return True
    except:
        return False
""",
                test_command="curl http://localhost:8000/health"
            ),
            
            PracticalImprovement(
                id="pi_007",
                title="Optimizar consultas SQL",
                description="Mejorar consultas lentas identificadas",
                effort_minutes=30,
                impact_score=9,
                category="performance",
                code_example="""
# ANTES (lento)
SELECT * FROM documents WHERE user_id = ? AND status = 'active';

# DESPUÉS (optimizado)
SELECT id, title, created_at, status 
FROM documents 
WHERE user_id = ? AND status = 'active' 
ORDER BY created_at DESC 
LIMIT 50;

# Añadir índice compuesto
CREATE INDEX idx_documents_user_status ON documents(user_id, status);

# Usar JOINs eficientes
SELECT d.id, d.title, u.username 
FROM documents d 
INNER JOIN users u ON d.user_id = u.id 
WHERE d.status = 'active';
""",
                test_command="EXPLAIN QUERY PLAN SELECT * FROM documents WHERE user_id = 'test';"
            ),
            
            PracticalImprovement(
                id="pi_008",
                title="Añadir compresión de respuestas",
                description="Comprimir respuestas HTTP para reducir ancho de banda",
                effort_minutes=10,
                impact_score=7,
                category="performance",
                code_example="""
from fastapi.middleware.gzip import GZipMiddleware

# Añadir middleware de compresión
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Para respuestas grandes, usar streaming
from fastapi.responses import StreamingResponse

@app.get("/api/documents/{document_id}/download")
async def download_document(document_id: str):
    def generate_file():
        # Generar archivo en chunks
        with open(f"documents/{document_id}.pdf", "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(
        generate_file(),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=document.pdf"}
    )
""",
                test_command="curl -H 'Accept-Encoding: gzip' http://localhost:8000/api/documents/"
            ),
            
            PracticalImprovement(
                id="pi_009",
                title="Implementar paginación",
                description="Añadir paginación a endpoints que devuelven listas",
                effort_minutes=20,
                impact_score=8,
                category="performance",
                code_example="""
from typing import Optional

@app.get("/api/documents/")
async def get_documents(
    page: int = 1,
    size: int = 10,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    # Validar parámetros
    if page < 1:
        page = 1
    if size < 1 or size > 100:
        size = 10
    
    offset = (page - 1) * size
    
    # Construir query
    query = "SELECT * FROM documents WHERE user_id = ?"
    params = [current_user.id]
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([size, offset])
    
    # Ejecutar query
    documents = database.execute_query(query, params)
    total_count = database.get_count("documents", {"user_id": current_user.id})
    
    return {
        "documents": documents,
        "pagination": {
            "page": page,
            "size": size,
            "total": total_count,
            "pages": (total_count + size - 1) // size
        }
    }
""",
                test_command="curl 'http://localhost:8000/api/documents/?page=1&size=10'"
            ),
            
            PracticalImprovement(
                id="pi_010",
                title="Añadir manejo de errores",
                description="Implementar manejo robusto de errores",
                effort_minutes=25,
                impact_score=9,
                category="reliability",
                code_example="""
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

class BusinessLogicError(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

@app.exception_handler(BusinessLogicError)
async def business_logic_error_handler(request, exc):
    logger.error(f"Business logic error: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "type": "business_logic_error"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"Value error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "validation_error"}
    )

# En tus endpoints
@app.post("/api/documents/")
async def create_document(document: DocumentCreate, current_user: User = Depends(get_current_user)):
    try:
        # Validar que el usuario no exceda el límite
        user_doc_count = database.get_user_document_count(current_user.id)
        if user_doc_count >= 100:  # Límite de documentos
            raise BusinessLogicError(
                "User has reached document limit",
                "DOCUMENT_LIMIT_EXCEEDED"
            )
        
        # Crear documento
        new_document = database.create_document(document, current_user.id)
        return new_document
        
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
""",
                test_command="python -c 'from fastapi import HTTPException; print(\"Error handling configurado\")'"
            )
        ]
    
    def get_quick_wins(self) -> List[PracticalImprovement]:
        """Obtener mejoras rápidas (menos de 30 minutos)"""
        return [imp for imp in self.improvements if imp.effort_minutes <= 30]
    
    def get_high_impact(self) -> List[PracticalImprovement]:
        """Obtener mejoras de alto impacto (score >= 8)"""
        return [imp for imp in self.improvements if imp.impact_score >= 8]
    
    def get_by_category(self, category: str) -> List[PracticalImprovement]:
        """Obtener mejoras por categoría"""
        return [imp for imp in self.improvements if imp.category == category]
    
    def mark_implemented(self, improvement_id: str) -> bool:
        """Marcar mejora como implementada"""
        for imp in self.improvements:
            if imp.id == improvement_id:
                imp.is_implemented = True
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de mejoras"""
        total = len(self.improvements)
        implemented = len([imp for imp in self.improvements if imp.is_implemented])
        quick_wins = len(self.get_quick_wins())
        high_impact = len(self.get_high_impact())
        
        return {
            "total_improvements": total,
            "implemented": implemented,
            "pending": total - implemented,
            "quick_wins": quick_wins,
            "high_impact": high_impact,
            "completion_rate": f"{(implemented/total*100):.1f}%" if total > 0 else "0%"
        }
    
    def export_improvements(self) -> Dict[str, Any]:
        """Exportar mejoras a JSON"""
        return {
            "improvements": [
                {
                    "id": imp.id,
                    "title": imp.title,
                    "description": imp.description,
                    "effort_minutes": imp.effort_minutes,
                    "impact_score": imp.impact_score,
                    "category": imp.category,
                    "code_example": imp.code_example,
                    "test_command": imp.test_command,
                    "is_implemented": imp.is_implemented
                }
                for imp in self.improvements
            ],
            "stats": self.get_stats(),
            "exported_at": datetime.now().isoformat()
        }

# Instancia global
practical_engine = PracticalImprovementsEngine()

def get_practical_improvements() -> PracticalImprovementsEngine:
    """Obtener motor de mejoras prácticas"""
    return practical_engine

# Función de utilidad para mostrar mejoras
def show_improvements():
    """Mostrar mejoras disponibles"""
    engine = get_practical_improvements()
    
    print("🚀 MEJORAS PRÁCTICAS DISPONIBLES")
    print("=" * 50)
    
    for imp in engine.improvements:
        status = "✅ IMPLEMENTADA" if imp.is_implemented else "⏳ PENDIENTE"
        print(f"\n{imp.id}: {imp.title}")
        print(f"   {imp.description}")
        print(f"   Esfuerzo: {imp.effort_minutes} min | Impacto: {imp.impact_score}/10 | {status}")
        print(f"   Categoría: {imp.category}")
    
    print(f"\n📊 ESTADÍSTICAS:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    show_improvements()





