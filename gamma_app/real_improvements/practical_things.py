#!/usr/bin/env python3
"""
Practical Things - Cosas prácticas y funcionales
Implementaciones que resuelven problemas reales
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class PracticalThings:
    """Cosas prácticas que resuelven problemas reales"""
    
    def __init__(self):
        self.db_path = "practical_things.db"
        self._init_database()
    
    def _init_database(self):
        """Inicializar base de datos práctica"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS practical_solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    code TEXT NOT NULL,
                    time_minutes INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    category TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Base de datos práctica inicializada")
            
        except Exception as e:
            print(f"❌ Error inicializando DB: {e}")
    
    def add_practical_solution(self, problem: str, solution: str, code: str, 
                             time_minutes: int, difficulty: str, category: str):
        """Añadir solución práctica"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO practical_solutions 
                (problem, solution, code, time_minutes, difficulty, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (problem, solution, code, time_minutes, difficulty, category, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            print(f"✅ Solución práctica añadida: {problem[:50]}...")
            
        except Exception as e:
            print(f"❌ Error añadiendo solución: {e}")
    
    def get_practical_solutions(self, category: str = None) -> List[Dict[str, Any]]:
        """Obtener soluciones prácticas"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT id, problem, solution, code, time_minutes, difficulty, category, created_at
                    FROM practical_solutions
                    WHERE category = ?
                    ORDER BY time_minutes ASC
                ''', (category,))
            else:
                cursor.execute('''
                    SELECT id, problem, solution, code, time_minutes, difficulty, category, created_at
                    FROM practical_solutions
                    ORDER BY time_minutes ASC
                ''')
            
            solutions = []
            for row in cursor.fetchall():
                solutions.append({
                    "id": row[0],
                    "problem": row[1],
                    "solution": row[2],
                    "code": row[3],
                    "time_minutes": row[4],
                    "difficulty": row[5],
                    "category": row[6],
                    "created_at": row[7]
                })
            
            conn.close()
            return solutions
            
        except Exception as e:
            print(f"❌ Error obteniendo soluciones: {e}")
            return []
    
    def get_categories(self) -> List[str]:
        """Obtener categorías disponibles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT DISTINCT category FROM practical_solutions')
            categories = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return categories
            
        except Exception as e:
            print(f"❌ Error obteniendo categorías: {e}")
            return []

def create_practical_solutions():
    """Crear soluciones prácticas reales"""
    practical = PracticalThings()
    
    # Performance Solutions
    practical.add_practical_solution(
        "Consultas de base de datos lentas",
        "Añadir índices estratégicos para mejorar performance de consultas",
        '''-- Crear índices para consultas frecuentes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_documents_user_id ON documents(user_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);

-- Índice compuesto para consultas complejas
CREATE INDEX idx_documents_user_status ON documents(user_id, status);

-- Verificar uso de índices
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'user@example.com';''',
        15,
        "Fácil",
        "Performance"
    )
    
    practical.add_practical_solution(
        "Respuestas API lentas",
        "Implementar caché LRU para respuestas frecuentes",
        '''from functools import lru_cache
import time
from typing import Optional, Any

class APICache:
    """Caché simple para respuestas API"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()

# Uso en endpoints
api_cache = APICache()

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    cache_key = f"user:{user_id}"
    cached_user = api_cache.get(cache_key)
    if cached_user:
        return cached_user
    
    user = get_user_from_db(user_id)
    if user:
        api_cache.set(cache_key, user.dict())
    return user''',
        25,
        "Medio",
        "Performance"
    )
    
    # Security Solutions
    practical.add_practical_solution(
        "API sin protección contra abuso",
        "Implementar rate limiting para prevenir abuso",
        '''from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configurar rate limiter
limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Aplicar límites por endpoint
@app.post("/api/documents/")
@limiter.limit("10/minute")
async def create_document(document: DocumentCreate):
    pass

@app.get("/api/users/")
@limiter.limit("100/hour")
async def get_users():
    pass

@app.post("/api/auth/login")
@limiter.limit("5/minute")
async def login(credentials: LoginRequest):
    pass''',
        20,
        "Fácil",
        "Security"
    )
    
    practical.add_practical_solution(
        "Validación de datos insuficiente",
        "Implementar validación robusta con Pydantic",
        '''from pydantic import BaseModel, EmailStr, validator
from typing import Optional
import re

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.isalnum():
            raise ValueError('Username must contain only letters and numbers')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError('Age must be between 0 and 120')
        return v

class DocumentCreate(BaseModel):
    title: str
    content: str
    template_type: str
    
    @validator('title')
    def validate_title(cls, v):
        if len(v.strip()) < 3:
            raise ValueError('Title must be at least 3 characters')
        return v.strip()
    
    @validator('template_type')
    def validate_template_type(cls, v):
        allowed_types = ['contract', 'report', 'letter', 'proposal', 'invoice']
        if v not in allowed_types:
            raise ValueError(f'Template type must be one of: {", ".join(allowed_types)}')
        return v''',
        30,
        "Medio",
        "Security"
    )
    
    # Monitoring Solutions
    practical.add_practical_solution(
        "Sin monitoreo de la aplicación",
        "Implementar health checks y métricas básicas",
        '''from fastapi import FastAPI
import psutil
from datetime import datetime

@app.get("/health")
async def health_check():
    """Health check básico"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Health check detallado"""
    checks = {
        "database": await check_database_connection(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage(),
        "cpu": check_cpu_usage()
    }
    
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

def check_disk_space():
    """Verificar espacio en disco"""
    try:
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        return free_percent > 10
    except:
        return False

def check_memory_usage():
    """Verificar uso de memoria"""
    try:
        memory = psutil.virtual_memory()
        return memory.percent < 90
    except:
        return False

def check_cpu_usage():
    """Verificar uso de CPU"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 90
    except:
        return False''',
        20,
        "Fácil",
        "Monitoring"
    )
    
    # Error Handling Solutions
    practical.add_practical_solution(
        "Errores no manejados correctamente",
        "Implementar manejo robusto de errores",
        '''from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BusinessLogicError(Exception):
    """Error de lógica de negocio"""
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
            "type": "business_logic_error",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Uso en endpoints
@app.get("/users/{user_id}")
async def get_user(user_id: str):
    try:
        user = get_user_from_db(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    except BusinessLogicError as e:
        raise e  # Re-lanzar para que sea manejado por el handler''',
        35,
        "Medio",
        "Error Handling"
    )
    
    # Logging Solutions
    practical.add_practical_solution(
        "Logs desorganizados y difíciles de leer",
        "Implementar logging estructurado",
        '''import logging
import structlog
from datetime import datetime
import json

# Configurar logging estructurado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt='iso'),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Middleware para logging de requests
@app.middleware('http')
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info(
        'Request started',
        method=request.method,
        url=str(request.url),
        request_id=request_id,
        user_agent=request.headers.get('user-agent'),
        ip_address=request.client.host if request.client else None
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(
            'Request completed',
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time,
            request_id=request_id
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        
        logger.error(
            'Request failed',
            method=request.method,
            url=str(request.url),
            error=str(e),
            process_time=process_time,
            request_id=request_id,
            exc_info=True
        )
        
        raise

# Uso en endpoints
@app.post("/users/")
async def create_user(user: UserCreate):
    logger.info("Creating user", email=user.email, username=user.username)
    
    try:
        new_user = create_user_in_db(user.dict())
        logger.info("User created successfully", user_id=new_user.id)
        return new_user
    except Exception as e:
        logger.error("Failed to create user", error=str(e), email=user.email)
        raise''',
        25,
        "Medio",
        "Logging"
    )
    
    return practical

def main():
    """Función principal"""
    print("🔧 PRACTICAL THINGS - Soluciones prácticas")
    print("=" * 50)
    
    # Crear soluciones prácticas
    practical = create_practical_solutions()
    
    # Mostrar menú
    while True:
        print("\n🎯 MENÚ DE SOLUCIONES PRÁCTICAS")
        print("1. Ver todas las soluciones")
        print("2. Ver soluciones por categoría")
        print("3. Buscar solución por problema")
        print("4. Ver categorías disponibles")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ").strip()
        
        if choice == "1":
            solutions = practical.get_practical_solutions()
            print(f"\n📋 TODAS LAS SOLUCIONES ({len(solutions)} disponibles)")
            print("=" * 50)
            
            for i, solution in enumerate(solutions, 1):
                print(f"\n{i}. {solution['problem']}")
                print(f"   🔧 Solución: {solution['solution']}")
                print(f"   ⏱️  Tiempo: {solution['time_minutes']} minutos")
                print(f"   📊 Dificultad: {solution['difficulty']}")
                print(f"   🏷️  Categoría: {solution['category']}")
        
        elif choice == "2":
            categories = practical.get_categories()
            print(f"\n📂 CATEGORÍAS DISPONIBLES:")
            for i, cat in enumerate(categories, 1):
                print(f"   {i}. {cat}")
            
            try:
                cat_choice = int(input("\nSelecciona categoría: ")) - 1
                if 0 <= cat_choice < len(categories):
                    category = categories[cat_choice]
                    solutions = practical.get_practical_solutions(category)
                    
                    print(f"\n🔧 SOLUCIONES DE {category.upper()} ({len(solutions)} disponibles)")
                    print("=" * 50)
                    
                    for i, solution in enumerate(solutions, 1):
                        print(f"\n{i}. {solution['problem']}")
                        print(f"   🔧 Solución: {solution['solution']}")
                        print(f"   ⏱️  Tiempo: {solution['time_minutes']} minutos")
                        print(f"   📊 Dificultad: {solution['difficulty']}")
                else:
                    print("❌ Categoría inválida")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == "3":
            search_term = input("\n🔍 Buscar problema: ").strip().lower()
            solutions = practical.get_practical_solutions()
            matches = [s for s in solutions if search_term in s['problem'].lower()]
            
            if matches:
                print(f"\n🎯 RESULTADOS DE BÚSQUEDA ({len(matches)} encontrados)")
                print("=" * 40)
                
                for i, solution in enumerate(matches, 1):
                    print(f"\n{i}. {solution['problem']}")
                    print(f"   🔧 Solución: {solution['solution']}")
                    print(f"   ⏱️  Tiempo: {solution['time_minutes']} minutos")
                    print(f"   🏷️  Categoría: {solution['category']}")
            else:
                print("❌ No se encontraron soluciones")
        
        elif choice == "4":
            categories = practical.get_categories()
            print(f"\n📂 CATEGORÍAS DISPONIBLES ({len(categories)}):")
            print("=" * 30)
            
            for i, cat in enumerate(categories, 1):
                solutions = practical.get_practical_solutions(cat)
                print(f"   {i}. {cat} ({len(solutions)} soluciones)")
        
        elif choice == "5":
            print("\n👋 ¡Hasta luego! Implementa las soluciones prácticas y resuelve problemas reales.")
            break
        
        else:
            print("❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()





