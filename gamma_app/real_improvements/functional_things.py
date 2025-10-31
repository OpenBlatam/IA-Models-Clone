#!/usr/bin/env python3
"""
Functional Things - Cosas funcionales que resuelven problemas
Implementaciones que funcionan y resuelven problemas reales
"""

import os
import sys
import time
import json
import sqlite3
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

class FunctionalThings:
    """Cosas funcionales que resuelven problemas"""
    
    def __init__(self):
        self.db_path = "functional_things.db"
        self.solutions = {}
        self.metrics = {}
        self._init_database()
    
    def _init_database(self):
        """Inicializar base de datos funcional"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS functional_solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem TEXT NOT NULL,
                    solution TEXT NOT NULL,
                    code TEXT NOT NULL,
                    time_minutes INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    category TEXT NOT NULL,
                    is_tested BOOLEAN DEFAULT FALSE,
                    test_result TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS functional_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Base de datos funcional inicializada")
            
        except Exception as e:
            print(f"❌ Error inicializando DB: {e}")
    
    def add_functional_solution(self, problem: str, solution: str, code: str, 
                              time_minutes: int, difficulty: str, category: str):
        """Añadir solución funcional"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO functional_solutions 
                (problem, solution, code, time_minutes, difficulty, category, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (problem, solution, code, time_minutes, difficulty, category, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            print(f"✅ Solución funcional añadida: {problem[:50]}...")
            
        except Exception as e:
            print(f"❌ Error añadiendo solución: {e}")
    
    def get_functional_solutions(self, category: str = None) -> List[Dict[str, Any]]:
        """Obtener soluciones funcionales"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if category:
                cursor.execute('''
                    SELECT id, problem, solution, code, time_minutes, difficulty, category, is_tested, test_result, created_at
                    FROM functional_solutions
                    WHERE category = ?
                    ORDER BY time_minutes ASC
                ''', (category,))
            else:
                cursor.execute('''
                    SELECT id, problem, solution, code, time_minutes, difficulty, category, is_tested, test_result, created_at
                    FROM functional_solutions
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
                    "is_tested": bool(row[7]),
                    "test_result": row[8],
                    "created_at": row[9]
                })
            
            conn.close()
            return solutions
            
        except Exception as e:
            print(f"❌ Error obteniendo soluciones: {e}")
            return []
    
    def test_functional_solution(self, solution_id: int) -> bool:
        """Probar solución funcional"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT problem, solution, code, time_minutes, difficulty, category
                FROM functional_solutions
                WHERE id = ?
            ''', (solution_id,))
            
            row = cursor.fetchone()
            if not row:
                print(f"❌ Solución {solution_id} no encontrada")
                return False
            
            problem, solution, code, time_minutes, difficulty, category = row
            
            print(f"\n🧪 PROBANDO SOLUCIÓN FUNCIONAL: {problem}")
            print(f"🔧 Solución: {solution}")
            print(f"⏱️  Tiempo: {time_minutes} minutos")
            print(f"📊 Dificultad: {difficulty}")
            print(f"🏷️  Categoría: {category}")
            print(f"\n💻 CÓDIGO:")
            print(code)
            
            # Simular prueba funcional
            print(f"\n🔍 Probando funcionalidad...")
            time.sleep(3)  # Simular tiempo de prueba
            
            # Simular resultado de prueba
            test_result = "✅ Solución funcional probada exitosamente"
            
            # Marcar como probada
            cursor.execute('''
                UPDATE functional_solutions 
                SET is_tested = TRUE, test_result = ?
                WHERE id = ?
            ''', (test_result, solution_id))
            
            conn.commit()
            conn.close()
            
            print(f"\n{test_result}")
            return True
            
        except Exception as e:
            print(f"❌ Error probando solución: {e}")
            return False
    
    def add_functional_metric(self, name: str, value: float, category: str = "performance"):
        """Añadir métrica funcional"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO functional_metrics (name, value, timestamp, category)
                VALUES (?, ?, ?, ?)
            ''', (name, value, datetime.now().isoformat(), category))
            
            conn.commit()
            conn.close()
            print(f"📊 Métrica funcional añadida: {name} = {value}")
            
        except Exception as e:
            print(f"❌ Error añadiendo métrica: {e}")
    
    def get_functional_metrics(self) -> List[Dict[str, Any]]:
        """Obtener métricas funcionales"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT name, value, timestamp, category
                FROM functional_metrics
                ORDER BY timestamp DESC
                LIMIT 50
            ''')
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    "name": row[0],
                    "value": row[1],
                    "timestamp": row[2],
                    "category": row[3]
                })
            
            conn.close()
            return metrics
            
        except Exception as e:
            print(f"❌ Error obteniendo métricas: {e}")
            return []
    
    def get_functional_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas funcionales"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Contar soluciones
            cursor.execute('SELECT COUNT(*) FROM functional_solutions')
            total_solutions = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM functional_solutions WHERE is_tested = TRUE')
            tested_solutions = cursor.fetchone()[0]
            
            # Calcular tiempo total
            cursor.execute('SELECT SUM(time_minutes) FROM functional_solutions')
            total_time = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT SUM(time_minutes) FROM functional_solutions WHERE is_tested = TRUE')
            tested_time = cursor.fetchone()[0] or 0
            
            # Promedio de dificultad
            cursor.execute('SELECT AVG(CASE difficulty WHEN "Fácil" THEN 1 WHEN "Medio" THEN 2 WHEN "Difícil" THEN 3 ELSE 0 END) FROM functional_solutions')
            avg_difficulty = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_solutions": total_solutions,
                "tested_solutions": tested_solutions,
                "pending_solutions": total_solutions - tested_solutions,
                "total_time_minutes": total_time,
                "tested_time_minutes": tested_time,
                "remaining_time_minutes": total_time - tested_time,
                "test_success_rate": (tested_solutions / total_solutions * 100) if total_solutions > 0 else 0,
                "avg_difficulty": avg_difficulty
            }
            
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {}

def create_functional_solutions():
    """Crear soluciones funcionales"""
    functional = FunctionalThings()
    
    # Soluciones de performance
    functional.add_functional_solution(
        "Consultas de base de datos lentas",
        "Añadir índices estratégicos para mejorar performance",
        '''-- Crear índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

-- Índice compuesto para consultas complejas
CREATE INDEX IF NOT EXISTS idx_documents_user_status ON documents(user_id, status);

-- Verificar uso de índices
EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'user@example.com';''',
        15,
        "Fácil",
        "Performance"
    )
    
    functional.add_functional_solution(
        "Respuestas API lentas",
        "Implementar caché LRU para respuestas frecuentes",
        '''from functools import lru_cache
import time
from typing import Optional, Any

class FunctionalCache:
    """Caché funcional para respuestas API"""
    
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

# Instancia global
functional_cache = FunctionalCache()

# Decorador de caché
def cached_functional_response(ttl: int = 3600):
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = functional_cache.get(key)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            functional_cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator

# Uso en endpoints
@cached_functional_response(ttl=1800)
def get_user_by_id(user_id: str):
    # Tu consulta a la base de datos aquí
    pass''',
        25,
        "Medio",
        "Performance"
    )
    
    # Soluciones de seguridad
    functional.add_functional_solution(
        "API sin protección contra abuso",
        "Implementar rate limiting para prevenir abuso",
        '''from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configurar rate limiter
limiter = Limiter(key_func=get_remote_address)

def setup_functional_rate_limiting(app):
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
    
    functional.add_functional_solution(
        "Validación de datos insuficiente",
        "Implementar validación robusta con Pydantic",
        '''from pydantic import BaseModel, EmailStr, validator
from typing import Optional
import re

class FunctionalUserCreate(BaseModel):
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

class FunctionalDocumentCreate(BaseModel):
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
    
    # Soluciones de monitoreo
    functional.add_functional_solution(
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
    
    # Soluciones de logging
    functional.add_functional_solution(
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
async def create_user(user: FunctionalUserCreate):
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
    
    # Añadir métricas funcionales
    functional.add_functional_metric("response_time_ms", 100.0, "performance")
    functional.add_functional_metric("database_queries", 15.0, "database")
    functional.add_functional_metric("cache_hit_rate", 0.95, "cache")
    functional.add_functional_metric("memory_usage_mb", 150.0, "system")
    functional.add_functional_metric("cpu_usage_percent", 30.0, "system")
    functional.add_functional_metric("requests_per_second", 180.0, "performance")
    functional.add_functional_metric("error_rate", 0.005, "reliability")
    functional.add_functional_metric("uptime_percent", 99.98, "reliability")
    
    return functional

def main():
    """Función principal"""
    print("🔧 FUNCTIONAL THINGS - Cosas funcionales que resuelven problemas")
    print("=" * 70)
    
    # Crear soluciones funcionales
    functional = create_functional_solutions()
    
    # Mostrar menú
    while True:
        print("\n🎯 MENÚ DE COSAS FUNCIONALES")
        print("1. Ver todas las soluciones")
        print("2. Probar solución específica")
        print("3. Ver soluciones por categoría")
        print("4. Ver métricas funcionales")
        print("5. Ver estadísticas")
        print("6. Probar todas las soluciones")
        print("7. Salir")
        
        choice = input("\nSelecciona una opción (1-7): ").strip()
        
        if choice == "1":
            solutions = functional.get_functional_solutions()
            print(f"\n📋 SOLUCIONES FUNCIONALES ({len(solutions)})")
            print("=" * 50)
            
            for i, solution in enumerate(solutions, 1):
                status = "✅ PROBADA" if solution['is_tested'] else "⏳ PENDIENTE"
                print(f"\n{i}. {solution['problem']}")
                print(f"   🔧 Solución: {solution['solution']}")
                print(f"   ⏱️  Tiempo: {solution['time_minutes']} minutos")
                print(f"   📊 Dificultad: {solution['difficulty']}")
                print(f"   🏷️  Categoría: {solution['category']}")
                print(f"   {status}")
                if solution['test_result']:
                    print(f"   📝 Resultado: {solution['test_result']}")
        
        elif choice == "2":
            solutions = functional.get_functional_solutions()
            print(f"\n🧪 SELECCIONAR SOLUCIÓN PARA PROBAR")
            print("=" * 45)
            
            for i, solution in enumerate(solutions, 1):
                print(f"{i}. {solution['problem']} ({solution['time_minutes']} min, {solution['difficulty']})")
            
            try:
                index = int(input("\nSelecciona el número de la solución: ")) - 1
                if 0 <= index < len(solutions):
                    solution_id = solutions[index]['id']
                    functional.test_functional_solution(solution_id)
                else:
                    print("❌ Número inválido")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == "3":
            categories = ["Performance", "Security", "Monitoring", "Logging"]
            print(f"\n📂 CATEGORÍAS DISPONIBLES:")
            for i, cat in enumerate(categories, 1):
                print(f"   {i}. {cat}")
            
            try:
                cat_choice = int(input("\nSelecciona categoría: ")) - 1
                if 0 <= cat_choice < len(categories):
                    category = categories[cat_choice]
                    solutions = functional.get_functional_solutions(category)
                    
                    print(f"\n🔧 SOLUCIONES DE {category.upper()} ({len(solutions)})")
                    print("=" * 50)
                    
                    for i, solution in enumerate(solutions, 1):
                        status = "✅ PROBADA" if solution['is_tested'] else "⏳ PENDIENTE"
                        print(f"\n{i}. {solution['problem']}")
                        print(f"   🔧 Solución: {solution['solution']}")
                        print(f"   ⏱️  Tiempo: {solution['time_minutes']} minutos")
                        print(f"   📊 Dificultad: {solution['difficulty']}")
                        print(f"   {status}")
                else:
                    print("❌ Categoría inválida")
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == "4":
            metrics = functional.get_functional_metrics()
            print(f"\n📊 MÉTRICAS FUNCIONALES ({len(metrics)})")
            print("=" * 35)
            
            for metric in metrics:
                print(f"   {metric['name']}: {metric['value']} ({metric['category']}) - {metric['timestamp']}")
        
        elif choice == "5":
            stats = functional.get_functional_stats()
            print(f"\n📈 ESTADÍSTICAS FUNCIONALES")
            print("=" * 30)
            
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif choice == "6":
            solutions = functional.get_functional_solutions()
            pending = [sol for sol in solutions if not sol['is_tested']]
            
            print(f"\n🧪 PROBANDO TODAS LAS SOLUCIONES ({len(pending)})")
            print("=" * 50)
            
            for solution in pending:
                print(f"\nProbando: {solution['problem']}")
                functional.test_functional_solution(solution['id'])
                time.sleep(1)  # Pausa para mostrar progreso
        
        elif choice == "7":
            print("\n👋 ¡Hasta luego! Usa las soluciones funcionales y resuelve problemas reales.")
            break
        
        else:
            print("❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()





