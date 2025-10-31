#!/usr/bin/env python3
"""
Quick Things - Cosas rápidas y funcionales
Implementaciones que puedes usar en 5 minutos
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any

class QuickThings:
    """Cosas rápidas que funcionan inmediatamente"""
    
    def __init__(self):
        self.things = []
        self.metrics = {}
    
    def add_quick_thing(self, name: str, description: str, code: str, time_minutes: int):
        """Añadir cosa rápida"""
        thing = {
            "name": name,
            "description": description,
            "code": code,
            "time_minutes": time_minutes,
            "created_at": datetime.now().isoformat(),
            "is_implemented": False
        }
        self.things.append(thing)
        print(f"✅ Cosa rápida añadida: {name}")
    
    def get_quick_things(self) -> List[Dict[str, Any]]:
        """Obtener cosas rápidas"""
        return sorted(self.things, key=lambda x: x['time_minutes'])
    
    def implement_quick_thing(self, index: int) -> bool:
        """Implementar cosa rápida"""
        if 0 <= index < len(self.things):
            thing = self.things[index]
            print(f"🚀 Implementando: {thing['name']}")
            print(f"⏱️  Tiempo estimado: {thing['time_minutes']} minutos")
            print(f"📝 Descripción: {thing['description']}")
            print(f"💻 Código:")
            print(thing['code'])
            
            thing['is_implemented'] = True
            thing['implemented_at'] = datetime.now().isoformat()
            
            print(f"✅ {thing['name']} implementado!")
            return True
        return False
    
    def show_quick_things(self):
        """Mostrar cosas rápidas"""
        print("\n⚡ COSAS RÁPIDAS DISPONIBLES")
        print("=" * 40)
        
        for i, thing in enumerate(self.things):
            status = "✅ IMPLEMENTADO" if thing['is_implemented'] else "⏳ PENDIENTE"
            print(f"\n{i+1}. {thing['name']}")
            print(f"   ⏱️  Tiempo: {thing['time_minutes']} minutos")
            print(f"   📝 {thing['description']}")
            print(f"   {status}")

def create_quick_things():
    """Crear cosas rápidas reales"""
    quick = QuickThings()
    
    # 1. Health Check (2 minutos)
    quick.add_quick_thing(
        "Health Check Endpoint",
        "Endpoint básico de salud para monitoreo",
        '''@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }''',
        2
    )
    
    # 2. Request ID (3 minutos)
    quick.add_quick_thing(
        "Request ID Tracking",
        "ID único para cada request para debugging",
        '''import uuid
from fastapi import Request

@app.middleware('http')
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers['X-Request-ID'] = request_id
    return response''',
        3
    )
    
    # 3. CORS (1 minuto)
    quick.add_quick_thing(
        "CORS Configuration",
        "Configuración básica de CORS",
        '''from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)''',
        1
    )
    
    # 4. GZip Compression (1 minuto)
    quick.add_quick_thing(
        "Response Compression",
        "Comprimir respuestas para mejor performance",
        '''from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)''',
        1
    )
    
    # 5. Basic Logging (4 minutos)
    quick.add_quick_thing(
        "Basic Logging",
        "Logging básico para debugging",
        '''import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.middleware('http')
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f'{request.method} {request.url.path} - '
        f'{response.status_code} - {process_time:.3f}s'
    )
    return response''',
        4
    )
    
    # 6. Environment Variables (2 minutos)
    quick.add_quick_thing(
        "Environment Configuration",
        "Usar variables de entorno para configuración",
        '''from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./app.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
API_KEY = os.getenv('API_KEY', '')''',
        2
    )
    
    # 7. Basic Error Handling (5 minutos)
    quick.add_quick_thing(
        "Basic Error Handling",
        "Manejo básico de errores con mensajes útiles",
        '''from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

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
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )''',
        5
    )
    
    # 8. Basic Metrics (3 minutos)
    quick.add_quick_thing(
        "Basic Metrics",
        "Métricas básicas de la aplicación",
        '''from collections import defaultdict
import time

class BasicMetrics:
    def __init__(self):
        self.request_count = 0
        self.response_times = []
        self.error_count = 0
    
    def record_request(self, response_time: float, status_code: int):
        self.request_count += 1
        self.response_times.append(response_time)
        if status_code >= 400:
            self.error_count += 1
    
    def get_stats(self):
        return {
            "total_requests": self.request_count,
            "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }

metrics = BasicMetrics()

@app.get("/metrics")
async def get_metrics():
    return metrics.get_stats()''',
        3
    )
    
    # 9. Database Index (2 minutos)
    quick.add_quick_thing(
        "Database Index",
        "Índice básico para mejorar consultas",
        '''-- Ejecutar en tu base de datos
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);''',
        2
    )
    
    # 10. Basic Validation (4 minutos)
    quick.add_quick_thing(
        "Basic Input Validation",
        "Validación básica con Pydantic",
        '''from pydantic import BaseModel, EmailStr, validator
from typing import Optional

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

@app.post("/users/")
async def create_user(user: UserCreate):
    # La validación es automática con Pydantic
    return {"message": "User created successfully", "user": user.dict()}''',
        4
    )
    
    return quick

def main():
    """Función principal"""
    print("⚡ QUICK THINGS - Cosas rápidas y funcionales")
    print("=" * 50)
    
    # Crear cosas rápidas
    quick = create_quick_things()
    
    # Mostrar menú
    while True:
        print("\n🎯 MENÚ DE COSAS RÁPIDAS")
        print("1. Ver todas las cosas rápidas")
        print("2. Implementar cosa específica")
        print("3. Implementar todas las cosas (menos de 5 min)")
        print("4. Ver estadísticas")
        print("5. Salir")
        
        choice = input("\nSelecciona una opción (1-5): ").strip()
        
        if choice == "1":
            quick.show_quick_things()
        
        elif choice == "2":
            quick.show_quick_things()
            try:
                index = int(input("\nSelecciona el número de la cosa a implementar: ")) - 1
                quick.implement_quick_thing(index)
            except ValueError:
                print("❌ Por favor ingresa un número válido")
        
        elif choice == "3":
            print("\n🚀 Implementando todas las cosas rápidas...")
            for i in range(len(quick.things)):
                if quick.things[i]['time_minutes'] <= 5:
                    quick.implement_quick_thing(i)
                    time.sleep(1)  # Pausa para mostrar progreso
        
        elif choice == "4":
            things = quick.get_quick_things()
            implemented = len([t for t in things if t['is_implemented']])
            total_time = sum(t['time_minutes'] for t in things)
            implemented_time = sum(t['time_minutes'] for t in things if t['is_implemented'])
            
            print(f"\n📊 ESTADÍSTICAS:")
            print(f"   Total de cosas: {len(things)}")
            print(f"   Implementadas: {implemented}")
            print(f"   Tiempo total: {total_time} minutos")
            print(f"   Tiempo implementado: {implemented_time} minutos")
            print(f"   Progreso: {(implemented/len(things)*100):.1f}%")
        
        elif choice == "5":
            print("\n👋 ¡Hasta luego! Implementa las cosas rápidas y verás resultados inmediatos.")
            break
        
        else:
            print("❌ Opción inválida")
        
        input("\nPresiona Enter para continuar...")

if __name__ == "__main__":
    main()





