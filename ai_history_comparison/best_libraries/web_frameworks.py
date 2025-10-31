"""
Web Frameworks - Mejores frameworks web
=====================================

Las mejores librerías para desarrollo web y APIs.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class LibraryInfo:
    """Información de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    pros: List[str]
    cons: List[str]
    installation: str
    example: str


class WebFrameworks:
    """
    Mejores frameworks web para el sistema.
    """
    
    def __init__(self):
        """Inicializar con los mejores frameworks web."""
        self.libraries = {
            # Web Frameworks
            'fastapi': LibraryInfo(
                name="fastapi",
                version="0.104.1",
                description="Framework web moderno y rápido para APIs",
                use_case="APIs REST, documentación automática, validación de datos",
                pros=[
                    "Muy rápido (basado en Starlette y Pydantic)",
                    "Documentación automática (Swagger/OpenAPI)",
                    "Validación automática de datos",
                    "Soporte nativo para async/await",
                    "Type hints integrados"
                ],
                cons=[
                    "Relativamente nuevo",
                    "Menos middleware que Django"
                ],
                installation="pip install fastapi uvicorn",
                example="""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class HistoryEntry(BaseModel):
    content: str
    model: str
    quality: float

@app.post("/analyze")
async def analyze_content(entry: HistoryEntry):
    # Procesar entrada
    return {"message": "Content analyzed", "entry": entry}

@app.get("/")
async def root():
    return {"message": "AI History Comparison API"}

# Ejecutar: uvicorn main:app --reload
"""
            ),
            
            'flask': LibraryInfo(
                name="flask",
                version="2.3.0",
                description="Framework web ligero y flexible",
                use_case="APIs simples, aplicaciones web ligeras, prototipado",
                pros=[
                    "Muy simple y ligero",
                    "Flexible y extensible",
                    "Gran ecosistema de extensiones",
                    "Fácil de aprender"
                ],
                cons=[
                    "Menos funcionalidades out-of-the-box",
                    "Requiere más configuración manual"
                ],
                installation="pip install flask",
                example="""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_content():
    data = request.get_json()
    content = data.get('content')
    model = data.get('model')
    
    # Procesar contenido
    result = {"message": "Content analyzed", "content": content}
    return jsonify(result)

@app.route('/')
def root():
    return jsonify({"message": "AI History Comparison API"})

if __name__ == '__main__':
    app.run(debug=True)
"""
            ),
            
            'django': LibraryInfo(
                name="django",
                version="4.2.0",
                description="Framework web completo y robusto",
                use_case="Aplicaciones web complejas, CMS, sistemas empresariales",
                pros=[
                    "Framework completo con ORM, admin, etc.",
                    "Muy maduro y estable",
                    "Excelente documentación",
                    "Gran ecosistema"
                ],
                cons=[
                    "Puede ser pesado para APIs simples",
                    "Curva de aprendizaje más empinada"
                ],
                installation="pip install django djangorestframework",
                example="""
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'rest_framework',
    'myapp',
]

# models.py
from django.db import models

class HistoryEntry(models.Model):
    content = models.TextField()
    model = models.CharField(max_length=100)
    quality = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

# views.py
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

class HistoryEntryViewSet(viewsets.ModelViewSet):
    queryset = HistoryEntry.objects.all()
    
    @action(detail=False, methods=['post'])
    def analyze(self, request):
        # Lógica de análisis
        return Response({"message": "Content analyzed"})
"""
            ),
            
            # ASGI Servers
            'uvicorn': LibraryInfo(
                name="uvicorn",
                version="0.24.0",
                description="Servidor ASGI de alto rendimiento",
                use_case="Servidor para FastAPI, aplicaciones async",
                pros=[
                    "Muy rápido",
                    "Soporte completo para async/await",
                    "HTTP/2 y WebSockets",
                    "Fácil de usar"
                ],
                cons=[
                    "Solo para aplicaciones ASGI",
                    "Menos opciones de configuración que Gunicorn"
                ],
                installation="pip install uvicorn",
                example="""
# Ejecutar servidor
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Con configuración personalizada
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info
"""
            ),
            
            'gunicorn': LibraryInfo(
                name="gunicorn",
                version="21.2.0",
                description="Servidor WSGI para Python",
                use_case="Servidor para Flask, Django, aplicaciones WSGI",
                pros=[
                    "Muy estable y confiable",
                    "Configuración flexible",
                    "Soporte para múltiples workers",
                    "Ampliamente usado en producción"
                ],
                cons=[
                    "Solo para aplicaciones WSGI",
                    "No soporte nativo para async"
                ],
                installation="pip install gunicorn",
                example="""
# Ejecutar servidor
gunicorn -w 4 -b 0.0.0.0:8000 main:app

# Con configuración
gunicorn -c gunicorn.conf.py main:app
"""
            ),
            
            # API Documentation
            'swagger-ui': LibraryInfo(
                name="swagger-ui",
                version="5.0.0",
                description="Interfaz de usuario para documentación de API",
                use_case="Documentación interactiva de APIs, testing de endpoints",
                pros=[
                    "Interfaz intuitiva",
                    "Testing integrado",
                    "Documentación automática",
                    "Estándar de la industria"
                ],
                cons=[
                    "Requiere especificación OpenAPI",
                    "Puede ser pesado para APIs simples"
                ],
                installation="pip install swagger-ui-bundle",
                example="""
# Con FastAPI (automático)
from fastapi import FastAPI

app = FastAPI(
    title="AI History Comparison API",
    description="API for comparing AI model history",
    version="1.0.0"
)

# Documentación disponible en /docs
"""
            ),
            
            # Authentication
            'python-jose': LibraryInfo(
                name="python-jose",
                version="3.3.0",
                description="Implementación de JWT para Python",
                use_case="Autenticación JWT, tokens de acceso",
                pros=[
                    "Implementación completa de JWT",
                    "Soporte para múltiples algoritmos",
                    "Fácil de usar",
                    "Bien documentado"
                ],
                cons=[
                    "Requiere conocimiento de JWT",
                    "Configuración de seguridad importante"
                ],
                installation="pip install python-jose[cryptography]",
                example="""
from jose import jwt, JWTError
from datetime import datetime, timedelta

# Crear token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Verificar token
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
"""
            ),
            
            # CORS
            'fastapi-cors': LibraryInfo(
                name="fastapi-cors",
                version="0.0.6",
                description="Middleware CORS para FastAPI",
                use_case="Habilitar CORS en APIs, desarrollo frontend",
                pros=[
                    "Fácil de configurar",
                    "Soporte para múltiples orígenes",
                    "Configuración flexible",
                    "Integración nativa con FastAPI"
                ],
                cons=[
                    "Solo para FastAPI",
                    "Configuración de seguridad importante"
                ],
                installation="pip install fastapi-cors",
                example="""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
            ),
            
            # Rate Limiting
            'slowapi': LibraryInfo(
                name="slowapi",
                version="0.1.9",
                description="Rate limiting para FastAPI",
                use_case="Limitar requests por usuario, prevenir abuso",
                pros=[
                    "Fácil de usar",
                    "Configuración flexible",
                    "Soporte para múltiples backends",
                    "Integración con FastAPI"
                ],
                cons=[
                    "Requiere Redis o memoria para almacenar límites",
                    "Configuración adicional necesaria"
                ],
                installation="pip install slowapi",
                example="""
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/")
@limiter.limit("5/minute")
def read_root(request: Request):
    return {"message": "Hello World"}
"""
            ),
            
            # WebSockets
            'websockets': LibraryInfo(
                name="websockets",
                version="11.0.3",
                description="Implementación de WebSockets para Python",
                use_case="Comunicación en tiempo real, notificaciones, chat",
                pros=[
                    "Implementación completa de WebSockets",
                    "Soporte para async/await",
                    "Fácil de usar",
                    "Bien documentado"
                ],
                cons=[
                    "Requiere conocimiento de WebSockets",
                    "Configuración de conexiones persistentes"
                ],
                installation="pip install websockets",
                example="""
import asyncio
import websockets

async def handle_client(websocket, path):
    async for message in websocket:
        # Procesar mensaje
        response = f"Received: {message}"
        await websocket.send(response)

# Iniciar servidor
start_server = websockets.serve(handle_client, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
"""
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_libraries_by_category(self, category: str) -> Dict[str, LibraryInfo]:
        """Obtener librerías por categoría."""
        categories = {
            'frameworks': ['fastapi', 'flask', 'django'],
            'servers': ['uvicorn', 'gunicorn'],
            'documentation': ['swagger-ui'],
            'authentication': ['python-jose'],
            'middleware': ['fastapi-cors', 'slowapi'],
            'websockets': ['websockets']
        }
        
        if category not in categories:
            return {}
        
        return {name: self.libraries[name] for name in categories[category] if name in self.libraries}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con los mejores frameworks web."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)




