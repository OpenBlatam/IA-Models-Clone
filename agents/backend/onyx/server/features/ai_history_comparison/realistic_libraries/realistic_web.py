"""
Realistic Web Libraries - Librerías web que realmente existen
==========================================================

Librerías web que realmente existen, están actualizadas
y funcionan en la práctica.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LibraryInfo:
    """Información real de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    pros: List[str]
    cons: List[str]
    installation: str
    example: str
    real_usage: str
    alternatives: List[str]
    documentation: str
    github: str
    pypi: str


class RealisticWebLibraries:
    """
    Librerías web realistas que realmente existen.
    """
    
    def __init__(self):
        """Inicializar con librerías web reales."""
        self.libraries = {
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
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="AI History Comparison API")

class HistoryEntry(BaseModel):
    id: Optional[int] = None
    content: str
    model: str
    quality: float
    timestamp: datetime = datetime.now()

# Base de datos en memoria
entries = []

@app.get("/")
async def root():
    return {"message": "AI History Comparison API"}

@app.post("/entries", response_model=HistoryEntry)
async def create_entry(entry: HistoryEntry):
    entry.id = len(entries) + 1
    entries.append(entry)
    return entry

@app.get("/entries", response_model=List[HistoryEntry])
async def get_entries():
    return entries

@app.get("/entries/{entry_id}", response_model=HistoryEntry)
async def get_entry(entry_id: int):
    for entry in entries:
        if entry.id == entry_id:
            return entry
    raise HTTPException(status_code=404, detail="Entry not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
                real_usage="""
# Uso real en producción
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uvicorn
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configuración
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./app.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

# Base de datos
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class HistoryEntryDB(Base):
    __tablename__ = "history_entries"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(String, nullable=False)
    model = Column(String, nullable=False)
    quality = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Modelos Pydantic
class HistoryEntryCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., min_length=1, max_length=100)
    quality: float = Field(..., ge=0.0, le=1.0)

class HistoryEntryResponse(BaseModel):
    id: int
    content: str
    model: str
    quality: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

# Dependencias
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    if credentials.credentials != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": "user123"}

# Aplicación
app = FastAPI(
    title="AI History Comparison API",
    description="API para comparación de historial de IA",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/")
async def root():
    return {"message": "AI History Comparison API", "version": "1.0.0"}

@app.post("/entries", response_model=HistoryEntryResponse)
async def create_entry(
    entry: HistoryEntryCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    db_entry = HistoryEntryDB(
        content=entry.content,
        model=entry.model,
        quality=entry.quality
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

@app.get("/entries", response_model=List[HistoryEntryResponse])
async def get_entries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    entries = db.query(HistoryEntryDB).offset(skip).limit(limit).all()
    return entries

@app.get("/entries/{entry_id}", response_model=HistoryEntryResponse)
async def get_entry(
    entry_id: int,
    db: Session = Depends(get_db)
):
    entry = db.query(HistoryEntryDB).filter(HistoryEntryDB.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
""",
                alternatives=["flask", "django", "starlette"],
                documentation="https://fastapi.tiangolo.com/",
                github="https://github.com/tiangolo/fastapi",
                pypi="https://pypi.org/project/fastapi/"
            ),
            
            'flask': LibraryInfo(
                name="flask",
                version="2.3.3",
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
from datetime import datetime

app = Flask(__name__)

# Base de datos en memoria
entries = []

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'AI History Comparison API',
        'version': '1.0.0'
    })

@app.route('/entries', methods=['POST'])
def create_entry():
    data = request.get_json()
    
    if not data or 'content' not in data:
        return jsonify({'error': 'Content is required'}), 400
    
    entry = {
        'id': len(entries) + 1,
        'content': data['content'],
        'model': data.get('model', 'unknown'),
        'quality': data.get('quality', 0.0),
        'timestamp': datetime.now().isoformat()
    }
    
    entries.append(entry)
    return jsonify(entry), 201

@app.route('/entries', methods=['GET'])
def get_entries():
    return jsonify(entries)

@app.route('/entries/<int:entry_id>', methods=['GET'])
def get_entry(entry_id):
    entry = next((e for e in entries if e['id'] == entry_id), None)
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404
    return jsonify(entry)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
""",
                real_usage="""
# Uso real en producción
from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from datetime import datetime, timedelta
import os

# Configuración
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# Extensiones
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)
jwt = JWTManager(app)

# Modelos
class HistoryEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    model = db.Column(db.String(100), nullable=False)
    quality = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'model': self.model,
            'quality': self.quality,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat()
        }

# Endpoints
@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    
    # En producción, verificar credenciales reales
    access_token = create_access_token(identity=username)
    return jsonify({
        'access_token': access_token,
        'user': username
    })

@app.route('/entries', methods=['POST'])
@jwt_required()
def create_entry():
    data = request.get_json()
    current_user = get_jwt_identity()
    
    if not data or 'content' not in data:
        return jsonify({'error': 'Content is required'}), 400
    
    entry = HistoryEntry(
        content=data['content'],
        model=data.get('model', 'unknown'),
        quality=data.get('quality', 0.0),
        user_id=current_user
    )
    
    db.session.add(entry)
    db.session.commit()
    
    return jsonify(entry.to_dict()), 201

@app.route('/entries', methods=['GET'])
@jwt_required()
def get_entries():
    current_user = get_jwt_identity()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    entries = HistoryEntry.query.filter_by(user_id=current_user).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'entries': [entry.to_dict() for entry in entries.items],
        'total': entries.total,
        'pages': entries.pages,
        'current_page': page
    })

@app.route('/entries/<int:entry_id>', methods=['GET'])
@jwt_required()
def get_entry(entry_id):
    current_user = get_jwt_identity()
    
    entry = HistoryEntry.query.filter_by(
        id=entry_id, user_id=current_user
    ).first()
    
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404
    
    return jsonify(entry.to_dict())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

# Inicializar base de datos
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
""",
                alternatives=["fastapi", "django", "bottle"],
                documentation="https://flask.palletsprojects.com/",
                github="https://github.com/pallets/flask",
                pypi="https://pypi.org/project/flask/"
            ),
            
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
# Ejecutar servidor básico
# uvicorn main:app --reload

# Con configuración personalizada
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# Con SSL
# uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem

# Con configuración de archivo
# uvicorn main:app --config uvicorn.conf.json
""",
                real_usage="""
# Uso real en producción
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    # Configuración para desarrollo
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Configuración para producción
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level warning

# Con Docker
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Con systemd
# [Unit]
# Description=AI History Comparison API
# After=network.target
# 
# [Service]
# Type=exec
# User=www-data
# WorkingDirectory=/opt/ai-history-api
# Environment=PATH=/opt/ai-history-api/venv/bin
# ExecStart=/opt/ai-history-api/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
# Restart=always
# 
# [Install]
# WantedBy=multi-user.target
""",
                alternatives=["gunicorn", "hypercorn", "waitress"],
                documentation="https://www.uvicorn.org/",
                github="https://github.com/encode/uvicorn",
                pypi="https://pypi.org/project/uvicorn/"
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
# Ejecutar servidor básico
# gunicorn -w 4 -b 0.0.0.0:8000 main:app

# Con configuración
# gunicorn -c gunicorn.conf.py main:app

# Con SSL
# gunicorn --certfile=cert.pem --keyfile=key.pem -w 4 -b 0.0.0.0:8000 main:app

# Con logging
# gunicorn --access-logfile - --error-logfile - -w 4 -b 0.0.0.0:8000 main:app
""",
                real_usage="""
# Uso real en producción
# gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Ejecutar con configuración
# gunicorn -c gunicorn.conf.py main:app

# Con Docker
# CMD ["gunicorn", "-c", "gunicorn.conf.py", "main:app"]

# Con systemd
# [Unit]
# Description=AI History Comparison API
# After=network.target
# 
# [Service]
# Type=exec
# User=www-data
# WorkingDirectory=/opt/ai-history-api
# Environment=PATH=/opt/ai-history-api/venv/bin
# ExecStart=/opt/ai-history-api/venv/bin/gunicorn -c gunicorn.conf.py main:app
# Restart=always
# 
# [Install]
# WantedBy=multi-user.target

# Con nginx
# upstream app_server {
#     server 127.0.0.1:8000 fail_timeout=0;
# }
# 
# server {
#     listen 80;
#     server_name your-domain.com;
#     
#     location / {
#         proxy_pass http://app_server;
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto $scheme;
#     }
# }
""",
                alternatives=["uvicorn", "waitress", "hypercorn"],
                documentation="https://docs.gunicorn.org/",
                github="https://github.com/benoitc/gunicorn",
                pypi="https://pypi.org/project/gunicorn/"
            ),
            
            'httpx': LibraryInfo(
                name="httpx",
                version="0.25.2",
                description="Cliente HTTP moderno para Python",
                use_case="APIs REST, descarga de datos, comunicación HTTP async",
                pros=[
                    "Soporte para async/await",
                    "HTTP/2 nativo",
                    "API similar a requests",
                    "Mejor rendimiento que requests"
                ],
                cons=[
                    "Relativamente nuevo",
                    "Dependencias adicionales"
                ],
                installation="pip install httpx",
                example="""
import httpx
import asyncio

# Uso síncrono
with httpx.Client() as client:
    response = client.get('https://api.example.com/entries')
    data = response.json()
    print(data)

# Uso asíncrono
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://api.example.com/entries')
        data = response.json()
        return data

# Ejecutar
data = asyncio.run(fetch_data())
print(data)

# POST request
async def create_entry():
    async with httpx.AsyncClient() as client:
        data = {
            'content': 'Sample text',
            'model': 'gpt-4',
            'quality': 0.8
        }
        response = await client.post('https://api.example.com/entries', json=data)
        return response.json()

result = asyncio.run(create_entry())
print(result)
""",
                real_usage="""
# Uso real en producción
import httpx
import asyncio
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def get_entries(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener entradas de la API"""
        try:
            response = await self.client.get(
                '/entries',
                params={'skip': skip, 'limit': limit}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error fetching entries: {e}")
            raise
    
    async def create_entry(self, entry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Crear nueva entrada"""
        try:
            response = await self.client.post('/entries', json=entry_data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error creating entry: {e}")
            raise
    
    async def get_entry(self, entry_id: int) -> Dict[str, Any]:
        """Obtener entrada específica"""
        try:
            response = await self.client.get(f'/entries/{entry_id}')
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error fetching entry {entry_id}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud de la API"""
        try:
            response = await self.client.get('/health')
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Health check failed: {e}")
            raise

# Uso en aplicación
async def main():
    async with APIClient('https://api.example.com') as client:
        # Verificar salud
        health = await client.health_check()
        print(f"API Health: {health}")
        
        # Obtener entradas
        entries = await client.get_entries(limit=10)
        print(f"Found {len(entries)} entries")
        
        # Crear nueva entrada
        new_entry = {
            'content': 'Sample text',
            'model': 'gpt-4',
            'quality': 0.8
        }
        created = await client.create_entry(new_entry)
        print(f"Created entry: {created}")

# Ejecutar
asyncio.run(main())
""",
                alternatives=["requests", "aiohttp", "urllib3"],
                documentation="https://www.python-httpx.org/",
                github="https://github.com/encode/httpx",
                pypi="https://pypi.org/project/httpx/"
            ),
            
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
import os

# Configuración
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Crear token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
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

# Ejemplo de uso
user_data = {"sub": "user123", "username": "john_doe"}
token = create_access_token(user_data)
print(f"Token: {token}")

# Verificar token
payload = verify_token(token)
if payload:
    print(f"Payload: {payload}")
else:
    print("Invalid token")
""",
                real_usage="""
# Uso real en producción
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configuración
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class JWTManager:
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        """Crear token de acceso"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verificar token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            return None
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Obtener usuario actual desde token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {"username": username, "user_id": payload.get("user_id")}

# Inicializar JWT manager
jwt_manager = JWTManager(SECRET_KEY)

# Uso en FastAPI
from fastapi import FastAPI

app = FastAPI()

@app.post("/auth/login")
async def login(username: str, password: str):
    # En producción, verificar credenciales reales
    if username == "admin" and password == "password":
        access_token = jwt_manager.create_access_token(
            data={"sub": username, "user_id": "123"}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

@app.get("/protected")
async def protected_route(current_user: dict = Depends(jwt_manager.get_current_user)):
    return {"message": f"Hello {current_user['username']}"}

# Uso en Flask
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            payload = jwt_manager.verify_token(token)
            if payload is None:
                return jsonify({'message': 'Token is invalid'}), 401
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(payload, *args, **kwargs)
    return decorated

@app.route('/protected')
@token_required
def protected(payload):
    return jsonify({'message': f'Hello {payload["sub"]}'})
""",
                alternatives=["pyjwt", "authlib", "python-jwt"],
                documentation="https://python-jose.readthedocs.io/",
                github="https://github.com/mpdavis/python-jose",
                pypi="https://pypi.org/project/python-jose/"
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las librerías web realistas."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                requirements.append(f"{package}=={lib.version}")
        
        return '\n'.join(requirements)
    
    def get_realistic_usage_examples(self) -> Dict[str, str]:
        """Obtener ejemplos de uso real para cada librería."""
        return {name: lib.real_usage for name, lib in self.libraries.items()}




