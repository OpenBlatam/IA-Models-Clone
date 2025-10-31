"""
Database Libraries - Mejores librerías de base de datos
=====================================================

Las mejores librerías para manejo de bases de datos.
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


class DatabaseLibraries:
    """
    Mejores librerías para manejo de bases de datos.
    """
    
    def __init__(self):
        """Inicializar con las mejores librerías de base de datos."""
        self.libraries = {
            # SQL Databases
            'sqlalchemy': LibraryInfo(
                name="sqlalchemy",
                version="2.0.23",
                description="ORM y toolkit de base de datos para Python",
                use_case="ORM, consultas SQL, migraciones, múltiples bases de datos",
                pros=[
                    "ORM muy potente y flexible",
                    "Soporte para múltiples bases de datos",
                    "Sistema de migraciones",
                    "Query builder avanzado",
                    "Excelente documentación"
                ],
                cons=[
                    "Curva de aprendizaje",
                    "Puede ser complejo para casos simples"
                ],
                installation="pip install sqlalchemy",
                example="""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class HistoryEntry(Base):
    __tablename__ = 'history_entries'
    
    id = Column(Integer, primary_key=True)
    content = Column(String)
    model = Column(String)
    quality = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Crear engine y sesión
engine = create_engine('sqlite:///ai_history.db')
Session = sessionmaker(bind=engine)
session = Session()

# Crear entrada
entry = HistoryEntry(content="Sample text", model="gpt-4", quality=0.8)
session.add(entry)
session.commit()
"""
            ),
            
            'alembic': LibraryInfo(
                name="alembic",
                version="1.12.1",
                description="Sistema de migraciones para SQLAlchemy",
                use_case="Migraciones de base de datos, versionado de esquemas",
                pros=[
                    "Integración perfecta con SQLAlchemy",
                    "Migraciones automáticas",
                    "Versionado de esquemas",
                    "Rollback de cambios"
                ],
                cons=[
                    "Solo para SQLAlchemy",
                    "Requiere configuración inicial"
                ],
                installation="pip install alembic",
                example="""
# Inicializar Alembic
alembic init alembic

# Crear migración
alembic revision --autogenerate -m "Add history entries table"

# Aplicar migración
alembic upgrade head

# Rollback
alembic downgrade -1
"""
            ),
            
            'psycopg2': LibraryInfo(
                name="psycopg2",
                version="2.9.7",
                description="Adaptador PostgreSQL para Python",
                use_case="Conexión a PostgreSQL, consultas SQL directas",
                pros=[
                    "Adaptador oficial de PostgreSQL",
                    "Muy rápido y eficiente",
                    "Soporte completo para PostgreSQL",
                    "Transacciones ACID"
                ],
                cons=[
                    "Solo para PostgreSQL",
                    "Requiere instalación de PostgreSQL"
                ],
                installation="pip install psycopg2-binary",
                example="""
import psycopg2
from psycopg2.extras import RealDictCursor

# Conectar a PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="ai_history",
    user="username",
    password="password"
)

# Ejecutar consulta
with conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute("SELECT * FROM history_entries WHERE quality > %s", (0.7,))
    results = cur.fetchall()

conn.close()
"""
            ),
            
            'pymongo': LibraryInfo(
                name="pymongo",
                version="4.6.0",
                description="Driver oficial de MongoDB para Python",
                use_case="Bases de datos NoSQL, documentos JSON, escalabilidad horizontal",
                pros=[
                    "Driver oficial de MongoDB",
                    "Soporte completo para MongoDB",
                    "Escalabilidad horizontal",
                    "Flexibilidad de esquemas"
                ],
                cons=[
                    "Solo para MongoDB",
                    "Requiere conocimiento de NoSQL"
                ],
                installation="pip install pymongo",
                example="""
from pymongo import MongoClient
from datetime import datetime

# Conectar a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['ai_history']
collection = db['entries']

# Insertar documento
entry = {
    'content': 'Sample text',
    'model': 'gpt-4',
    'quality': 0.8,
    'created_at': datetime.utcnow()
}
result = collection.insert_one(entry)

# Consultar documentos
entries = collection.find({'quality': {'$gt': 0.7}})
for entry in entries:
    print(entry)
"""
            ),
            
            'redis': LibraryInfo(
                name="redis",
                version="5.0.1",
                description="Cliente Redis para Python",
                use_case="Caché, sesiones, colas, datos en memoria",
                pros=[
                    "Muy rápido (en memoria)",
                    "Múltiples estructuras de datos",
                    "Persistencia opcional",
                    "Escalabilidad horizontal"
                ],
                cons=[
                    "Limitado por memoria RAM",
                    "Requiere conocimiento de Redis"
                ],
                installation="pip install redis",
                example="""
import redis
import json

# Conectar a Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Almacenar datos
entry_data = {'content': 'Sample text', 'quality': 0.8}
r.set('entry:1', json.dumps(entry_data))

# Recuperar datos
data = r.get('entry:1')
entry = json.loads(data)

# Usar como caché
r.setex('cache:key', 3600, 'cached_value')
"""
            ),
            
            # Database Tools
            'databases': LibraryInfo(
                name="databases",
                version="0.8.0",
                description="Librería de base de datos async para Python",
                use_case="Bases de datos async, FastAPI, aplicaciones de alto rendimiento",
                pros=[
                    "Soporte nativo para async/await",
                    "Integración con FastAPI",
                    "Múltiples bases de datos",
                    "Alto rendimiento"
                ],
                cons=[
                    "Relativamente nuevo",
                    "Menos funcionalidades que SQLAlchemy"
                ],
                installation="pip install databases[postgresql]",
                example="""
import databases
import asyncio

# Crear conexión async
database = databases.Database('postgresql://user:pass@localhost/db')

async def main():
    await database.connect()
    
    # Ejecutar consulta
    query = "SELECT * FROM history_entries WHERE quality > :quality"
    results = await database.fetch_all(query=query, values={"quality": 0.7})
    
    await database.disconnect()

asyncio.run(main())
"""
            ),
            
            'tortoise-orm': LibraryInfo(
                name="tortoise-orm",
                version="0.20.0",
                description="ORM async para Python",
                use_case="ORM async, FastAPI, aplicaciones de alto rendimiento",
                pros=[
                    "ORM async nativo",
                    "Sintaxis similar a Django ORM",
                    "Integración con FastAPI",
                    "Migraciones automáticas"
                ],
                cons=[
                    "Menos maduro que SQLAlchemy",
                    "Comunidad más pequeña"
                ],
                installation="pip install tortoise-orm[asyncpg]",
                example="""
from tortoise.models import Model
from tortoise import fields

class HistoryEntry(Model):
    id = fields.IntField(pk=True)
    content = fields.TextField()
    model = fields.CharField(max_length=100)
    quality = fields.FloatField()
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "history_entries"

# Usar en FastAPI
from tortoise.contrib.fastapi import register_tortoise

register_tortoise(
    app,
    db_url="sqlite://ai_history.db",
    modules={"models": ["models"]},
    generate_schemas=True,
)
"""
            ),
            
            # Database Testing
            'factory-boy': LibraryInfo(
                name="factory-boy",
                version="3.3.0",
                description="Librería para crear objetos de prueba",
                use_case="Testing, fixtures, datos de prueba",
                pros=[
                    "Fácil creación de objetos de prueba",
                    "Integración con ORMs",
                    "Datos realistas",
                    "Reutilización de factories"
                ],
                cons=[
                    "Solo para testing",
                    "Curva de aprendizaje"
                ],
                installation="pip install factory-boy",
                example="""
import factory
from models import HistoryEntry

class HistoryEntryFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = HistoryEntry
    
    content = factory.Faker('text', max_nb_chars=200)
    model = factory.Iterator(['gpt-4', 'claude-3', 'gpt-3.5'])
    quality = factory.Faker('pyfloat', min_value=0.0, max_value=1.0)

# Usar en tests
entry = HistoryEntryFactory()
entries = HistoryEntryFactory.create_batch(10)
"""
            ),
            
            # Database Monitoring
            'sqlalchemy-utils': LibraryInfo(
                name="sqlalchemy-utils",
                version="0.41.1",
                description="Utilidades adicionales para SQLAlchemy",
                use_case="Utilidades de base de datos, tipos personalizados, funciones helper",
                pros=[
                    "Tipos de datos adicionales",
                    "Funciones helper útiles",
                    "Integración con SQLAlchemy",
                    "Utilidades para testing"
                ],
                cons=[
                    "Dependencia adicional",
                    "Funcionalidades específicas"
                ],
                installation="pip install sqlalchemy-utils",
                example="""
from sqlalchemy_utils import create_database, drop_database, database_exists
from sqlalchemy import create_engine

# Verificar si existe la base de datos
engine = create_engine('postgresql://user:pass@localhost/db')
if not database_exists(engine.url):
    create_database(engine.url)

# Tipos adicionales
from sqlalchemy_utils import URLType, EmailType, PhoneNumberType

class User(Base):
    id = Column(Integer, primary_key=True)
    email = Column(EmailType)
    website = Column(URLType)
    phone = Column(PhoneNumberType)
"""
            ),
            
            # Database Migration
            'yoyo-migrations': LibraryInfo(
                name="yoyo-migrations",
                version="8.2.0",
                description="Sistema de migraciones simple para Python",
                use_case="Migraciones de base de datos, versionado de esquemas",
                pros=[
                    "Simple de usar",
                    "Soporte para múltiples bases de datos",
                    "Migraciones en Python",
                    "Rollback de cambios"
                ],
                cons=[
                    "Menos funcionalidades que Alembic",
                    "Comunidad más pequeña"
                ],
                installation="pip install yoyo-migrations",
                example="""
# Crear migración
from yoyo import step

steps = [
    step(
        "CREATE TABLE history_entries (id SERIAL PRIMARY KEY, content TEXT)",
        "DROP TABLE history_entries"
    )
]

# Aplicar migraciones
yoyo apply --database postgresql://user:pass@localhost/db ./migrations
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
            'orm': ['sqlalchemy', 'tortoise-orm'],
            'migrations': ['alembic', 'yoyo-migrations'],
            'drivers': ['psycopg2', 'pymongo', 'redis'],
            'async': ['databases', 'tortoise-orm'],
            'testing': ['factory-boy'],
            'utilities': ['sqlalchemy-utils']
        }
        
        if category not in categories:
            return {}
        
        return {name: self.libraries[name] for name in categories[category] if name in self.libraries}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías de base de datos."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)




