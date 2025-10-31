"""
Realistic Core Libraries - Librerías core que realmente existen
============================================================

Librerías fundamentales que realmente existen, están actualizadas
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


class RealisticCoreLibraries:
    """
    Librerías core realistas que realmente existen.
    """
    
    def __init__(self):
        """Inicializar con librerías core reales."""
        self.libraries = {
            'pandas': LibraryInfo(
                name="pandas",
                version="2.1.4",
                description="Manipulación y análisis de datos estructurados",
                use_case="Procesamiento de datos, análisis estadístico, limpieza de datos",
                pros=[
                    "Muy estable y maduro",
                    "Excelente documentación",
                    "Amplia comunidad",
                    "Integración con NumPy",
                    "Operaciones vectorizadas rápidas"
                ],
                cons=[
                    "Puede ser lento para datasets muy grandes",
                    "Uso de memoria alto",
                    "Curva de aprendizaje para operaciones complejas"
                ],
                installation="pip install pandas",
                example="""
import pandas as pd
import numpy as np

# Crear DataFrame simple
df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'content': ['Texto 1', 'Texto 2', 'Texto 3', 'Texto 4', 'Texto 5'],
    'model': ['gpt-4', 'claude-3', 'gpt-4', 'claude-3', 'gpt-3.5'],
    'quality': [0.8, 0.9, 0.7, 0.6, 0.8]
})

# Operaciones básicas
print(df.head())
print(df.groupby('model')['quality'].mean())
print(df[df['quality'] > 0.8])
""",
                real_usage="""
# Uso real en producción
import pandas as pd
from sqlalchemy import create_engine

# Conectar a base de datos
engine = create_engine('postgresql://user:pass@localhost/db')

# Leer datos
df = pd.read_sql('SELECT * FROM history_entries', engine)

# Procesar datos
df['quality_score'] = df['quality'] * 100
df['is_high_quality'] = df['quality'] > 0.8

# Guardar resultados
df.to_sql('processed_entries', engine, if_exists='replace', index=False)
""",
                alternatives=["polars", "dask", "modin"],
                documentation="https://pandas.pydata.org/docs/",
                github="https://github.com/pandas-dev/pandas",
                pypi="https://pypi.org/project/pandas/"
            ),
            
            'numpy': LibraryInfo(
                name="numpy",
                version="1.24.4",
                description="Fundación para computación numérica en Python",
                use_case="Cálculos matemáticos, operaciones vectorizadas, álgebra lineal",
                pros=[
                    "Muy rápido para operaciones numéricas",
                    "Base para muchas librerías científicas",
                    "API estable y consistente",
                    "Optimizado en C"
                ],
                cons=[
                    "Sintaxis puede ser confusa para principiantes",
                    "Limitado a operaciones numéricas"
                ],
                installation="pip install numpy",
                example="""
import numpy as np

# Crear arrays
quality_scores = np.array([0.8, 0.9, 0.7, 0.6, 0.8])
word_counts = np.array([100, 150, 80, 120, 110])

# Operaciones básicas
mean_quality = np.mean(quality_scores)
std_quality = np.std(quality_scores)
normalized = (quality_scores - mean_quality) / std_quality

print(f"Mean: {mean_quality:.3f}")
print(f"Std: {std_quality:.3f}")
print(f"Normalized: {normalized}")
""",
                real_usage="""
# Uso real en análisis de datos
import numpy as np
from sklearn.preprocessing import StandardScaler

# Datos de entrada
data = np.array([[0.8, 100], [0.9, 150], [0.7, 80], [0.6, 120]])

# Normalización
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Operaciones matemáticas
correlation = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
print(f"Correlation: {correlation:.3f}")
""",
                alternatives=["jax", "cupy", "numba"],
                documentation="https://numpy.org/doc/stable/",
                github="https://github.com/numpy/numpy",
                pypi="https://pypi.org/project/numpy/"
            ),
            
            'requests': LibraryInfo(
                name="requests",
                version="2.31.0",
                description="Cliente HTTP para Python",
                use_case="APIs REST, descarga de datos, comunicación HTTP",
                pros=[
                    "Muy simple de usar",
                    "API intuitiva",
                    "Manejo automático de sesiones",
                    "Soporte para autenticación"
                ],
                cons=[
                    "Síncrono (no async)",
                    "No soporte nativo para HTTP/2"
                ],
                installation="pip install requests",
                example="""
import requests

# GET request
response = requests.get('https://api.example.com/entries')
data = response.json()

# POST request
new_entry = {
    'content': 'Sample text',
    'model': 'gpt-4',
    'quality': 0.8
}
response = requests.post('https://api.example.com/entries', json=new_entry)

# Con autenticación
headers = {'Authorization': 'Bearer your-token'}
response = requests.get('https://api.example.com/protected', headers=headers)
""",
                real_usage="""
# Uso real en producción
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configurar sesión con retry
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Usar sesión
try:
    response = session.get('https://api.example.com/data', timeout=30)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
""",
                alternatives=["httpx", "aiohttp", "urllib3"],
                documentation="https://requests.readthedocs.io/",
                github="https://github.com/psf/requests",
                pypi="https://pypi.org/project/requests/"
            ),
            
            'python-dotenv': LibraryInfo(
                name="python-dotenv",
                version="1.0.0",
                description="Carga variables de entorno desde archivos .env",
                use_case="Configuración de aplicación, variables de entorno",
                pros=[
                    "Muy simple de usar",
                    "Estándar de la industria",
                    "No dependencias adicionales"
                ],
                cons=[
                    "Funcionalidad limitada",
                    "No validación de tipos"
                ],
                installation="pip install python-dotenv",
                example="""
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Usar variables
database_url = os.getenv('DATABASE_URL')
api_key = os.getenv('API_KEY')
debug = os.getenv('DEBUG', 'False').lower() == 'true'

print(f"Database URL: {database_url}")
print(f"API Key: {api_key[:10]}..." if api_key else "No API key")
print(f"Debug mode: {debug}")
""",
                real_usage="""
# Uso real en aplicación
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# Cargar configuración
load_dotenv()

# Configuración de base de datos
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Crear engine
engine = create_engine(DATABASE_URL)

# Configuración de API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
""",
                alternatives=["pydantic-settings", "dynaconf", "python-decouple"],
                documentation="https://python-dotenv.readthedocs.io/",
                github="https://github.com/theskumar/python-dotenv",
                pypi="https://pypi.org/project/python-dotenv/"
            ),
            
            'pydantic': LibraryInfo(
                name="pydantic",
                version="2.5.0",
                description="Validación de datos usando type hints de Python",
                use_case="Validación de entrada, serialización, configuración",
                pros=[
                    "Validación automática de tipos",
                    "Integración con FastAPI",
                    "Serialización JSON automática",
                    "Excelente para APIs"
                ],
                cons=[
                    "Puede ser estricto con tipos",
                    "Curva de aprendizaje para validaciones complejas"
                ],
                installation="pip install pydantic",
                example="""
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class HistoryEntry(BaseModel):
    id: Optional[int] = None
    content: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., min_length=1, max_length=100)
    quality: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v.strip()

# Uso
entry = HistoryEntry(
    content="Sample text",
    model="gpt-4",
    quality=0.8
)

print(entry.model_dump())
print(entry.model_dump_json())
""",
                real_usage="""
# Uso real en FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI()

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

@app.post("/entries", response_model=HistoryEntryResponse)
async def create_entry(entry: HistoryEntryCreate):
    # Validación automática por Pydantic
    # entry ya está validado
    return HistoryEntryResponse(
        id=1,
        **entry.model_dump(),
        timestamp=datetime.now()
    )
""",
                alternatives=["marshmallow", "dataclasses", "attrs"],
                documentation="https://docs.pydantic.dev/",
                github="https://github.com/pydantic/pydantic",
                pypi="https://pypi.org/project/pydantic/"
            ),
            
            'loguru': LibraryInfo(
                name="loguru",
                version="0.7.2",
                description="Sistema de logging moderno y simple",
                use_case="Logging de aplicación, debugging, monitoreo",
                pros=[
                    "API muy simple",
                    "Formateo automático",
                    "Rotación de logs automática",
                    "Colores en terminal"
                ],
                cons=[
                    "Menos configuración que logging estándar",
                    "Dependencia adicional"
                ],
                installation="pip install loguru",
                example="""
from loguru import logger
import sys

# Configurar logger
logger.remove()  # Remover handler por defecto
logger.add(sys.stderr, level="INFO")
logger.add("app.log", rotation="1 day", retention="30 days")

# Usar logger
logger.info("Application started")
logger.error("An error occurred")
logger.debug("Debug information")

# Con contexto
logger.bind(user_id="123").info("User action")
""",
                real_usage="""
# Uso real en producción
from loguru import logger
import sys
import os

# Configuración de logging
def setup_logging():
    logger.remove()
    
    # Console logging
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    # Error logging
    logger.add(
        "logs/errors.log",
        rotation="1 week",
        retention="12 weeks",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )

# Usar en aplicación
setup_logging()

logger.info("Application started")
logger.error("Database connection failed")
logger.bind(request_id="123").info("Processing request")
""",
                alternatives=["logging", "structlog", "python-json-logger"],
                documentation="https://loguru.readthedocs.io/",
                github="https://github.com/Delgan/loguru",
                pypi="https://pypi.org/project/loguru/"
            ),
            
            'click': LibraryInfo(
                name="click",
                version="8.1.7",
                description="Framework para crear interfaces de línea de comandos",
                use_case="CLI tools, scripts de administración, comandos personalizados",
                pros=[
                    "API muy simple",
                    "Decoradores intuitivos",
                    "Ayuda automática",
                    "Validación de parámetros"
                ],
                cons=[
                    "Solo para CLI",
                    "Limitado para interfaces complejas"
                ],
                installation="pip install click",
                example="""
import click

@click.command()
@click.option('--name', default='World', help='Name to greet')
@click.option('--count', default=1, help='Number of greetings')
def hello(name, count):
    """Simple program that greets NAME for a total of COUNT times."""
    for _ in range(count):
        click.echo(f'Hello {name}!')

if __name__ == '__main__':
    hello()
""",
                real_usage="""
# Uso real en aplicación
import click
import pandas as pd
from sqlalchemy import create_engine

@click.group()
def cli():
    """AI History Comparison CLI"""
    pass

@cli.command()
@click.option('--database-url', required=True, help='Database URL')
@click.option('--output', default='report.csv', help='Output file')
def generate_report(database_url, output):
    """Generate quality report"""
    engine = create_engine(database_url)
    df = pd.read_sql('SELECT * FROM history_entries', engine)
    
    report = df.groupby('model').agg({
        'quality': ['mean', 'std', 'count']
    }).round(3)
    
    report.to_csv(output)
    click.echo(f"Report saved to {output}")

@cli.command()
@click.option('--entry-id', type=int, required=True, help='Entry ID to analyze')
def analyze_entry(entry_id):
    """Analyze specific entry"""
    click.echo(f"Analyzing entry {entry_id}...")
    # Lógica de análisis aquí

if __name__ == '__main__':
    cli()
""",
                alternatives=["argparse", "typer", "fire"],
                documentation="https://click.palletsprojects.com/",
                github="https://github.com/pallets/click",
                pypi="https://pypi.org/project/click/"
            ),
            
            'python-dateutil': LibraryInfo(
                name="python-dateutil",
                version="2.8.2",
                description="Extensiones para el módulo datetime de Python",
                use_case="Parsing de fechas, cálculos de tiempo, zonas horarias",
                pros=[
                    "Parsing flexible de fechas",
                    "Cálculos de tiempo avanzados",
                    "Soporte para zonas horarias",
                    "Integración con pandas"
                ],
                cons=[
                    "Dependencia adicional",
                    "Puede ser lento para operaciones masivas"
                ],
                installation="pip install python-dateutil",
                example="""
from dateutil import parser, relativedelta
from datetime import datetime

# Parsing flexible de fechas
date1 = parser.parse("2023-10-15")
date2 = parser.parse("Oct 15, 2023")
date3 = parser.parse("15/10/2023")

# Cálculos de tiempo
now = datetime.now()
next_month = now + relativedelta(months=1)
last_week = now - relativedelta(weeks=1)

print(f"Next month: {next_month}")
print(f"Last week: {last_week}")

# Diferencia entre fechas
diff = relativedelta(date2, date1)
print(f"Difference: {diff}")
""",
                real_usage="""
# Uso real en aplicación
from dateutil import parser, relativedelta
from datetime import datetime, timedelta
import pandas as pd

def process_dates(data):
    """Procesar fechas en datos"""
    # Convertir strings a datetime
    data['created_at'] = pd.to_datetime(data['created_at'])
    
    # Agregar columnas de tiempo
    data['year'] = data['created_at'].dt.year
    data['month'] = data['created_at'].dt.month
    data['day_of_week'] = data['created_at'].dt.day_name()
    
    # Filtrar por rango de fechas
    start_date = datetime.now() - relativedelta(months=1)
    recent_data = data[data['created_at'] >= start_date]
    
    return recent_data

# Parsing de fechas de usuario
def parse_user_date(date_string):
    """Parsear fecha ingresada por usuario"""
    try:
        return parser.parse(date_string)
    except ValueError:
        raise ValueError(f"Invalid date format: {date_string}")
""",
                alternatives=["arrow", "pendulum", "maya"],
                documentation="https://dateutil.readthedocs.io/",
                github="https://github.com/dateutil/dateutil",
                pypi="https://pypi.org/project/python-dateutil/"
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
        """Generar requirements.txt con las librerías core realistas."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                requirements.append(f"{package}=={lib.version}")
        
        return '\n'.join(requirements)
    
    def get_realistic_usage_examples(self) -> Dict[str, str]:
        """Obtener ejemplos de uso real para cada librería."""
        return {name: lib.real_usage for name, lib in self.libraries.items()}




