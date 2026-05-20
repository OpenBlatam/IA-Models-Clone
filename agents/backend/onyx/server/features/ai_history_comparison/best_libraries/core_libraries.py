"""
Core Libraries - Mejores librerías fundamentales
===============================================

Las mejores librerías para funcionalidades core del sistema.
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


class CoreLibraries:
    """
    Mejores librerías para funcionalidades core.
    """
    
    def __init__(self):
        """Inicializar con las mejores librerías core."""
        self.libraries = {
            # Data Processing
            'pandas': LibraryInfo(
                name="pandas",
                version="2.1.0",
                description="Manipulación y análisis de datos estructurados",
                use_case="Procesamiento de datos de historial, análisis estadístico",
                pros=[
                    "Excelente para manipulación de datos",
                    "Operaciones vectorizadas rápidas",
                    "Integración con NumPy",
                    "Amplia comunidad"
                ],
                cons=[
                    "Puede ser pesado para datos pequeños",
                    "Curva de aprendizaje para operaciones complejas"
                ],
                installation="pip install pandas",
                example="""
import pandas as pd

# Crear DataFrame con datos de historial
df = pd.DataFrame({
    'id': ['1', '2', '3'],
    'content': ['Texto 1', 'Texto 2', 'Texto 3'],
    'model': ['gpt-4', 'claude-3', 'gpt-4'],
    'quality': [0.8, 0.9, 0.7]
})

# Análisis estadístico
print(df.groupby('model')['quality'].mean())
"""
            ),
            
            'numpy': LibraryInfo(
                name="numpy",
                version="1.24.0",
                description="Computación numérica eficiente",
                use_case="Cálculos matemáticos, operaciones vectorizadas",
                pros=[
                    "Muy rápido para operaciones numéricas",
                    "Interfaz simple y consistente",
                    "Base para muchas otras librerías",
                    "Optimizado en C"
                ],
                cons=[
                    "Sintaxis puede ser confusa para principiantes",
                    "Limitado a operaciones numéricas"
                ],
                installation="pip install numpy",
                example="""
import numpy as np

# Calcular métricas de calidad
quality_scores = np.array([0.8, 0.9, 0.7, 0.6])
mean_quality = np.mean(quality_scores)
std_quality = np.std(quality_scores)

# Operaciones vectorizadas
normalized_scores = (quality_scores - mean_quality) / std_quality
"""
            ),
            
            # Text Processing
            'nltk': LibraryInfo(
                name="nltk",
                version="3.8.1",
                description="Procesamiento de lenguaje natural",
                use_case="Análisis de texto, tokenización, análisis de sentimientos",
                pros=[
                    "Amplia gama de herramientas NLP",
                    "Bien documentado",
                    "Módulos especializados",
                    "Datos de entrenamiento incluidos"
                ],
                cons=[
                    "Puede ser lento para textos grandes",
                    "Requiere descarga de datos adicionales"
                ],
                installation="pip install nltk",
                example="""
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Análisis de sentimientos
sia = SentimentIntensityAnalyzer()
text = "This is a great product!"
sentiment = sia.polarity_scores(text)
print(f"Sentiment: {sentiment}")
"""
            ),
            
            'spacy': LibraryInfo(
                name="spacy",
                version="3.7.0",
                description="Procesamiento de lenguaje natural industrial",
                use_case="Análisis de texto avanzado, entidades nombradas, POS tagging",
                pros=[
                    "Muy rápido y eficiente",
                    "Modelos pre-entrenados de alta calidad",
                    "API moderna y limpia",
                    "Excelente para producción"
                ],
                cons=[
                    "Modelos pueden ser grandes",
                    "Menos flexible que NLTK"
                ],
                installation="pip install spacy && python -m spacy download en_core_web_sm",
                example="""
import spacy

# Cargar modelo de inglés
nlp = spacy.load("en_core_web_sm")

# Procesar texto
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
"""
            ),
            
            # Data Validation
            'pydantic': LibraryInfo(
                name="pydantic",
                version="2.5.0",
                description="Validación de datos con type hints",
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
from pydantic import BaseModel, validator
from typing import Optional

class HistoryEntry(BaseModel):
    id: str
    content: str
    model: str
    quality: float
    
    @validator('quality')
    def quality_must_be_between_0_and_1(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Quality must be between 0.0 and 1.0')
        return v

# Uso
entry = HistoryEntry(
    id="1",
    content="Sample text",
    model="gpt-4",
    quality=0.8
)
"""
            ),
            
            # Configuration
            'python-dotenv': LibraryInfo(
                name="python-dotenv",
                version="1.0.0",
                description="Carga variables de entorno desde archivos .env",
                use_case="Configuración de aplicación, variables de entorno",
                pros=[
                    "Simple de usar",
                    "Integración con .env files",
                    "No dependencias adicionales",
                    "Estándar de la industria"
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
"""
            ),
            
            # Logging
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

# Configurar logger
logger.add("app.log", rotation="1 day", retention="30 days")

# Usar logger
logger.info("Application started")
logger.error("An error occurred")
logger.debug("Debug information")
"""
            ),
            
            # Async Support
            'asyncio': LibraryInfo(
                name="asyncio",
                version="Built-in",
                description="Programación asíncrona",
                use_case="Operaciones I/O asíncronas, APIs concurrentes",
                pros=[
                    "Incluido en Python estándar",
                    "Excelente para I/O",
                    "Integración con FastAPI",
                    "Mejor rendimiento para APIs"
                ],
                cons=[
                    "Curva de aprendizaje",
                    "Debugging más complejo"
                ],
                installation="Built-in",
                example="""
import asyncio
import aiohttp

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()

# Ejecutar tareas asíncronas
async def main():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results
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
            'data_processing': ['pandas', 'numpy'],
            'text_processing': ['nltk', 'spacy'],
            'validation': ['pydantic'],
            'configuration': ['python-dotenv'],
            'logging': ['loguru'],
            'async': ['asyncio']
        }
        
        if category not in categories:
            return {}
        
        return {name: self.libraries[name] for name in categories[category] if name in self.libraries}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)




