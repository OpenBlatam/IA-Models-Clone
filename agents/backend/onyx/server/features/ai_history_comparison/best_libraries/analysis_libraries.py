"""
Analysis Libraries - Mejores librerías de análisis
================================================

Las mejores librerías para análisis de datos y métricas.
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


class AnalysisLibraries:
    """
    Mejores librerías para análisis de datos y métricas.
    """
    
    def __init__(self):
        """Inicializar con las mejores librerías de análisis."""
        self.libraries = {
            # Statistical Analysis
            'scipy': LibraryInfo(
                name="scipy",
                version="1.11.0",
                description="Librería científica para Python",
                use_case="Análisis estadístico, optimización, procesamiento de señales",
                pros=[
                    "Amplia gama de funciones científicas",
                    "Optimizado para rendimiento",
                    "Integración con NumPy",
                    "Algoritmos avanzados"
                ],
                cons=[
                    "Puede ser complejo para casos simples",
                    "Curva de aprendizaje"
                ],
                installation="pip install scipy",
                example="""
from scipy import stats
import numpy as np

# Análisis estadístico
data = np.array([0.8, 0.9, 0.7, 0.6, 0.8])
mean = np.mean(data)
std = np.std(data)

# Test de normalidad
statistic, p_value = stats.normaltest(data)

# Correlación
x = np.array([1, 2, 3, 4, 5])
y = np.array([0.8, 0.9, 0.7, 0.6, 0.8])
correlation, p_value = stats.pearsonr(x, y)
"""
            ),
            
            'statsmodels': LibraryInfo(
                name="statsmodels",
                version="0.14.0",
                description="Modelado estadístico y econométrico",
                use_case="Regresión, análisis de series temporales, pruebas estadísticas",
                pros=[
                    "Modelado estadístico avanzado",
                    "Análisis de series temporales",
                    "Pruebas estadísticas",
                    "Visualizaciones estadísticas"
                ],
                cons=[
                    "Curva de aprendizaje empinada",
                    "Puede ser lento para datasets grandes"
                ],
                installation="pip install statsmodels",
                example="""
import statsmodels.api as sm
import pandas as pd

# Regresión lineal
data = pd.DataFrame({
    'quality': [0.8, 0.9, 0.7, 0.6, 0.8],
    'words': [100, 150, 80, 120, 110]
})

X = data['words']
y = data['quality']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
"""
            ),
            
            # Text Analysis
            'textstat': LibraryInfo(
                name="textstat",
                version="0.7.3",
                description="Métricas de legibilidad y complejidad de texto",
                use_case="Análisis de legibilidad, métricas de texto, evaluación de calidad",
                pros=[
                    "Múltiples métricas de legibilidad",
                    "Fácil de usar",
                    "Soporte para múltiples idiomas",
                    "Métricas estándar de la industria"
                ],
                cons=[
                    "Limitado a métricas de legibilidad",
                    "Puede ser lento para textos muy largos"
                ],
                installation="pip install textstat",
                example="""
import textstat

text = "This is a sample text for readability analysis."

# Métricas de legibilidad
flesch_score = textstat.flesch_reading_ease(text)
flesch_kincaid = textstat.flesch_kincaid_grade(text)
gunning_fog = textstat.gunning_fog(text)
smog_index = textstat.smog_index(text)

print(f"Flesch Reading Ease: {flesch_score}")
print(f"Flesch-Kincaid Grade: {flesch_kincaid}")
print(f"Gunning Fog: {gunning_fog}")
print(f"SMOG Index: {smog_index}")
"""
            ),
            
            'readability': LibraryInfo(
                name="readability",
                version="0.3.1",
                description="Análisis de legibilidad de texto",
                use_case="Métricas de legibilidad, evaluación de texto, análisis de complejidad",
                pros=[
                    "Métricas de legibilidad específicas",
                    "Fácil de usar",
                    "Bueno para análisis de texto",
                    "Métricas estándar"
                ],
                cons=[
                    "Funcionalidad limitada",
                    "Solo para inglés"
                ],
                installation="pip install readability",
                example="""
from readability import Readability

text = "This is a sample text for readability analysis."
r = Readability(text)

# Métricas de legibilidad
flesch = r.flesch()
flesch_kincaid = r.flesch_kincaid()
gunning_fog = r.gunning_fog()
smog = r.smog()

print(f"Flesch: {flesch.score}")
print(f"Flesch-Kincaid: {flesch_kincaid.score}")
print(f"Gunning Fog: {gunning_fog.score}")
print(f"SMOG: {smog.score}")
"""
            ),
            
            # Similarity Analysis
            'fuzzywuzzy': LibraryInfo(
                name="fuzzywuzzy",
                version="0.18.0",
                description="Matching difuso de cadenas",
                use_case="Similitud de texto, matching difuso, búsqueda aproximada",
                pros=[
                    "Múltiples algoritmos de similitud",
                    "Fácil de usar",
                    "Bueno para matching difuso",
                    "Optimizado para rendimiento"
                ],
                cons=[
                    "Limitado a similitud de cadenas",
                    "Puede ser lento para textos muy largos"
                ],
                installation="pip install fuzzywuzzy python-levenshtein",
                example="""
from fuzzywuzzy import fuzz, process

# Similitud de cadenas
text1 = "This is a sample text"
text2 = "This is a sample text with more words"

# Diferentes tipos de similitud
ratio = fuzz.ratio(text1, text2)
partial_ratio = fuzz.partial_ratio(text1, text2)
token_sort_ratio = fuzz.token_sort_ratio(text1, text2)
token_set_ratio = fuzz.token_set_ratio(text1, text2)

print(f"Ratio: {ratio}")
print(f"Partial Ratio: {partial_ratio}")
print(f"Token Sort Ratio: {token_sort_ratio}")
print(f"Token Set Ratio: {token_set_ratio}")

# Búsqueda difusa
choices = ["apple", "banana", "orange", "grape"]
result = process.extractOne("aple", choices)
print(f"Best match: {result}")
"""
            ),
            
            'jellyfish': LibraryInfo(
                name="jellyfish",
                version="0.9.0",
                description="Algoritmos de similitud de cadenas",
                use_case="Distancia de Levenshtein, similitud fonética, matching de nombres",
                pros=[
                    "Múltiples algoritmos de distancia",
                    "Similitud fonética",
                    "Optimizado para rendimiento",
                    "Fácil de usar"
                ],
                cons=[
                    "Limitado a similitud de cadenas",
                    "Algunos algoritmos son específicos para inglés"
                ],
                installation="pip install jellyfish",
                example="""
import jellyfish

# Distancia de Levenshtein
text1 = "kitten"
text2 = "sitting"
levenshtein_distance = jellyfish.levenshtein_distance(text1, text2)
print(f"Levenshtein distance: {levenshtein_distance}")

# Similitud fonética
name1 = "Smith"
name2 = "Smyth"
soundex1 = jellyfish.soundex(name1)
soundex2 = jellyfish.soundex(name2)
print(f"Soundex: {soundex1} vs {soundex2}")

# Similitud Jaro-Winkler
jaro_winkler = jellyfish.jaro_winkler_similarity(text1, text2)
print(f"Jaro-Winkler similarity: {jaro_winkler}")
"""
            ),
            
            # Time Series Analysis
            'prophet': LibraryInfo(
                name="prophet",
                version="1.1.4",
                description="Pronóstico de series temporales",
                use_case="Pronósticos, análisis de tendencias, series temporales",
                pros=[
                    "Fácil de usar",
                    "Maneja estacionalidad automáticamente",
                    "Robusto a valores faltantes",
                    "Visualizaciones automáticas"
                ],
                cons=[
                    "Solo para series temporales",
                    "Puede ser lento para datos grandes"
                ],
                installation="pip install prophet",
                example="""
from prophet import Prophet
import pandas as pd

# Crear datos de serie temporal
df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=365),
    'y': [0.8 + 0.1 * (i % 7) + 0.05 * (i % 30) for i in range(365)]
})

# Crear y entrenar modelo
model = Prophet()
model.fit(df)

# Hacer pronóstico
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Visualizar
model.plot(forecast)
"""
            ),
            
            # Clustering
            'hdbscan': LibraryInfo(
                name="hdbscan",
                version="0.8.33",
                description="Clustering basado en densidad",
                use_case="Clustering, agrupación de datos, análisis de patrones",
                pros=[
                    "Clustering de alta calidad",
                    "No requiere especificar número de clusters",
                    "Maneja outliers automáticamente",
                    "Bueno para datos de alta dimensión"
                ],
                cons=[
                    "Puede ser lento para datasets grandes",
                    "Parámetros sensibles"
                ],
                installation="pip install hdbscan",
                example="""
import hdbscan
import numpy as np

# Crear datos de ejemplo
data = np.random.random((100, 2))

# Aplicar clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
cluster_labels = clusterer.fit_predict(data)

# Obtener información de clusters
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of clusters: {n_clusters}")
print(f"Cluster labels: {cluster_labels}")
"""
            ),
            
            # Dimensionality Reduction
            'umap': LibraryInfo(
                name="umap",
                version="0.5.4",
                description="Reducción de dimensionalidad",
                use_case="Visualización de datos, reducción de dimensionalidad, análisis exploratorio",
                pros=[
                    "Preserva estructura local y global",
                    "Bueno para visualización",
                    "Maneja datos de alta dimensión",
                    "Integración con scikit-learn"
                ],
                cons=[
                    "Puede ser lento para datasets grandes",
                    "Parámetros sensibles"
                ],
                installation="pip install umap-learn",
                example="""
import umap
import numpy as np
import matplotlib.pyplot as plt

# Crear datos de alta dimensión
data = np.random.random((1000, 50))

# Reducir dimensionalidad
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(data)

# Visualizar
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title('UMAP Embedding')
plt.show()
"""
            ),
            
            # Performance Metrics
            'sklearn-metrics': LibraryInfo(
                name="sklearn-metrics",
                version="1.3.0",
                description="Métricas de rendimiento de scikit-learn",
                use_case="Métricas de evaluación, análisis de rendimiento, comparación de modelos",
                pros=[
                    "Amplia gama de métricas",
                    "Integración con scikit-learn",
                    "Fácil de usar",
                    "Bien documentado"
                ],
                cons=[
                    "Parte de scikit-learn",
                    "Limitado a métricas estándar"
                ],
                installation="pip install scikit-learn",
                example="""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Métricas de clasificación
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Métricas de regresión
y_true_reg = np.array([0.8, 0.9, 0.7, 0.6])
y_pred_reg = np.array([0.8, 0.8, 0.7, 0.7])

mse = mean_squared_error(y_true_reg, y_pred_reg)
mae = mean_absolute_error(y_true_reg, y_pred_reg)
r2 = r2_score(y_true_reg, y_pred_reg)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
"""
            ),
            
            # Data Quality
            'great-expectations': LibraryInfo(
                name="great-expectations",
                version="0.17.0",
                description="Validación de calidad de datos",
                use_case="Validación de datos, calidad de datos, testing de datos",
                pros=[
                    "Validación de calidad de datos",
                    "Testing de datos",
                    "Documentación automática",
                    "Integración con pipelines"
                ],
                cons=[
                    "Curva de aprendizaje",
                    "Puede ser complejo para casos simples"
                ],
                installation="pip install great-expectations",
                example="""
import great_expectations as ge
import pandas as pd

# Crear dataset
df = pd.DataFrame({
    'quality': [0.8, 0.9, 0.7, 0.6],
    'model': ['gpt-4', 'claude-3', 'gpt-4', 'claude-3']
})

# Crear expectativas
ge_df = ge.from_pandas(df)

# Validar expectativas
ge_df.expect_column_values_to_be_between('quality', 0.0, 1.0)
ge_df.expect_column_values_to_be_in_set('model', ['gpt-4', 'claude-3'])

# Ejecutar validación
results = ge_df.validate()
print(results)
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
            'statistical': ['scipy', 'statsmodels'],
            'text_analysis': ['textstat', 'readability'],
            'similarity': ['fuzzywuzzy', 'jellyfish'],
            'time_series': ['prophet'],
            'clustering': ['hdbscan'],
            'dimensionality': ['umap'],
            'metrics': ['sklearn-metrics'],
            'quality': ['great-expectations']
        }
        
        if category not in categories:
            return {}
        
        return {name: self.libraries[name] for name in categories[category] if name in self.libraries}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values()]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías de análisis."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)




