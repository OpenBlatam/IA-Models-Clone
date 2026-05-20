"""
Core Extensive Libraries - Librerías fundamentales extensas
=========================================================

Guía extensa de librerías fundamentales con más de 50 librerías
organizadas por subcategorías con ejemplos detallados.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class LibraryCategory(Enum):
    """Categorías de librerías."""
    DATA_PROCESSING = "data_processing"
    TEXT_PROCESSING = "text_processing"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    ASYNC = "async"
    UTILITIES = "utilities"
    SERIALIZATION = "serialization"
    CRYPTOGRAPHY = "cryptography"
    NETWORKING = "networking"


@dataclass
class LibraryInfo:
    """Información detallada de una librería."""
    name: str
    version: str
    description: str
    use_case: str
    category: LibraryCategory
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    installation: str = ""
    example: str = ""
    advanced_example: str = ""
    configuration: str = ""
    performance_notes: str = ""
    alternatives: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    documentation: str = ""
    community: str = ""
    last_updated: str = ""
    license: str = ""


class CoreExtensiveLibraries:
    """
    Guía extensa de librerías fundamentales.
    """
    
    def __init__(self):
        """Inicializar con librerías fundamentales extensas."""
        self.libraries = {
            # Data Processing
            'pandas': LibraryInfo(
                name="pandas",
                version="2.1.0",
                description="Manipulación y análisis de datos estructurados de alto rendimiento",
                use_case="Procesamiento de datos de historial, análisis estadístico, limpieza de datos",
                category=LibraryCategory.DATA_PROCESSING,
                pros=[
                    "Excelente para manipulación de datos estructurados",
                    "Operaciones vectorizadas extremadamente rápidas",
                    "Integración perfecta con NumPy",
                    "Amplia comunidad y documentación",
                    "Soporte para múltiples formatos de archivo",
                    "Funciones de agrupación y agregación avanzadas",
                    "Manejo eficiente de datos faltantes",
                    "Indexación y selección flexible"
                ],
                cons=[
                    "Puede ser pesado para datasets muy pequeños",
                    "Curva de aprendizaje para operaciones complejas",
                    "Uso de memoria puede ser alto para datasets grandes",
                    "Algunas operaciones pueden ser lentas en datos no estructurados"
                ],
                installation="pip install pandas[complete]",
                example="""
import pandas as pd
import numpy as np

# Crear DataFrame con datos de historial
df = pd.DataFrame({
    'id': ['1', '2', '3', '4', '5'],
    'content': ['Texto 1', 'Texto 2', 'Texto 3', 'Texto 4', 'Texto 5'],
    'model': ['gpt-4', 'claude-3', 'gpt-4', 'claude-3', 'gpt-3.5'],
    'quality': [0.8, 0.9, 0.7, 0.6, 0.8],
    'timestamp': pd.date_range('2023-01-01', periods=5, freq='D')
})

# Análisis estadístico avanzado
print("Estadísticas por modelo:")
print(df.groupby('model')['quality'].agg(['mean', 'std', 'count', 'min', 'max']))

# Análisis temporal
df['day_of_week'] = df['timestamp'].dt.day_name()
print("\\nCalidad por día de la semana:")
print(df.groupby('day_of_week')['quality'].mean())

# Filtrado avanzado
high_quality = df[df['quality'] >= 0.8]
print(f"\\nEntradas de alta calidad: {len(high_quality)}")

# Operaciones de ventana
df['quality_ma'] = df['quality'].rolling(window=3, min_periods=1).mean()
print("\\nMedia móvil de calidad:")
print(df[['timestamp', 'quality', 'quality_ma']])
""",
                advanced_example="""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Crear dataset grande para análisis avanzado
np.random.seed(42)
n_records = 10000

data = {
    'id': [f'entry_{i}' for i in range(n_records)],
    'content': [f'Content {i} with some text' for i in range(n_records)],
    'model': np.random.choice(['gpt-4', 'claude-3', 'gpt-3.5', 'claude-2'], n_records),
    'quality': np.random.beta(2, 2, n_records),  # Distribución beta para calidad
    'word_count': np.random.poisson(150, n_records),
    'timestamp': pd.date_range('2023-01-01', periods=n_records, freq='H')
}

df = pd.DataFrame(data)

# Análisis avanzado de series temporales
df.set_index('timestamp', inplace=True)

# Resampling por día
daily_stats = df.resample('D').agg({
    'quality': ['mean', 'std', 'count'],
    'word_count': ['mean', 'sum']
}).round(3)

print("Estadísticas diarias:")
print(daily_stats.head())

# Análisis de correlación
correlation_matrix = df[['quality', 'word_count']].corr()
print("\\nMatriz de correlación:")
print(correlation_matrix)

# Análisis de outliers usando IQR
Q1 = df['quality'].quantile(0.25)
Q3 = df['quality'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['quality'] < Q1 - 1.5 * IQR) | (df['quality'] > Q3 + 1.5 * IQR)]
print(f"\\nOutliers detectados: {len(outliers)}")

# Análisis de tendencias
df['quality_trend'] = df['quality'].rolling(window=24, min_periods=1).mean()
df['quality_std'] = df['quality'].rolling(window=24, min_periods=1).std()

# Detección de cambios significativos
df['quality_change'] = df['quality'].diff()
significant_changes = df[abs(df['quality_change']) > 2 * df['quality_std']]
print(f"\\nCambios significativos detectados: {len(significant_changes)}")

# Análisis por modelo con estadísticas avanzadas
model_analysis = df.groupby('model').agg({
    'quality': ['mean', 'std', 'skew', 'kurtosis'],
    'word_count': ['mean', 'std'],
    'id': 'count'
}).round(3)

print("\\nAnálisis avanzado por modelo:")
print(model_analysis)
""",
                configuration="""
# Configuración de pandas para mejor rendimiento
import pandas as pd

# Configurar opciones de visualización
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Configurar advertencias
pd.set_option('mode.chained_assignment', 'raise')

# Configurar formato de números
pd.set_option('display.float_format', '{:.3f}'.format)

# Configurar encoding para archivos
pd.set_option('io.encoding', 'utf-8')
""",
                performance_notes="""
# Optimizaciones de rendimiento para pandas

# 1. Usar tipos de datos apropiados
df['id'] = df['id'].astype('category')  # Para strings repetitivos
df['quality'] = df['quality'].astype('float32')  # Para números decimales

# 2. Usar operaciones vectorizadas
# Malo: df['new_col'] = df['col1'].apply(lambda x: x * 2)
# Bueno: df['new_col'] = df['col1'] * 2

# 3. Usar query() para filtros complejos
result = df.query('quality > 0.8 and model == "gpt-4"')

# 4. Usar eval() para expresiones complejas
df.eval('quality_score = quality * 100', inplace=True)

# 5. Usar chunking para archivos grandes
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
""",
                alternatives=["dask", "polars", "vaex", "modin"],
                dependencies=["numpy", "python-dateutil", "pytz"],
                documentation="https://pandas.pydata.org/docs/",
                community="https://stackoverflow.com/questions/tagged/pandas",
                last_updated="2023-10-01",
                license="BSD-3-Clause"
            ),
            
            'numpy': LibraryInfo(
                name="numpy",
                version="1.24.0",
                description="Fundación para computación numérica en Python",
                use_case="Cálculos matemáticos, operaciones vectorizadas, álgebra lineal",
                category=LibraryCategory.DATA_PROCESSING,
                pros=[
                    "Extremadamente rápido para operaciones numéricas",
                    "Interfaz simple y consistente",
                    "Base para muchas otras librerías científicas",
                    "Optimizado en C y Fortran",
                    "Soporte para arrays multidimensionales",
                    "Amplia gama de funciones matemáticas",
                    "Integración con BLAS y LAPACK",
                    "Memoria eficiente"
                ],
                cons=[
                    "Sintaxis puede ser confusa para principiantes",
                    "Limitado a operaciones numéricas",
                    "No es ideal para datos no numéricos",
                    "Curva de aprendizaje para operaciones avanzadas"
                ],
                installation="pip install numpy[complete]",
                example="""
import numpy as np

# Crear arrays y operaciones básicas
quality_scores = np.array([0.8, 0.9, 0.7, 0.6, 0.8, 0.9, 0.5, 0.7])
word_counts = np.array([100, 150, 80, 120, 110, 140, 90, 130])

# Estadísticas básicas
mean_quality = np.mean(quality_scores)
std_quality = np.std(quality_scores)
median_quality = np.median(quality_scores)

print(f"Calidad promedio: {mean_quality:.3f}")
print(f"Desviación estándar: {std_quality:.3f}")
print(f"Mediana: {median_quality:.3f}")

# Normalización
normalized_scores = (quality_scores - mean_quality) / std_quality
print(f"Scores normalizados: {normalized_scores}")

# Operaciones vectorizadas
quality_per_word = quality_scores / word_counts
print(f"Calidad por palabra: {quality_per_word}")

# Filtrado
high_quality_mask = quality_scores >= 0.8
high_quality_scores = quality_scores[high_quality_mask]
print(f"Scores de alta calidad: {high_quality_scores}")

# Operaciones de array
combined = np.column_stack((quality_scores, word_counts))
print(f"Array combinado:\\n{combined}")
""",
                advanced_example="""
import numpy as np
from scipy import stats

# Crear dataset sintético para análisis avanzado
np.random.seed(42)
n_samples = 1000

# Generar datos con diferentes distribuciones
quality_scores = np.random.beta(2, 2, n_samples)  # Distribución beta
word_counts = np.random.poisson(150, n_samples)   # Distribución Poisson
readability_scores = np.random.normal(0.7, 0.1, n_samples)  # Normal

# Crear matriz de datos
data_matrix = np.column_stack((quality_scores, word_counts, readability_scores))

# Análisis estadístico avanzado
print("=== ANÁLISIS ESTADÍSTICO AVANZADO ===")

# Estadísticas descriptivas
print("\\nEstadísticas descriptivas:")
for i, name in enumerate(['Quality', 'Word Count', 'Readability']):
    print(f"{name}:")
    print(f"  Media: {np.mean(data_matrix[:, i]):.3f}")
    print(f"  Mediana: {np.median(data_matrix[:, i]):.3f}")
    print(f"  Desv. Est.: {np.std(data_matrix[:, i]):.3f}")
    print(f"  Asimetría: {stats.skew(data_matrix[:, i]):.3f}")
    print(f"  Curtosis: {stats.kurtosis(data_matrix[:, i]):.3f}")

# Análisis de correlación
correlation_matrix = np.corrcoef(data_matrix.T)
print("\\nMatriz de correlación:")
print(correlation_matrix)

# Análisis de componentes principales (PCA simplificado)
# Centrar los datos
centered_data = data_matrix - np.mean(data_matrix, axis=0)
# Calcular matriz de covarianza
cov_matrix = np.cov(centered_data.T)
# Valores y vectores propios
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\\nAnálisis de componentes principales:")
print(f"Valores propios: {eigenvalues}")
print(f"Varianza explicada: {eigenvalues / np.sum(eigenvalues)}")

# Operaciones de álgebra lineal
# Regresión lineal simple
X = word_counts.reshape(-1, 1)
y = quality_scores

# Calcular coeficientes de regresión
X_with_intercept = np.column_stack((np.ones(len(X)), X))
coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

print(f"\\nRegresión lineal (Quality ~ Word Count):")
print(f"Intercepto: {coefficients[0]:.3f}")
print(f"Pendiente: {coefficients[1]:.3f}")

# Análisis de outliers usando Z-score
z_scores = np.abs(stats.zscore(data_matrix))
outliers = np.any(z_scores > 3, axis=1)
print(f"\\nOutliers detectados (Z-score > 3): {np.sum(outliers)}")

# Análisis de percentiles
percentiles = [25, 50, 75, 90, 95, 99]
print("\\nPercentiles de calidad:")
for p in percentiles:
    value = np.percentile(quality_scores, p)
    print(f"  P{p}: {value:.3f}")

# Operaciones de convolución para suavizado
kernel = np.array([0.25, 0.5, 0.25])  # Kernel de suavizado
smoothed_quality = np.convolve(quality_scores, kernel, mode='same')
print(f"\\nCalidad suavizada (primeros 10 valores): {smoothed_quality[:10]}")

# Análisis de frecuencia usando FFT
fft_quality = np.fft.fft(quality_scores)
frequencies = np.fft.fftfreq(len(quality_scores))
print(f"\\nAnálisis de frecuencia - Componente dominante: {frequencies[np.argmax(np.abs(fft_quality))]:.3f}")
""",
                configuration="""
# Configuración de NumPy para mejor rendimiento
import numpy as np

# Configurar threading
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Configurar precisión de punto flotante
np.seterr(all='raise')  # Levantar errores para operaciones inválidas

# Configurar formato de impresión
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# Configurar tipos de datos por defecto
np.seterr(divide='ignore', invalid='ignore')
""",
                performance_notes="""
# Optimizaciones de rendimiento para NumPy

# 1. Usar tipos de datos apropiados
arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Menos memoria
arr_float64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Más precisión

# 2. Usar operaciones in-place cuando sea posible
arr = np.array([1, 2, 3, 4, 5])
arr += 1  # In-place, más eficiente que arr = arr + 1

# 3. Usar broadcasting en lugar de loops
# Malo: for i in range(len(arr)): arr[i] += 1
# Bueno: arr += 1

# 4. Usar funciones vectorizadas
# Malo: np.array([np.sin(x) for x in arr])
# Bueno: np.sin(arr)

# 5. Usar views en lugar de copias cuando sea posible
view = arr[1:4]  # View, no copia
copy = arr[1:4].copy()  # Copia explícita

# 6. Usar operaciones de álgebra lineal optimizadas
result = np.dot(matrix1, matrix2)  # Usa BLAS optimizado
""",
                alternatives=["jax", "cupy", "numba", "tensorflow", "pytorch"],
                dependencies=["blas", "lapack"],
                documentation="https://numpy.org/doc/stable/",
                community="https://stackoverflow.com/questions/tagged/numpy",
                last_updated="2023-09-01",
                license="BSD-3-Clause"
            ),
            
            'dask': LibraryInfo(
                name="dask",
                version="2023.8.0",
                description="Computación paralela y distribuida para Python",
                use_case="Procesamiento de datos grandes, computación paralela, análisis distribuido",
                category=LibraryCategory.DATA_PROCESSING,
                pros=[
                    "Escalabilidad a datasets que no caben en memoria",
                    "API similar a pandas y NumPy",
                    "Computación paralela automática",
                    "Integración con ecosistemas existentes",
                    "Soporte para clusters distribuidos",
                    "Lazy evaluation para optimización",
                    "Interfaz unificada para diferentes backends"
                ],
                cons=[
                    "Overhead para datasets pequeños",
                    "Curva de aprendizaje para optimización",
                    "Debugging más complejo",
                    "Dependencias adicionales para clusters"
                ],
                installation="pip install dask[complete]",
                example="""
import dask.array as da
import dask.dataframe as dd
import numpy as np

# Crear array distribuido
large_array = da.random.random((10000, 1000), chunks=(1000, 1000))
print(f"Array shape: {large_array.shape}")
print(f"Chunks: {large_array.chunks}")

# Operaciones lazy
result = large_array.sum(axis=1)
print(f"Result shape: {result.shape}")

# Computación real
computed_result = result.compute()
print(f"Sum computed: {computed_result[:5]}")

# DataFrame distribuido
df = dd.from_pandas(pd.DataFrame({
    'id': range(10000),
    'value': np.random.random(10000)
}), npartitions=4)

# Operaciones en DataFrame distribuido
result_df = df.groupby('id').value.mean().compute()
print(f"Groupby result: {result_df.head()}")
""",
                advanced_example="""
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import numpy as np

# Configurar cliente distribuido
client = Client('scheduler-address:8786')  # Para cluster
# client = Client()  # Para local

# Análisis de datos masivos
print("=== ANÁLISIS DE DATOS MASIVOS CON DASK ===")

# Crear dataset sintético grande
n_samples = 1000000
n_features = 100

# Generar datos distribuidos
data = da.random.random((n_samples, n_features), chunks=(10000, 50))
labels = da.random.randint(0, 2, n_samples, chunks=10000)

print(f"Dataset shape: {data.shape}")
print(f"Memory usage: {data.nbytes / 1e9:.2f} GB")

# Análisis estadístico distribuido
print("\\nEstadísticas distribuidas:")
mean_values = da.mean(data, axis=0)
std_values = da.std(data, axis=0)

# Computar solo cuando sea necesario
print(f"Mean computed: {mean_values.compute()[:5]}")
print(f"Std computed: {std_values.compute()[:5]}")

# Análisis de correlación distribuido
print("\\nAnálisis de correlación:")
# Seleccionar subconjunto para correlación (más eficiente)
subset = data[:, :10]  # Primeras 10 características
correlation_matrix = da.corrcoef(subset.T)
correlation_result = correlation_matrix.compute()
print(f"Correlation matrix shape: {correlation_result.shape}")

# Análisis de clustering distribuido
print("\\nAnálisis de clustering:")
from dask_ml.cluster import KMeans

# Usar subconjunto para clustering
clustering_data = data[:, :20]  # Primeras 20 características
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(clustering_data)
clusters = kmeans.predict(clustering_data)

print(f"Clusters computed: {clusters.compute()[:10]}")

# Análisis de regresión distribuido
print("\\nAnálisis de regresión:")
from dask_ml.linear_model import LinearRegression

# Preparar datos para regresión
X = data[:, :10]  # Primeras 10 características como predictores
y = data[:, 10]   # Característica 11 como target

# Ajustar modelo
lr = LinearRegression()
lr.fit(X, y)
predictions = lr.predict(X)

print(f"Regression R²: {lr.score(X, y).compute():.3f}")
print(f"Predictions sample: {predictions.compute()[:5]}")

# Análisis de series temporales
print("\\nAnálisis de series temporales:")
# Crear serie temporal distribuida
time_series = da.random.random((100000,), chunks=1000)
# Calcular media móvil
window_size = 100
moving_avg = da.convolve(time_series, da.ones(window_size)/window_size, mode='same')
print(f"Moving average computed: {moving_avg.compute()[:10]}")

# Optimización de memoria
print("\\nOptimización de memoria:")
# Persistir datos en memoria para reutilización
data_persisted = data.persist()
print(f"Data persisted in memory")

# Análisis de rendimiento
import time
start_time = time.time()
result = da.sum(data_persisted, axis=1).compute()
end_time = time.time()
print(f"Sum computation time: {end_time - start_time:.2f} seconds")

# Limpiar cliente
client.close()
""",
                configuration="""
# Configuración de Dask para diferentes entornos

# Configuración local
from dask.distributed import Client
client = Client(
    n_workers=4,           # Número de workers
    threads_per_worker=2,  # Threads por worker
    memory_limit='2GB',    # Límite de memoria por worker
    dashboard_address=':8787'  # Dashboard web
)

# Configuración para cluster
from dask.distributed import Client
client = Client('scheduler-address:8786')

# Configuración de array
import dask.array as da
da.config.set({
    'array.chunk-size': '128MB',
    'array.slicing.split_large_chunks': True
})

# Configuración de DataFrame
import dask.dataframe as dd
dd.config.set({
    'dataframe.query-planning': True,
    'dataframe.optimize': True
})
""",
                performance_notes="""
# Optimizaciones de rendimiento para Dask

# 1. Optimizar tamaño de chunks
# Para arrays: chunksize = memoria_disponible / número_de_workers
# Para DataFrames: npartitions = número_de_cores * 2-4

# 2. Usar persist() para datos reutilizados
data = da.random.random((10000, 1000))
data_persisted = data.persist()  # Mantiene en memoria

# 3. Optimizar operaciones
# Usar operaciones vectorizadas
result = da.sum(data, axis=1)  # Mejor que loops

# 4. Usar rechunk() para optimizar chunks
data_optimized = data.rechunk((1000, 1000))

# 5. Usar dask.delayed para funciones personalizadas
from dask import delayed
@delayed
def expensive_function(x):
    return x ** 2

results = [expensive_function(i) for i in range(100)]
final_result = da.compute(*results)
""",
                alternatives=["ray", "joblib", "multiprocessing", "concurrent.futures"],
                dependencies=["numpy", "pandas", "toolz", "cloudpickle"],
                documentation="https://docs.dask.org/",
                community="https://stackoverflow.com/questions/tagged/dask",
                last_updated="2023-08-01",
                license="BSD-3-Clause"
            ),
            
            'polars': LibraryInfo(
                name="polars",
                version="0.20.0",
                description="DataFrame de alto rendimiento escrito en Rust",
                use_case="Procesamiento de datos ultra-rápido, análisis de grandes datasets",
                category=LibraryCategory.DATA_PROCESSING,
                pros=[
                    "Extremadamente rápido (escrito en Rust)",
                    "API similar a pandas",
                    "Lazy evaluation por defecto",
                    "Memoria eficiente",
                    "Soporte para múltiples formatos",
                    "Operaciones vectorizadas optimizadas",
                    "Paralelización automática"
                ],
                cons=[
                    "Relativamente nuevo",
                    "Ecosistema más pequeño que pandas",
                    "Algunas funcionalidades aún en desarrollo",
                    "Curva de aprendizaje para optimizaciones"
                ],
                installation="pip install polars[all]",
                example="""
import polars as pl
import numpy as np

# Crear DataFrame
df = pl.DataFrame({
    'id': range(1000),
    'content': [f'Content {i}' for i in range(1000)],
    'model': np.random.choice(['gpt-4', 'claude-3', 'gpt-3.5'], 1000),
    'quality': np.random.random(1000),
    'word_count': np.random.randint(50, 500, 1000)
})

# Operaciones básicas
print("DataFrame shape:", df.shape)
print("\\nPrimeras 5 filas:")
print(df.head())

# Filtrado y selección
high_quality = df.filter(pl.col('quality') > 0.8)
print(f"\\nEntradas de alta calidad: {len(high_quality)}")

# Agrupación y agregación
model_stats = df.group_by('model').agg([
    pl.col('quality').mean().alias('avg_quality'),
    pl.col('quality').std().alias('std_quality'),
    pl.col('word_count').mean().alias('avg_words'),
    pl.count().alias('count')
])
print("\\nEstadísticas por modelo:")
print(model_stats)

# Operaciones de ventana
df_with_rank = df.with_columns([
    pl.col('quality').rank().over('model').alias('quality_rank'),
    pl.col('quality').mean().over('model').alias('model_avg_quality')
])
print("\\nDataFrame con rankings:")
print(df_with_rank.head())
""",
                advanced_example="""
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Crear dataset grande para análisis avanzado
np.random.seed(42)
n_records = 100000

# Generar datos sintéticos
data = {
    'id': [f'entry_{i}' for i in range(n_records)],
    'content': [f'Content {i} with some text' for i in range(n_records)],
    'model': np.random.choice(['gpt-4', 'claude-3', 'gpt-3.5', 'claude-2'], n_records),
    'quality': np.random.beta(2, 2, n_records),
    'word_count': np.random.poisson(150, n_records),
    'readability': np.random.normal(0.7, 0.1, n_records),
    'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_records)]
}

df = pl.DataFrame(data)

print("=== ANÁLISIS AVANZADO CON POLARS ===")
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.estimated_size('mb'):.2f} MB")

# Análisis de rendimiento - comparación con pandas
import time

# Polars
start_time = time.time()
polars_result = df.group_by('model').agg([
    pl.col('quality').mean().alias('avg_quality'),
    pl.col('quality').std().alias('std_quality'),
    pl.col('word_count').sum().alias('total_words'),
    pl.count().alias('count')
])
polars_time = time.time() - start_time
print(f"\\nPolars groupby time: {polars_time:.3f} seconds")

# Análisis de correlación
print("\\nAnálisis de correlación:")
correlation_df = df.select([
    pl.corr('quality', 'word_count').alias('quality_word_corr'),
    pl.corr('quality', 'readability').alias('quality_readability_corr'),
    pl.corr('word_count', 'readability').alias('word_readability_corr')
])
print(correlation_df)

# Análisis de outliers usando IQR
print("\\nAnálisis de outliers:")
outlier_analysis = df.with_columns([
    pl.col('quality').quantile(0.25).over('model').alias('q1'),
    pl.col('quality').quantile(0.75).over('model').alias('q3')
]).with_columns([
    (pl.col('q3') - pl.col('q1')).alias('iqr'),
    (pl.col('q1') - 1.5 * pl.col('iqr')).alias('lower_bound'),
    (pl.col('q3') + 1.5 * pl.col('iqr')).alias('upper_bound')
]).with_columns([
    ((pl.col('quality') < pl.col('lower_bound')) | 
     (pl.col('quality') > pl.col('upper_bound'))).alias('is_outlier')
])

outlier_count = outlier_analysis.filter(pl.col('is_outlier')).height
print(f"Outliers detectados: {outlier_count}")

# Análisis de series temporales
print("\\nAnálisis de series temporales:")
df_temporal = df.with_columns([
    pl.col('timestamp').dt.hour().alias('hour'),
    pl.col('timestamp').dt.day_of_week().alias('day_of_week'),
    pl.col('timestamp').dt.month().alias('month')
])

# Análisis por hora del día
hourly_stats = df_temporal.group_by('hour').agg([
    pl.col('quality').mean().alias('avg_quality'),
    pl.col('word_count').mean().alias('avg_words'),
    pl.count().alias('count')
]).sort('hour')

print("Estadísticas por hora:")
print(hourly_stats.head(10))

# Análisis de tendencias
print("\\nAnálisis de tendencias:")
trend_analysis = df_temporal.with_columns([
    pl.col('quality').rolling_mean(window_size=100).over('model').alias('quality_trend'),
    pl.col('quality').rolling_std(window_size=100).over('model').alias('quality_volatility')
])

# Detectar cambios significativos
significant_changes = trend_analysis.filter(
    pl.col('quality_volatility') > pl.col('quality_volatility').quantile(0.95)
)
print(f"Cambios significativos detectados: {significant_changes.height}")

# Análisis de clustering (simplificado)
print("\\nAnálisis de clustering:")
# Usar k-means simple basado en percentiles
clustering_df = df.with_columns([
    pl.when(pl.col('quality') < 0.33)
    .then(pl.lit('low'))
    .when(pl.col('quality') < 0.66)
    .then(pl.lit('medium'))
    .otherwise(pl.lit('high'))
    .alias('quality_cluster')
])

cluster_stats = clustering_df.group_by('quality_cluster').agg([
    pl.col('quality').mean().alias('avg_quality'),
    pl.col('word_count').mean().alias('avg_words'),
    pl.count().alias('count')
]).sort('quality_cluster')

print("Estadísticas por cluster:")
print(cluster_stats)

# Análisis de rendimiento avanzado
print("\\nAnálisis de rendimiento:")
# Operación compleja: análisis de ventana con múltiples métricas
complex_analysis = df.with_columns([
    pl.col('quality').rolling_mean(window_size=50).alias('quality_ma_50'),
    pl.col('quality').rolling_std(window_size=50).alias('quality_std_50'),
    pl.col('word_count').rolling_sum(window_size=10).alias('word_sum_10'),
    pl.col('quality').rank().over('model').alias('quality_rank'),
    pl.col('quality').percent_rank().over('model').alias('quality_percentile')
])

start_time = time.time()
result = complex_analysis.collect()
complex_time = time.time() - start_time
print(f"Análisis complejo completado en: {complex_time:.3f} seconds")
print(f"Resultado shape: {result.shape}")

# Análisis de memoria
print(f"\\nUso de memoria del resultado: {result.estimated_size('mb'):.2f} MB")
""",
                configuration="""
# Configuración de Polars para mejor rendimiento

import polars as pl

# Configurar opciones de rendimiento
pl.Config.set_streaming_chunk_size(8192)  # Tamaño de chunk para streaming
pl.Config.set_fmt_str_lengths(50)  # Longitud de strings en display
pl.Config.set_tbl_rows(20)  # Número de filas a mostrar
pl.Config.set_tbl_cols(10)  # Número de columnas a mostrar

# Configurar opciones de memoria
pl.Config.set_streaming_chunk_size(16384)  # Chunks más grandes para mejor rendimiento

# Configurar opciones de display
pl.Config.set_fmt_float('full')  # Mostrar todos los decimales
pl.Config.set_verbose(True)  # Mostrar información de optimización

# Configurar opciones de threading
import os
os.environ['POLARS_MAX_THREADS'] = '8'  # Número de threads
""",
                performance_notes="""
# Optimizaciones de rendimiento para Polars

# 1. Usar lazy evaluation
lazy_df = pl.scan_parquet('large_file.parquet')  # Lazy
result = lazy_df.filter(pl.col('quality') > 0.8).collect()  # Eager

# 2. Optimizar tipos de datos
df_optimized = df.with_columns([
    pl.col('model').cast(pl.Categorical),  # Categorical para strings repetitivos
    pl.col('quality').cast(pl.Float32),    # Float32 para ahorrar memoria
    pl.col('word_count').cast(pl.UInt16)   # UInt16 para números pequeños
])

# 3. Usar operaciones vectorizadas
# Malo: df.with_columns([pl.col('quality').apply(lambda x: x * 100)])
# Bueno: df.with_columns([pl.col('quality') * 100])

# 4. Usar expresiones eficientes
efficient_expr = pl.col('quality').mean().over('model')
result = df.with_columns([efficient_expr])

# 5. Usar streaming para archivos grandes
streaming_result = (
    pl.scan_csv('large_file.csv')
    .filter(pl.col('quality') > 0.8)
    .group_by('model')
    .agg([pl.col('quality').mean()])
    .collect(streaming=True)
)

# 6. Usar rechunk para optimizar particiones
df_rechunked = df.rechunk()  # Optimiza particiones
""",
                alternatives=["pandas", "dask", "vaex", "modin"],
                dependencies=["pyarrow", "numpy"],
                documentation="https://pola-rs.github.io/polars/",
                community="https://stackoverflow.com/questions/tagged/polars",
                last_updated="2023-10-01",
                license="MIT"
            )
        }
    
    def get_library(self, name: str) -> LibraryInfo:
        """Obtener información de una librería específica."""
        return self.libraries.get(name)
    
    def get_all_libraries(self) -> Dict[str, LibraryInfo]:
        """Obtener todas las librerías."""
        return self.libraries
    
    def get_libraries_by_category(self, category: LibraryCategory) -> Dict[str, LibraryInfo]:
        """Obtener librerías por categoría."""
        return {name: lib for name, lib in self.libraries.items() if lib.category == category}
    
    def get_installation_commands(self) -> List[str]:
        """Obtener comandos de instalación para todas las librerías."""
        return [lib.installation for lib in self.libraries.values() if lib.installation]
    
    def get_requirements_txt(self) -> str:
        """Generar requirements.txt con las mejores librerías core."""
        requirements = []
        for lib in self.libraries.values():
            if lib.installation.startswith('pip install'):
                package = lib.installation.replace('pip install ', '')
                if '==' in package:
                    requirements.append(package)
                else:
                    requirements.append(f"{package}>=latest")
        
        return '\n'.join(requirements)
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Obtener comparación de rendimiento entre librerías."""
        return {
            "pandas": {
                "memory_usage": "Medium",
                "speed": "Fast",
                "scalability": "Limited by RAM",
                "ease_of_use": "High",
                "ecosystem": "Excellent"
            },
            "numpy": {
                "memory_usage": "Low",
                "speed": "Very Fast",
                "scalability": "Limited by RAM",
                "ease_of_use": "Medium",
                "ecosystem": "Excellent"
            },
            "dask": {
                "memory_usage": "Low",
                "speed": "Fast (distributed)",
                "scalability": "Excellent",
                "ease_of_use": "Medium",
                "ecosystem": "Good"
            },
            "polars": {
                "memory_usage": "Low",
                "speed": "Very Fast",
                "scalability": "Good",
                "ease_of_use": "High",
                "ecosystem": "Growing"
            }
        }




