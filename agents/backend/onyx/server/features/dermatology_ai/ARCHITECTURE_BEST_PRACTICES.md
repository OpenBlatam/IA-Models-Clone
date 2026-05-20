# Mejores Prácticas - Arquitectura V8.0

## 📋 Tabla de Contenidos

1. [Principios de Diseño](#principios-de-diseño)
2. [Patrones de Código](#patrones-de-código)
3. [Naming Conventions](#naming-conventions)
4. [Error Handling](#error-handling)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Tips](#performance-tips)
7. [Security Best Practices](#security-best-practices)

---

## 🎯 Principios de Diseño

### 1. Single Responsibility Principle (SRP)

**✅ Correcto:**
```python
class ImageAnalysisService:
    """Servicio enfocado solo en análisis de imágenes"""
    
    async def analyze_image(self, image_data: bytes) -> Analysis:
        """Analiza imagen y retorna resultados"""
        pass
```

**❌ Incorrecto:**
```python
class ImageService:
    """Servicio que hace demasiadas cosas"""
    
    async def analyze_image(self, image_data: bytes) -> Analysis:
        """Analiza imagen"""
        pass
    
    async def save_to_database(self, analysis: Analysis):
        """Guarda en base de datos"""
        pass
    
    async def send_notification(self, user_id: str):
        """Envía notificación"""
        pass
```

### 2. Dependency Inversion Principle (DIP)

**✅ Correcto:**
```python
class SkincareRecommender:
    def __init__(self, product_repository: IProductRepository):
        """Depende de abstracción, no de implementación"""
        self.product_repository = product_repository
```

**❌ Incorrecto:**
```python
class SkincareRecommender:
    def __init__(self):
        """Depende directamente de implementación"""
        self.product_repository = SQLiteProductRepository()
```

### 3. Interface Segregation Principle (ISP)

**✅ Correcto:**
```python
# Interfaces específicas y pequeñas
class IProductReader(Protocol):
    async def get_all(self) -> List[Product]: ...

class IProductWriter(Protocol):
    async def save(self, product: Product): ...

# Clase puede implementar solo lo que necesita
class ReadOnlyProductRepository(IProductReader):
    async def get_all(self) -> List[Product]:
        pass
```

**❌ Incorrecto:**
```python
# Interface grande que fuerza implementar todo
class IProductRepository(Protocol):
    async def get_all(self) -> List[Product]: ...
    async def save(self, product: Product): ...
    async def delete(self, product_id: str): ...
    async def update(self, product: Product): ...
    async def search(self, query: str): ...
    # ... 20 métodos más
```

---

## 💻 Patrones de Código

### 1. Factory Pattern para Servicios

```python
# ✅ Correcto: Factory con configuración
class ServiceFactory:
    def create_recommender(self, config: Dict) -> IRecommendationService:
        if config.get("use_ml"):
            return MLRecommender(config)
        return SkincareRecommender(config)

# ❌ Incorrecto: Lógica de creación dispersa
def get_recommender():
    if settings.USE_ML:
        return MLRecommender()
    return SkincareRecommender()
```

### 2. Repository Pattern

```python
# ✅ Correcto: Repository abstracto
class IAnalysisRepository(Protocol):
    async def save(self, analysis: Analysis) -> None: ...
    async def get_by_id(self, analysis_id: str) -> Optional[Analysis]: ...
    async def get_by_user(self, user_id: str) -> List[Analysis]: ...

# Implementación concreta
class AnalysisRepository(IAnalysisRepository):
    def __init__(self, database_adapter: IDatabaseAdapter):
        self.db = database_adapter
    
    async def save(self, analysis: Analysis) -> None:
        await self.db.insert("analyses", analysis.to_dict())
```

### 3. Strategy Pattern

```python
# ✅ Correcto: Estrategias intercambiables
class RecommendationStrategy(Protocol):
    async def recommend(self, analysis: Analysis) -> List[Product]: ...

class MLRecommendationStrategy:
    async def recommend(self, analysis: Analysis) -> List[Product]:
        # Usa ML model
        pass

class RuleBasedRecommendationStrategy:
    async def recommend(self, analysis: Analysis) -> List[Product]:
        # Usa reglas
        pass

class Recommender:
    def __init__(self, strategy: RecommendationStrategy):
        self.strategy = strategy
    
    async def generate(self, analysis: Analysis) -> List[Product]:
        return await self.strategy.recommend(analysis)
```

---

## 📝 Naming Conventions

### Servicios

```python
# ✅ Correcto
class ImageAnalysisService:
    """Servicio de análisis de imágenes"""
    pass

class SkincareRecommender:
    """Recomendador de skincare"""
    pass

# ❌ Incorrecto
class ImageService:  # Muy genérico
    pass

class Recommender:  # Falta contexto
    pass
```

### Métodos

```python
# ✅ Correcto: Verbos claros
async def analyze_image(image_data: bytes) -> Analysis:
    pass

async def generate_recommendations(analysis: Analysis) -> List[Product]:
    pass

# ❌ Incorrecto: Nombres ambiguos
async def process(data: bytes):  # ¿Qué procesa?
    pass

async def get(data: dict):  # ¿Qué obtiene?
    pass
```

### Variables

```python
# ✅ Correcto: Descriptivo
user_analysis_results = await analyzer.analyze(user_image)
recommended_products = await recommender.generate(analysis)

# ❌ Incorrecto: Abreviaciones o nombres genéricos
res = await a.analyze(img)
prods = await r.generate(a)
```

---

## ⚠️ Error Handling

### 1. Excepciones Específicas

```python
# ✅ Correcto: Excepciones específicas del dominio
class InvalidImageError(Exception):
    """Error cuando imagen es inválida"""
    pass

class AnalysisFailedError(Exception):
    """Error cuando análisis falla"""
    pass

# Uso
async def analyze_image(image_data: bytes) -> Analysis:
    if not image_data:
        raise InvalidImageError("image_data is required")
    
    try:
        return await self._perform_analysis(image_data)
    except ProcessingError as e:
        raise AnalysisFailedError(f"Analysis failed: {e}") from e

# ❌ Incorrecto: Usar Exception genérica
async def analyze_image(image_data: bytes) -> Analysis:
    if not image_data:
        raise Exception("Error")  # Muy genérico
```

### 2. Error Context

```python
# ✅ Correcto: Incluir contexto útil
class AnalysisFailedError(Exception):
    def __init__(self, message: str, analysis_id: str, user_id: str):
        super().__init__(message)
        self.analysis_id = analysis_id
        self.user_id = user_id

# Uso
try:
    analysis = await self._perform_analysis(image_data)
except ProcessingError as e:
    raise AnalysisFailedError(
        f"Analysis failed: {e}",
        analysis_id=analysis.id,
        user_id=user_id
    ) from e
```

### 3. Logging de Errores

```python
# ✅ Correcto: Logging con contexto
logger.error(
    "Analysis failed",
    extra={
        "analysis_id": analysis_id,
        "user_id": user_id,
        "error_type": type(e).__name__,
        "error_message": str(e)
    },
    exc_info=True
)

# ❌ Incorrecto: Logging sin contexto
logger.error("Error")  # No hay información útil
```

---

## 🧪 Testing Guidelines

### 1. Estructura de Tests

```python
# ✅ Correcto: Arrange-Act-Assert
async def test_analyze_image_success():
    # Arrange
    image_data = b"fake_image_data"
    mock_processor = Mock(IImageProcessor)
    mock_processor.process.return_value = {"score": 75.0}
    service = ImageAnalysisService(mock_processor)
    
    # Act
    result = await service.analyze_image(image_data)
    
    # Assert
    assert result.score == 75.0
    mock_processor.process.assert_called_once_with(image_data)
```

### 2. Test Naming

```python
# ✅ Correcto: Nombres descriptivos
def test_analyze_image_with_valid_data_returns_analysis():
    pass

def test_analyze_image_with_invalid_data_raises_error():
    pass

def test_generate_recommendations_for_dry_skin_includes_moisturizer():
    pass

# ❌ Incorrecto: Nombres genéricos
def test_analyze():
    pass

def test_recommendations():
    pass
```

### 3. Mocks y Stubs

```python
# ✅ Correcto: Mock de interfaces
mock_repository = Mock(IAnalysisRepository)
mock_repository.get_by_id.return_value = analysis

# ❌ Incorrecto: Mock de implementaciones concretas
mock_repository = Mock(AnalysisRepository)  # Acoplamiento fuerte
```

---

## ⚡ Performance Tips

### 1. Async/Await Correcto

```python
# ✅ Correcto: Usar async cuando hay I/O
async def analyze_image(image_data: bytes) -> Analysis:
    processed = await self.image_processor.process(image_data)
    metrics = await self.ml_model.predict(processed)
    return Analysis(metrics=metrics)

# ❌ Incorrecto: Bloquear en async
async def analyze_image(image_data: bytes) -> Analysis:
    processed = self.image_processor.process(image_data)  # Bloquea
    return Analysis(metrics=processed)
```

### 2. Caching Estratégico

```python
# ✅ Correcto: Cachear resultados costosos
@lru_cache(maxsize=100)
def get_product_recommendations(skin_type: str) -> List[Product]:
    # Cálculo costoso
    return expensive_calculation(skin_type)

# ❌ Incorrecto: Cachear todo
@lru_cache(maxsize=1000)  # Cache muy grande
def process_user_input(user_input: str) -> str:
    return user_input.upper()  # Operación barata, no necesita cache
```

### 3. Batch Processing

```python
# ✅ Correcto: Procesar en batches
async def analyze_multiple_images(images: List[bytes]) -> List[Analysis]:
    batch_size = 10
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        batch_results = await asyncio.gather(*[
            self.analyze_image(img) for img in batch
        ])
        results.extend(batch_results)
    
    return results

# ❌ Incorrecto: Procesar uno por uno
async def analyze_multiple_images(images: List[bytes]) -> List[Analysis]:
    results = []
    for img in images:
        result = await self.analyze_image(img)  # Secuencial
        results.append(result)
    return results
```

---

## 🔒 Security Best Practices

### 1. Input Validation

```python
# ✅ Correcto: Validar y sanitizar entrada
async def analyze_image(image_data: bytes, user_id: str):
    # Validar user_id
    if not user_id or not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise ValueError("Invalid user_id")
    
    # Validar tamaño de imagen
    if len(image_data) > 10 * 1024 * 1024:  # 10MB
        raise ValueError("Image too large")
    
    # Validar formato
    if not image_data.startswith((b'\xff\xd8', b'\x89PNG')):
        raise ValueError("Invalid image format")
    
    return await self._analyze(image_data)

# ❌ Incorrecto: Sin validación
async def analyze_image(image_data: bytes, user_id: str):
    return await self._analyze(image_data)  # Confía en entrada
```

### 2. SQL Injection Prevention

```python
# ✅ Correcto: Usar parámetros
async def get_analysis(self, user_id: str) -> List[Analysis]:
    query = "SELECT * FROM analyses WHERE user_id = ?"
    return await self.db.execute(query, (user_id,))

# ❌ Incorrecto: Concatenación de strings
async def get_analysis(self, user_id: str) -> List[Analysis]:
    query = f"SELECT * FROM analyses WHERE user_id = '{user_id}'"  # Vulnerable
    return await self.db.execute(query)
```

### 3. Rate Limiting

```python
# ✅ Correcto: Rate limiting en endpoints
@router.post("/analyze")
@rate_limit(requests_per_minute=10)
async def analyze_image(request: AnalysisRequest):
    return await analysis_service.analyze(request.image_data)

# ❌ Incorrecto: Sin rate limiting
@router.post("/analyze")
async def analyze_image(request: AnalysisRequest):
    return await analysis_service.analyze(request.image_data)  # Vulnerable a abuso
```

---

## 📚 Code Organization

### 1. Imports Organizados

```python
# ✅ Correcto: Imports organizados
# Standard library
import asyncio
from typing import List, Optional
from dataclasses import dataclass

# Third-party
from fastapi import APIRouter, Depends
from pydantic import BaseModel

# Local
from core.domain.interfaces import IAnalysisService
from features.analysis.services.image_analysis import ImageAnalysisService

# ❌ Incorrecto: Imports desordenados
from fastapi import APIRouter
import asyncio
from features.analysis.services.image_analysis import ImageAnalysisService
from typing import List
from core.domain.interfaces import IAnalysisService
```

### 2. Docstrings

```python
# ✅ Correcto: Docstrings completos
class ImageAnalysisService:
    """
    Servicio de análisis de imágenes.
    
    Procesa imágenes de piel y genera análisis con métricas
    y condiciones detectadas.
    
    Args:
        image_processor: Procesador de imágenes
        ml_model: Modelo ML opcional para análisis avanzado
    
    Example:
        >>> service = ImageAnalysisService(image_processor)
        >>> analysis = await service.analyze_image(image_data)
        >>> print(analysis.score)
    """
    
    async def analyze_image(self, image_data: bytes) -> Analysis:
        """
        Analiza imagen y retorna análisis completo.
        
        Args:
            image_data: Datos de la imagen en bytes
            
        Returns:
            Análisis con métricas y condiciones
            
        Raises:
            InvalidImageError: Si imagen es inválida
            AnalysisFailedError: Si análisis falla
        """
        pass

# ❌ Incorrecto: Sin docstrings o incompletos
class ImageAnalysisService:
    async def analyze_image(self, image_data: bytes) -> Analysis:
        pass
```

---

## 🎨 Code Style

### 1. Type Hints

```python
# ✅ Correcto: Type hints completos
async def analyze_image(
    self,
    image_data: bytes,
    metadata: Optional[Dict[str, Any]] = None
) -> Analysis:
    pass

# ❌ Incorrecto: Sin type hints
async def analyze_image(self, image_data, metadata=None):
    pass
```

### 2. Line Length

```python
# ✅ Correcto: Líneas razonables (< 100 caracteres)
result = await self.service.analyze_image(
    image_data=image_data,
    metadata={"source": "mobile_app"}
)

# ❌ Incorrecto: Líneas muy largas
result = await self.service.analyze_image(image_data=image_data, metadata={"source": "mobile_app", "device": "iphone", "version": "1.0.0", "timestamp": "2024-01-01"})
```

### 3. Magic Numbers

```python
# ✅ Correcto: Constantes con nombre
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
CACHE_TTL = 3600  # 1 hora

if len(image_data) > MAX_IMAGE_SIZE:
    raise ValueError("Image too large")

# ❌ Incorrecto: Magic numbers
if len(image_data) > 10485760:  # ¿Qué es este número?
    raise ValueError("Image too large")
```

---

## 🔄 Refactoring Guidelines

### 1. Cuando Refactorizar

- ✅ Código duplicado (DRY)
- ✅ Funciones/métodos muy largos (> 50 líneas)
- ✅ Clases con demasiadas responsabilidades
- ✅ Dependencias circulares
- ✅ Código difícil de testear

### 2. Cómo Refactorizar

1. **Escribir tests primero** (si no existen)
2. **Refactorizar en pequeños pasos**
3. **Ejecutar tests después de cada paso**
4. **No cambiar comportamiento** (solo estructura)
5. **Commit frecuente**

### 3. Ejemplo de Refactoring

```python
# ❌ Antes: Método largo y complejo
async def process_analysis(self, image_data: bytes, user_id: str):
    # Validar
    if not image_data:
        raise ValueError("No image")
    if len(image_data) > 10 * 1024 * 1024:
        raise ValueError("Too large")
    
    # Procesar
    processed = await self.processor.process(image_data)
    
    # Analizar
    metrics = await self.ml_model.predict(processed)
    conditions = await self.detector.detect(processed)
    
    # Guardar
    analysis = Analysis(metrics=metrics, conditions=conditions)
    await self.repository.save(analysis)
    
    # Notificar
    await self.notifier.send(user_id, analysis)
    
    return analysis

# ✅ Después: Métodos pequeños y enfocados
async def process_analysis(self, image_data: bytes, user_id: str) -> Analysis:
    self._validate_input(image_data)
    processed = await self._process_image(image_data)
    analysis = await self._create_analysis(processed)
    await self._persist_analysis(analysis)
    await self._notify_user(user_id, analysis)
    return analysis

def _validate_input(self, image_data: bytes):
    if not image_data:
        raise ValueError("No image")
    if len(image_data) > MAX_IMAGE_SIZE:
        raise ValueError("Too large")

async def _process_image(self, image_data: bytes) -> ProcessedImage:
    return await self.processor.process(image_data)

async def _create_analysis(self, processed: ProcessedImage) -> Analysis:
    metrics = await self.ml_model.predict(processed)
    conditions = await self.detector.detect(processed)
    return Analysis(metrics=metrics, conditions=conditions)

async def _persist_analysis(self, analysis: Analysis):
    await self.repository.save(analysis)

async def _notify_user(self, user_id: str, analysis: Analysis):
    await self.notifier.send(user_id, analysis)
```

---

**Versión:** 1.0.0  
**Fecha:** 2024




