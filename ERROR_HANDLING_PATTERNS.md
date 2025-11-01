# ⚠️ Patrones de Manejo de Errores - Blatam Academy Features

## 🎯 Principios de Error Handling

### 1. Falla Rápida (Fail Fast)

```python
def validate_input(data: dict):
    """Validar input y fallar rápido si es inválido."""
    if not data:
        raise ValueError("Data cannot be empty")
    
    if "query" not in data:
        raise ValueError("Missing required field: query")
    
    if not isinstance(data["query"], str):
        raise TypeError("Query must be a string")
    
    if len(data["query"]) > 10000:
        raise ValueError("Query too long (max 10000 chars)")
    
    return True
```

### 2. Error Handling Específico

```python
try:
    result = await cache_engine.process_request(request)
except CacheKeyError as e:
    logger.error(f"Cache key error: {e}")
    # Retry con key diferente
    result = await retry_with_different_key(request)
except CacheMemoryError as e:
    logger.error(f"Cache memory error: {e}")
    # Liberar memoria y retry
    await cache_engine.gc()
    result = await cache_engine.process_request(request)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

## 🔄 Patrones de Retry

### Exponential Backoff

```python
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry con exponential backoff."""
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
            await asyncio.sleep(delay)
            delay *= backoff_factor
    
    raise Exception("Max retries exceeded")

# Uso
result = await retry_with_backoff(
    lambda: cache_engine.process_request(request),
    max_retries=3
)
```

### Circuit Breaker Pattern

```python
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker para prevenir cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func):
        """Ejecutar función con circuit breaker."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Manejar éxito."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        
        if self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Manejar fallo."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Uso
breaker = CircuitBreaker()
result = await breaker.call(lambda: external_api.call())
```

## 📦 Error Wrapping

### Custom Exception Hierarchy

```python
class BlatamError(Exception):
    """Base exception para Blatam Academy."""
    pass

class CacheError(BlatamError):
    """Error relacionado con cache."""
    pass

class CacheKeyError(CacheError):
    """Error de key de cache."""
    pass

class CacheMemoryError(CacheError):
    """Error de memoria de cache."""
    pass

class ValidationError(BlatamError):
    """Error de validación."""
    pass

class ConfigurationError(BlatamError):
    """Error de configuración."""
    pass
```

### Context Manager para Error Handling

```python
from contextlib import contextmanager

@contextmanager
def handle_errors(context: dict):
    """Context manager para manejo de errores."""
    try:
        yield
    except ValueError as e:
        logger.error(f"Validation error in {context}: {e}")
        raise ValidationError(f"Validation failed: {e}") from e
    except KeyError as e:
        logger.error(f"Missing key in {context}: {e}")
        raise ConfigurationError(f"Missing configuration: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error in {context}: {e}", exc_info=True)
        raise BlatamError(f"Unexpected error: {e}") from e

# Uso
with handle_errors(context={"operation": "cache_get"}):
    result = cache.get(key)
```

## 🔍 Error Logging y Monitoring

### Structured Error Logging

```python
import logging
import traceback

class ErrorLogger:
    """Logger estructurado de errores."""
    
    def __init__(self, logger_name: str = "blatam_errors"):
        self.logger = logging.getLogger(logger_name)
    
    def log_error(
        self,
        error: Exception,
        context: dict,
        level: str = "ERROR"
    ):
        """Log error estructurado."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        getattr(self.logger, level.lower())(
            json.dumps(error_info, default=str)
        )

# Uso
error_logger = ErrorLogger()
try:
    result = risky_operation()
except Exception as e:
    error_logger.log_error(e, {"user_id": 123, "operation": "query"})
```

### Error Tracking Integration

```python
# Integración con Sentry
import sentry_sdk

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    environment="production",
    traces_sample_rate=1.0
)

try:
    result = risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

## 🛡️ Error Recovery

### Graceful Degradation

```python
async def process_request_with_fallback(request):
    """Procesar request con fallback."""
    try:
        # Intentar con cache
        result = await cache_engine.process_request(request)
        return result
    except CacheError:
        logger.warning("Cache failed, trying direct processing")
        # Fallback: procesar sin cache
        try:
            result = await direct_process(request)
            return result
        except Exception as e:
            logger.error(f"Direct processing failed: {e}")
            # Fallback final: respuesta por defecto
            return {"result": "Service temporarily unavailable"}
```

### Partial Success Handling

```python
async def process_batch_with_partial_success(requests):
    """Procesar batch permitiendo éxito parcial."""
    results = []
    errors = []
    
    for request in requests:
        try:
            result = await process_request(request)
            results.append({"status": "success", "data": result})
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            errors.append({"status": "error", "error": str(e)})
            results.append({"status": "error", "error": str(e)})
    
    return {
        "results": results,
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": len(errors),
        "errors": errors
    }
```

## ✅ Checklist de Error Handling

### Diseño
- [ ] Jerarquía de excepciones definida
- [ ] Error messages descriptivos
- [ ] Context information incluido
- [ ] Error recovery implementado

### Implementación
- [ ] Try-catch apropiados
- [ ] Retry logic implementado
- [ ] Circuit breaker para servicios externos
- [ ] Fallback strategies definidas

### Observabilidad
- [ ] Errors logueados estructuradamente
- [ ] Error tracking configurado
- [ ] Alertas configuradas
- [ ] Error dashboard disponible

---

**Más información:**
- [Security Guide](SECURITY_GUIDE.md)
- [Best Practices](BEST_PRACTICES_SUMMARY.md)
- [Troubleshooting](TROUBLESHOOTING_BY_SYMPTOM.md)

