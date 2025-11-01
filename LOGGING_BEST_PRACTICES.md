# 📝 Mejores Prácticas de Logging - Blatam Academy Features

## 🎯 Principios de Logging

### Estructura de Logs

```python
import logging
import json
from datetime import datetime

# Configurar logger estructurado
logger = logging.getLogger('blatam_academy')

# Formato estructurado
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Handler para archivo
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)

# Handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
```

### Logging Estructurado (JSON)

```python
import json
import logging

class StructuredLogger:
    """Logger con formato JSON."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log(self, level, message, **kwargs):
        """Log estructurado."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(log_entry))

# Uso
logger = StructuredLogger('blatam_academy')
logger.log('info', 'Request processed', 
           user_id=123, 
           endpoint='/api/query',
           latency_ms=45.2)
```

## 📊 Niveles de Logging

### Cuándo Usar Cada Nivel

```python
# DEBUG: Información detallada para debugging
logger.debug("Cache lookup for key: %s", cache_key)

# INFO: Eventos normales del sistema
logger.info("Request processed successfully", extra={
    'user_id': user_id,
    'endpoint': endpoint,
    'latency_ms': latency
})

# WARNING: Situaciones inesperadas pero manejables
logger.warning("Cache hit rate below threshold", extra={
    'hit_rate': 0.45,
    'threshold': 0.5
})

# ERROR: Errores que no detienen el sistema
logger.error("Failed to persist cache", extra={
    'error': str(e),
    'path': persistence_path
}, exc_info=True)

# CRITICAL: Errores críticos que detienen el sistema
logger.critical("System failure", extra={
    'component': 'database',
    'error': str(e)
}, exc_info=True)
```

## 🔒 Logging Seguro

### Redactar Información Sensible

```python
import re

def redact_sensitive_data(message):
    """Redactar información sensible de logs."""
    # Redactar API keys
    message = re.sub(r'sk-[a-zA-Z0-9]{32,}', 'sk-***', message)
    
    # Redactar passwords
    message = re.sub(r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+', 
                     'password="***"', message)
    
    # Redactar emails (opcional, según necesidad)
    # message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
    #                  '***@***', message)
    
    return message

# Uso
logger.info(redact_sensitive_data(f"Processing request with API key: {api_key}"))
```

### Logging de Auditoría

```python
class AuditLogger:
    """Logger especializado para auditoría."""
    
    def __init__(self):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        # Handler separado para auditoría
        audit_handler = logging.FileHandler('audit.log')
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        ))
        self.logger.addHandler(audit_handler)
    
    def log_access(self, user_id, action, resource, result='success'):
        """Log de acceso."""
        self.logger.info(json.dumps({
            'event': 'access',
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        }))
    
    def log_config_change(self, user_id, config_key, old_value, new_value):
        """Log de cambio de configuración."""
        self.logger.info(json.dumps({
            'event': 'config_change',
            'user_id': user_id,
            'config_key': config_key,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'timestamp': datetime.utcnow().isoformat()
        }))

# Uso
audit_logger = AuditLogger()
audit_logger.log_access(user_id=123, action='read', resource='/api/query')
```

## 📈 Logging de Performance

### Logging de Métricas

```python
class PerformanceLogger:
    """Logger para métricas de performance."""
    
    def log_request(self, endpoint, method, latency_ms, cache_hit=False):
        """Log de request con métricas."""
        logger.info("Request processed", extra={
            'type': 'request',
            'endpoint': endpoint,
            'method': method,
            'latency_ms': latency_ms,
            'cache_hit': cache_hit,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_cache_stats(self, stats):
        """Log de estadísticas de cache."""
        logger.info("Cache statistics", extra={
            'type': 'cache_stats',
            'hit_rate': stats['hit_rate'],
            'hits': stats['cache_hits'],
            'misses': stats['cache_misses'],
            'memory_usage': stats['memory_usage'],
            'timestamp': datetime.utcnow().isoformat()
        })

# Uso
perf_logger = PerformanceLogger()
perf_logger.log_request('/api/query', 'POST', 45.2, cache_hit=True)
```

## 🔍 Context Managers para Logging

### Logging con Contexto

```python
from contextlib import contextmanager
import time

@contextmanager
def log_execution_time(operation_name):
    """Context manager para logging de tiempo de ejecución."""
    start = time.time()
    logger.info(f"Starting {operation_name}")
    
    try:
        yield
        duration = time.time() - start
        logger.info(f"Completed {operation_name} in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start
        logger.error(f"Failed {operation_name} after {duration:.2f}s: {e}", exc_info=True)
        raise

# Uso
with log_execution_time("cache_operation"):
    result = await engine.process_request(request)
```

## 📊 Configuración de Logging por Entorno

### Desarrollo

```python
# logging_config_dev.py
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

### Producción

```python
# logging_config_prod.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/blatam/app.log'),
        logging.handlers.RotatingFileHandler(
            '/var/log/blatam/app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)
```

### Structured Logging para ELK

```python
# logging_config_elk.py
import json

class ELKFormatter(logging.Formatter):
    """Formatter para ELK Stack."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Agregar extras si existen
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        
        return json.dumps(log_entry)

# Configurar
handler = logging.StreamHandler()
handler.setFormatter(ELKFormatter())
logger.addHandler(handler)
```

## ✅ Checklist de Logging

### Pre-Deployment
- [ ] Logging estructurado configurado
- [ ] Niveles de log apropiados
- [ ] Información sensible redactada
- [ ] Logs rotando (no infinitos)
- [ ] Logs enviados a sistema centralizado (ELK)

### Producción
- [ ] Log level = INFO o WARNING
- [ ] No DEBUG en producción
- [ ] Audit logs separados
- [ ] Performance logs configurados
- [ ] Error logs con stack traces

---

**Más información:**
- [Best Practices](BEST_PRACTICES.md)
- [Security Guide](SECURITY_GUIDE.md)
- [Production Ready](bulk/PRODUCTION_READY.md)

