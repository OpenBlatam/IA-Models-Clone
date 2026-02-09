# Tests para Robot Maintenance AI

## Ejecutar Tests

### Todos los tests
```bash
pytest tests/
```

### Tests específicos
```bash
pytest tests/test_validators.py
pytest tests/test_cache_manager.py
pytest tests/test_rate_limiter.py
```

### Con cobertura
```bash
pytest tests/ --cov=. --cov-report=html
```

## Estructura de Tests

- `test_validators.py`: Tests para validación de inputs
- `test_cache_manager.py`: Tests para el sistema de caché
- `test_rate_limiter.py`: Tests para rate limiting
- `conftest.py`: Fixtures compartidos

## Requisitos

```bash
pip install pytest pytest-cov
```






