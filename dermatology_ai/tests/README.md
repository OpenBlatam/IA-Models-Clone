# Tests para Dermatology AI

Este directorio contiene la suite completa de tests para el sistema Dermatology AI.

## Estructura de Tests

```
tests/
├── conftest.py                      # Configuración y fixtures compartidos
├── test_utils.py                    # Utilidades y helpers para testing
│
├── API & Endpoints
├── test_api_routers.py             # Tests para endpoints de API
│
├── Application Layer
├── test_use_cases_comprehensive.py # Tests para casos de uso
├── test_use_cases.py               # Tests básicos de casos de uso
│
├── Domain Layer
├── test_domain_entities.py         # Tests para entidades de dominio
├── test_domain.py                  # Tests básicos de dominio
├── test_services.py                # Tests para servicios de dominio
│
├── Infrastructure Layer
├── test_infrastructure.py          # Tests para capa de infraestructura
├── test_mappers.py                 # Tests para mappers
├── test_validators.py              # Tests para validadores
│
├── Architecture Patterns
├── test_cqrs.py                    # Tests para patrón CQRS
├── test_event_sourcing.py          # Tests para event sourcing
│
├── Cross-Cutting Concerns
├── test_middleware.py              # Tests para middleware
├── test_performance.py             # Tests de performance y caching
├── test_security.py                # Tests de seguridad
├── test_configuration.py           # Tests de configuración
│
├── ML & Advanced Features
├── test_ml_components.py           # Tests para componentes ML
├── test_skin_analyzer.py           # Tests para SkinAnalyzer (legacy)
│
├── Integration & Edge Cases
├── test_integration.py             # Tests de integración end-to-end
├── test_edge_cases.py              # Tests para casos límite
├── test_comprehensive.py           # Tests comprehensivos adicionales
├── test_advanced_features.py       # Tests para características avanzadas
│
└── Documentation
    ├── README.md                   # Este archivo
    ├── TEST_SUMMARY.md             # Resumen de tests
    └── test_improvements_summary.md # Resumen de mejoras
```

## Ejecutar Tests

### Ejecutar todos los tests

```bash
pytest tests/
```

### Ejecutar tests específicos

```bash
# Tests de API
pytest tests/test_api_routers.py

# Tests de casos de uso
pytest tests/test_use_cases_comprehensive.py

# Tests de integración
pytest tests/test_integration.py -m integration

# Tests de dominio
pytest tests/test_domain_entities.py -m domain
```

### Ejecutar con cobertura

```bash
pytest --cov=core --cov=api --cov-report=html
```

### Ejecutar tests en paralelo

```bash
pytest -n auto
```

## Tipos de Tests

### Unit Tests
- **test_domain_entities.py**: Tests para entidades y value objects del dominio
- **test_services.py**: Tests para servicios de dominio
- **test_infrastructure.py**: Tests para repositorios y adapters

### Integration Tests
- **test_integration.py**: Tests end-to-end que verifican flujos completos
- **test_api_routers.py**: Tests de endpoints de API con mocks

### Use Case Tests
- **test_use_cases_comprehensive.py**: Tests para todos los casos de uso de la aplicación

## Fixtures Disponibles

### Fixtures de Repositorios
- `mock_analysis_repository`: Mock del repositorio de análisis
- `mock_user_repository`: Mock del repositorio de usuarios
- `mock_product_repository`: Mock del repositorio de productos

### Fixtures de Servicios
- `mock_image_processor`: Mock del procesador de imágenes
- `mock_cache_service`: Mock del servicio de cache
- `mock_event_publisher`: Mock del publicador de eventos
- `mock_analysis_service`: Mock del servicio de análisis
- `mock_recommendation_service`: Mock del servicio de recomendaciones

### Fixtures de Datos
- `sample_analysis_data`: Datos de ejemplo para análisis
- `sample_user_data`: Datos de ejemplo para usuarios
- `sample_image_bytes`: Bytes de imagen de ejemplo
- `sample_video_bytes`: Bytes de video de ejemplo
- `sample_metrics`: Métricas de piel de ejemplo
- `sample_conditions`: Condiciones de ejemplo
- `complete_analysis`: Análisis completo con todos los datos

### Fixtures de Utilidades
- `test_data_factory`: Factory para crear datos de prueba
- `test_assertions`: Assertions personalizadas
- `test_helpers`: Funciones helper
- `performance_monitor`: Monitor de performance
- `cache_strategy_manager`: Manager de estrategias de cache
- `query_optimizer`: Optimizador de queries
- `security_validator`: Validador de seguridad

### Fixtures de Aplicación
- `service_factory`: Factory de servicios para testing
- `plugin_registry`: Registry de plugins para testing
- `client`: Cliente de test para FastAPI

## Marcadores de Tests

Los tests pueden ser marcados con diferentes categorías:

```python
@pytest.mark.unit
def test_unit_example():
    """Test unitario"""
    pass

@pytest.mark.integration
def test_integration_example():
    """Test de integración"""
    pass

@pytest.mark.asyncio
async def test_async_example():
    """Test asíncrono"""
    pass

@pytest.mark.slow
def test_slow_example():
    """Test lento"""
    pass
```

Ejecutar tests por marcador:

```bash
# Solo tests unitarios
pytest -m unit

# Solo tests de integración
pytest -m integration

# Excluir tests lentos
pytest -m "not slow"
```

## Cobertura de Código

El objetivo de cobertura es mínimo del 75%. Ver el reporte:

```bash
# Cobertura completa
pytest --cov=core --cov=api --cov=config --cov=middleware --cov-report=html

# Ver reporte
open htmlcov/index.html

# Cobertura con detalle
pytest --cov=core --cov=api --cov-report=term-missing --cov-report=html
```

### Áreas Cubiertas
- ✅ API endpoints (routers)
- ✅ Use cases (casos de uso)
- ✅ Domain entities (entidades)
- ✅ Infrastructure (repositorios, adapters)
- ✅ Services (servicios de dominio)
- ✅ Validators (validadores)
- ✅ Mappers (mapeadores)
- ✅ Middleware (middleware de API)
- ✅ CQRS (comandos y queries)
- ✅ Event Sourcing (eventos)
- ✅ ML Components (componentes ML)
- ✅ Performance (caching, optimización)
- ✅ Security (sanitización, validación)
- ✅ Configuration (configuración)

## Mejores Prácticas

1. **Usar fixtures compartidos**: Reutilizar fixtures de `conftest.py` cuando sea posible
2. **Mockear dependencias externas**: Usar mocks para servicios externos, bases de datos, etc.
3. **Tests independientes**: Cada test debe poder ejecutarse de forma independiente
4. **Nombres descriptivos**: Usar nombres claros que describan qué se está probando
5. **Arrange-Act-Assert**: Seguir el patrón AAA en los tests
6. **Tests rápidos**: Mantener los tests unitarios rápidos (< 1 segundo)

## Troubleshooting

### Error: "No module named 'core'"
Asegúrate de estar ejecutando los tests desde el directorio raíz del proyecto:

```bash
cd agents/backend/onyx/server/features/dermatology_ai
pytest
```

### Error: "Event loop is closed"
Para tests asíncronos, asegúrate de que `pytest-asyncio` esté instalado y configurado correctamente.

### Tests que fallan intermitentemente
- Verifica que los mocks estén configurados correctamente
- Asegúrate de que no haya dependencias de estado compartido
- Revisa que los fixtures se estén limpiando correctamente

## Contribuir

Al agregar nuevos tests:

1. Coloca el test en el archivo apropiado según su tipo
2. Usa fixtures existentes cuando sea posible
3. Agrega nuevos fixtures a `conftest.py` si son reutilizables
4. Asegúrate de que el test pase antes de hacer commit
5. Verifica la cobertura de código

