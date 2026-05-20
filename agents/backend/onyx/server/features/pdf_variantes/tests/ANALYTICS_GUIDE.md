# Guía de Analytics

## Introducción

Las utilidades de analytics permiten analizar resultados de tests, generar métricas y comparar con baselines.

## PlaywrightAnalytics

### Uso Básico

```python
from playwright_analytics import create_analytics

def test_with_analytics(page, api_base_url):
    analytics = create_analytics()
    
    # Registrar métricas de un test
    start_time = time.time()
    # ... ejecutar test ...
    duration = time.time() - start_time
    
    requests = [
        {"duration": 100, "status": 200},
        {"duration": 150, "status": 200}
    ]
    
    analytics.record_test_metrics(
        test_name="test_example",
        duration=duration,
        status="passed",
        requests=requests
    )
    
    # Calcular métricas de suite
    suite_metrics = analytics.calculate_suite_metrics()
    
    # Generar reporte
    report_path = analytics.generate_report(format="html")
```

### Registrar Métricas de Test

```python
analytics = create_analytics()

analytics.record_test_metrics(
    test_name="test_upload",
    duration=2.5,
    status="passed",
    requests=[
        {"duration": 100, "status": 200},
        {"duration": 200, "status": 201}
    ],
    memory_usage=50.5,  # MB
    cpu_usage=25.0      # %
)
```

### Calcular Métricas de Suite

```python
suite_metrics = analytics.calculate_suite_metrics("My Test Suite")

print(f"Total tests: {suite_metrics.total_tests}")
print(f"Passed: {suite_metrics.passed}")
print(f"Failed: {suite_metrics.failed}")
print(f"Total duration: {suite_metrics.total_duration}s")
print(f"Avg response time: {suite_metrics.avg_response_time}ms")
```

### Generar Reportes

#### Reporte JSON

```python
report_path = analytics.generate_report(format="json")
# Guardado en analytics/analytics_1234567890.json
```

#### Reporte HTML

```python
report_path = analytics.generate_report(format="html")
# Guardado en analytics/analytics_1234567890.html
```

### Comparar con Baseline

```python
from pathlib import Path

# Cargar baseline
baseline_path = Path("analytics/baseline.json")

# Comparar
comparison = analytics.compare_with_baseline(baseline_path)

print(f"Duration change: {comparison['duration_percent_change']:.2f}%")
print(f"Response time change: {comparison['response_time_percent_change']:.2f}%")
print(f"Failure rate change: {comparison['failure_rate_diff']:.4f}")
```

### Identificar Tests Lentos

```python
slow_tests = analytics.identify_slow_tests(threshold=5.0)  # > 5 segundos

for test in slow_tests:
    print(f"{test.test_name}: {test.duration:.2f}s")
```

### Identificar Tests Flaky

```python
# Múltiples ejecuciones
run1_metrics = [...]  # Lista de TestMetrics
run2_metrics = [...]
run3_metrics = [...]

flaky_tests = analytics.identify_flaky_tests([
    run1_metrics,
    run2_metrics,
    run3_metrics
])

print(f"Flaky tests: {flaky_tests}")
```

### Generar Tendencias

```python
# Datos históricos
historical_data = [
    suite_metrics_1,
    suite_metrics_2,
    suite_metrics_3
]

trends = analytics.generate_trends(historical_data)

print(f"Duration trend: {trends['duration_trend']['trend']}")
print(f"Response time trend: {trends['response_time_trend']['trend']}")
print(f"Failure rate trend: {trends['failure_rate_trend']['trend']}")
```

## Integración con Tests

### Fixture de Analytics

```python
# conftest.py
@pytest.fixture(scope="session")
def analytics():
    from playwright_analytics import create_analytics
    return create_analytics()

# test_file.py
def test_with_analytics(page, analytics, api_base_url):
    start_time = time.time()
    
    # ... ejecutar test ...
    
    duration = time.time() - start_time
    requests = [...]  # Capturar requests
    
    analytics.record_test_metrics(
        test_name="test_example",
        duration=duration,
        status="passed",
        requests=requests
    )
```

### Hook para Auto-registro

```python
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call":
        if "analytics" in item.fixturenames:
            analytics = item.funcargs.get("analytics")
            if analytics:
                # Capturar métricas automáticamente
                # (requiere implementación adicional)
                pass
```

## Estructura de Métricas

### TestMetrics

```python
@dataclass
class TestMetrics:
    test_name: str
    duration: float
    status: str  # passed, failed, skipped
    requests_count: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    memory_usage: Optional[float]
    cpu_usage: Optional[float]
```

### SuiteMetrics

```python
@dataclass
class SuiteMetrics:
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    total_duration: float
    avg_test_duration: float
    total_requests: int
    failed_requests: int
    avg_response_time: float
    timestamp: float
    test_metrics: List[TestMetrics]
```

## Mejores Prácticas

1. **Registrar métricas consistentemente**: Para todos los tests importantes
2. **Establecer baseline**: Guardar métricas iniciales para comparación
3. **Revisar tendencias**: Monitorear cambios en performance
4. **Identificar problemas**: Usar identificación de tests lentos/flaky
5. **Generar reportes regularmente**: Para análisis continuo
6. **Comparar con baseline**: Detectar regresiones

## Output Directory

Por defecto, los reportes se guardan en `analytics/`. Puedes cambiar esto:

```python
analytics = create_analytics(output_dir="custom_analytics")
```

## Uso en CI/CD

```python
# En pipeline de CI/CD
analytics = create_analytics()

# Ejecutar tests y registrar métricas
# ...

# Generar reportes
report_path = analytics.generate_report(format="html")

# Comparar con baseline
baseline_path = Path("analytics/baseline.json")
if baseline_path.exists():
    comparison = analytics.compare_with_baseline(baseline_path)
    
    # Fallar si hay regresión significativa
    if comparison["duration_percent_change"] > 20:
        raise Exception("Significant performance regression detected")
```



