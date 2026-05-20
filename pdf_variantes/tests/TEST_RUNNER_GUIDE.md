# Guía de Test Runner

## Introducción

El Test Runner proporciona utilidades para ejecutar, gestionar y reportar tests de Playwright de manera estructurada.

## Componentes

### `PlaywrightTestRunner`

Ejecuta tests y genera reportes:

```python
from playwright_test_runner import PlaywrightTestRunner

runner = PlaywrightTestRunner(output_dir="test_results")
suite_result = runner.run_tests(
    test_path="tests/test_playwright*.py",
    markers=["smoke"],
    verbose=True
)

# Guardar resultados
runner.save_results(suite_result, "results.json")

# Generar reporte HTML
runner.save_html_report(suite_result, "report.html")
```

### `PlaywrightTestFilter`

Filtra tests por criterios:

```python
from playwright_test_runner import PlaywrightTestFilter

# Filtrar por markers
marker_filter = PlaywrightTestFilter.filter_by_marker(["smoke", "api"])

# Filtrar por nombre
name_filter = PlaywrightTestFilter.filter_by_name("test_upload")

# Filtrar por archivos
file_filter = PlaywrightTestFilter.filter_by_file([
    "tests/test_playwright_api.py",
    "tests/test_playwright_ui.py"
])
```

### `PlaywrightTestExecutor`

Ejecutor con opciones predefinidas:

```python
from playwright_test_runner import PlaywrightTestExecutor

executor = PlaywrightTestExecutor()

# Ejecutar smoke tests
smoke_result = executor.run_smoke_tests()

# Ejecutar tests críticos
critical_result = executor.run_critical_tests()

# Ejecutar tests rápidos
fast_result = executor.run_fast_tests()

# Ejecutar todos los tests
all_result = executor.run_all_tests()

# Ejecutar con coverage
coverage_result = executor.run_with_coverage()

# Ejecutar en paralelo
parallel_result = executor.run_parallel(workers=4)
```

## Estructura de Resultados

### `TestResult`

```python
@dataclass
class TestResult:
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    trace_path: Optional[str] = None
```

### `TestSuiteResult`

```python
@dataclass
class TestSuiteResult:
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration: float
    results: List[TestResult]
    timestamp: float
```

## Reportes

### Reporte JSON

```python
runner.save_results(suite_result, "test_results.json")
```

Formato:
```json
{
  "suite_name": "Playwright Tests",
  "total_tests": 100,
  "passed": 95,
  "failed": 3,
  "skipped": 2,
  "duration": 45.67,
  "results": [...],
  "timestamp": 1234567890.0
}
```

### Reporte HTML

```python
runner.save_html_report(suite_result, "test_report.html")
```

El reporte HTML incluye:
- Resumen de la suite
- Estadísticas (total, passed, failed, skipped)
- Duración
- Tabla de resultados detallada

## Uso en CI/CD

### Ejemplo de Script

```python
#!/usr/bin/env python3
"""Script para ejecutar tests en CI/CD."""

from playwright_test_runner import PlaywrightTestExecutor
import sys

def main():
    executor = PlaywrightTestExecutor()
    
    # Ejecutar smoke tests primero
    print("Running smoke tests...")
    smoke_result = executor.run_smoke_tests()
    
    if smoke_result.failed > 0:
        print(f"Smoke tests failed: {smoke_result.failed}")
        sys.exit(1)
    
    # Ejecutar tests críticos
    print("Running critical tests...")
    critical_result = executor.run_critical_tests()
    
    if critical_result.failed > 0:
        print(f"Critical tests failed: {critical_result.failed}")
        sys.exit(1)
    
    # Ejecutar todos los tests
    print("Running all tests...")
    all_result = executor.run_all_tests()
    
    # Guardar reportes
    executor.runner.save_results(all_result, "ci_results.json")
    executor.runner.save_html_report(all_result, "ci_report.html")
    
    # Exit code basado en resultados
    sys.exit(0 if all_result.failed == 0 else 1)

if __name__ == "__main__":
    main()
```

## Integración con Pytest

El Test Runner puede integrarse con pytest usando hooks:

```python
# conftest.py
import pytest
from playwright_test_runner import PlaywrightTestRunner

runner = PlaywrightTestRunner()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test results."""
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call" and rep.failed:
        # Capturar screenshot
        if hasattr(item, "page"):
            screenshot_path = f"screenshots/{item.name}.png"
            item.page.screenshot(path=screenshot_path)
            rep.screenshot_path = screenshot_path
```

## Mejores Prácticas

1. **Usar markers**: Organizar tests con markers para fácil filtrado
2. **Guardar resultados**: Siempre guardar resultados para análisis posterior
3. **Generar reportes**: Generar reportes HTML para visualización
4. **Integrar en CI/CD**: Usar en pipelines para reportes automáticos
5. **Paralelización**: Usar ejecución paralela para tests grandes

## Extensión

Para agregar nuevas funcionalidades:

```python
class CustomTestRunner(PlaywrightTestRunner):
    """Custom test runner with additional features."""
    
    def generate_junit_xml(self, suite_result: TestSuiteResult) -> str:
        """Generate JUnit XML format."""
        # Implementation
        pass
    
    def send_to_slack(self, suite_result: TestSuiteResult):
        """Send results to Slack."""
        # Implementation
        pass
```



