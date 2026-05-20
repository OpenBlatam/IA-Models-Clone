# Guía de Debugging

## Introducción

Las utilidades de debugging ayudan a identificar y resolver problemas en los tests de Playwright.

## PlaywrightDebugger

### Uso Básico

```python
from playwright_debug import create_debugger

def test_with_debugging(page, api_base_url):
    debugger = create_debugger(page)
    
    # Hacer requests
    page.goto(api_base_url)
    page.request.get(f"{api_base_url}/health")
    
    # Capturar screenshot
    screenshot_path = debugger.capture_screenshot("test.png")
    
    # Capturar network log
    network_log = debugger.capture_network_log()
    print(f"Total requests: {network_log['total_requests']}")
    
    # Guardar toda la información de debug
    debug_file = debugger.save_debug_info("test_debug")
```

### Captura de Screenshots

```python
debugger = create_debugger(page)
screenshot_path = debugger.capture_screenshot("my_test.png")
# Screenshot guardado en debug_output/my_test.png
```

### Network Logs

```python
debugger = create_debugger(page)
page.request.get(f"{api_base_url}/endpoint")

network_log = debugger.capture_network_log()
print(f"Requests: {network_log['total_requests']}")
print(f"Responses: {network_log['total_responses']}")
print(f"Failed: {network_log['failed_requests']}")
```

### Console Logs

```python
debugger = create_debugger(page)
page.evaluate("console.log('Test'); console.error('Error')")

console_log = debugger.capture_console_log()
print(f"Total logs: {console_log['total_logs']}")
print(f"Errors: {console_log['errors']}")
```

### Análisis de Performance

```python
debugger = create_debugger(page)
page.goto(api_base_url)

performance = debugger.analyze_performance()
print(f"DOM Loading: {performance['metrics']['dom_loading']}ms")
print(f"Load Complete: {performance['metrics']['load_complete']}ms")
```

### Wait y Debug

```python
debugger = create_debugger(page)

# Esperar condición, capturar debug si falla
result = debugger.wait_and_debug(
    lambda: page.locator("#element").is_visible(),
    timeout=5000
)

if not result:
    # Screenshot automáticamente capturado en wait_failed.png
    pass
```

## PlaywrightTroubleshooter

### Diagnóstico de Timeouts

```python
from playwright_debug import troubleshoot_timeout

def test_with_timeout_diagnosis(page, api_base_url):
    page.goto(api_base_url)
    
    diagnosis = troubleshoot_timeout(page)
    
    if diagnosis["has_issues"]:
        print("Issues found:")
        for issue in diagnosis["issues"]:
            print(f"  - {issue}")
    
    print(f"Pending requests: {diagnosis['pending_requests']}")
    print(f"Console errors: {diagnosis['console_errors']}")
```

### Diagnóstico de Performance

```python
from playwright_debug import troubleshoot_performance

def test_with_performance_diagnosis(page, api_base_url):
    page.goto(api_base_url)
    
    diagnosis = troubleshoot_performance(page)
    
    if diagnosis["has_issues"]:
        print("Performance issues:")
        for issue in diagnosis["issues"]:
            print(f"  - {issue}")
    
    perf_data = diagnosis["performance_data"]
    print(f"DOM Loading: {perf_data['dom_loading']}ms")
    print(f"Load Complete: {perf_data['load_complete']}ms")
    print(f"Slow resources: {len(perf_data['slow_resources'])}")
```

## Integración con Tests

### Fixture de Debugger

```python
# conftest.py
@pytest.fixture
def debugger(page):
    from playwright_debug import create_debugger
    return create_debugger(page)

# test_file.py
def test_with_debugger(page, debugger, api_base_url):
    page.goto(api_base_url)
    
    # Usar debugger
    debugger.capture_screenshot("test.png")
    
    # Si el test falla, guardar debug info
    try:
        # ... test code ...
        pass
    except Exception:
        debugger.save_debug_info("failed_test")
        raise
```

### Auto-captura en Fallos

```python
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    
    if rep.when == "call" and rep.failed:
        if "debugger" in item.fixturenames:
            debugger = item.funcargs.get("debugger")
            if debugger:
                debugger.save_debug_info(item.name)
```

## Mejores Prácticas

1. **Usar debugger en tests problemáticos**: Solo cuando sea necesario
2. **Guardar debug info en fallos**: Automatizar la captura
3. **Analizar network logs**: Identificar requests problemáticos
4. **Revisar console logs**: Detectar errores JavaScript
5. **Comparar performance**: Usar análisis de performance
6. **Limpiar output**: Eliminar archivos de debug antiguos periódicamente

## Output Directory

Por defecto, los archivos de debug se guardan en `debug_output/`. Puedes cambiar esto:

```python
debugger = create_debugger(page, output_dir="custom_debug")
```

## Estructura de Debug Info

El archivo JSON de debug info contiene:

```json
{
  "test_name": "test_name",
  "timestamp": 1234567890.0,
  "url": "https://api.example.com",
  "title": "Page Title",
  "network": {
    "total_requests": 10,
    "total_responses": 10,
    "failed_requests": 0,
    "logs": [...]
  },
  "console": {
    "total_logs": 5,
    "errors": 0,
    "warnings": 1,
    "logs": [...]
  },
  "errors": [...],
  "screenshot": "path/to/screenshot.png"
}
```



