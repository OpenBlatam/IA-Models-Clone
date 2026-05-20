# Guía de Tests con Playwright

## Introducción

Playwright es una herramienta de automatización de navegadores que permite probar aplicaciones web de manera realista, simulando interacciones de usuarios reales.

## Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Instalar navegadores de Playwright

```bash
playwright install
```

O instalar solo Chromium:

```bash
playwright install chromium
```

## Estructura de Tests

### `test_playwright.py`
Tests fundamentales de Playwright:
- Requests HTTP a la API
- Navegación del navegador
- Envío de formularios
- Documentación de API
- Manejo de errores
- Performance
- Screenshots
- Monitoreo de red

### `test_playwright_scenarios.py`
Escenarios reales con Playwright:
- Journey completo de usuario
- Interacción con documentación
- Escenarios de error
- Monitoreo de performance
- Headers de seguridad

## Ejecutar Tests de Playwright

### Ejecutar todos los tests de Playwright

```bash
# Todos los tests
pytest tests/test_playwright.py tests/test_playwright_scenarios.py -v

# Solo tests básicos
pytest tests/test_playwright.py -v

# Solo tests de escenarios
pytest tests/test_playwright_scenarios.py -v

# Con marcador
pytest -m playwright -v
```

### Opciones de ejecución

```bash
# Ejecutar con navegador visible (no headless)
pytest tests/test_playwright.py --headless=false

# Especificar navegador
pytest tests/test_playwright.py --browser=firefox

# Especificar URL de API
pytest tests/test_playwright.py --api-url=http://localhost:8000

# Ejecutar con video
pytest tests/test_playwright.py --record-video
```

## Tipos de Tests

### 1. Tests de API Requests

Prueban la API usando requests HTTP de Playwright:

```python
def test_health_check_via_playwright(page, api_base_url):
    response = page.request.get(f"{api_base_url}/health")
    assert response.status == 200
```

### 2. Tests de Navegación

Prueban la navegación del navegador:

```python
def test_navigate_to_api_base(page, api_base_url):
    page.goto(api_base_url)
    assert page.url.startswith(api_base_url)
```

### 3. Tests de Documentación

Prueban la documentación de la API (Swagger/OpenAPI):

```python
def test_swagger_ui_loads(page, api_base_url):
    page.goto(f"{api_base_url}/docs")
    # Verificar elementos de Swagger
```

### 4. Tests de Performance

Miden tiempos de respuesta:

```python
def test_api_response_time(page, api_base_url):
    start = time.time()
    response = page.request.get(f"{api_base_url}/health")
    elapsed = time.time() - start
    assert elapsed < 1.0
```

### 5. Tests de Screenshots

Capturan screenshots de páginas:

```python
def test_capture_screenshot(page, api_base_url, tmp_path):
    page.goto(api_base_url)
    page.screenshot(path="screenshot.png")
```

## Configuración

### Variables de Entorno

```bash
export API_URL=http://localhost:8000
export HEADLESS=true
export BROWSER=chromium
```

### Opciones de Pytest

```bash
# En pytest.ini o conflaguración
[pytest]
addopts = 
    --api-url=http://localhost:8000
    --headless=true
    --browser=chromium
```

## Fixtures Disponibles

### `browser`
Instancia del navegador (sesión completa)

### `context`
Contexto del navegador (nuevo para cada test)

### `page`
Página del navegador (nuevo para cada test)

### `api_base_url`
URL base de la API (configurable)

## Mejores Prácticas

### 1. Timeouts Apropiados

```python
page.set_default_timeout(30000)  # 30 segundos
```

### 2. Esperar Elementos

```python
page.wait_for_selector("selector", timeout=5000)
page.wait_for_load_state("networkidle")
```

### 3. Manejo de Errores

```python
try:
    page.goto(url, timeout=5000)
except Exception:
    pytest.skip("Page not available")
```

### 4. Limpieza

Los fixtures se encargan de cerrar páginas y contextos automáticamente.

## Debugging

### Ejecutar en Modo Debug

```bash
# Con navegador visible
pytest tests/test_playwright.py --headless=false -s

# Con pausa en errores
pytest tests/test_playwright.py --pdb
```

### Screenshots en Fallos

Playwright automáticamente captura screenshots cuando los tests fallan.

### Videos

```bash
# Grabar videos de tests
pytest tests/test_playwright.py --record-video
```

## CI/CD

### GitHub Actions

```yaml
- name: Install Playwright
  run: |
    pip install playwright
    playwright install chromium

- name: Run Playwright tests
  run: pytest tests/test_playwright.py -m playwright
```

### Docker

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.40.0

RUN pip install -r requirements.txt
RUN playwright install chromium
```

## Troubleshooting

### Tests Fallan por Timeout

Aumentar timeout:

```python
page.set_default_timeout(60000)  # 60 segundos
```

### Navegador No Se Inicia

Verificar instalación:

```bash
playwright install --help
playwright install chromium
```

### API No Disponible

Verificar que la API esté corriendo:

```bash
curl http://localhost:8000/health
```

## Ventajas de Playwright

1. **Multi-navegador**: Chromium, Firefox, WebKit
2. **Auto-wait**: Espera automática de elementos
3. **Network Control**: Control de requests/responses
4. **Screenshots/Videos**: Captura automática
5. **Mobile**: Soporte para dispositivos móviles
6. **API Testing**: Puede hacer requests HTTP directamente

## Próximos Pasos

1. Agregar más tests de interacción con UI
2. Tests de accesibilidad
3. Tests de responsive design
4. Tests de diferentes navegadores
5. Tests de dispositivos móviles



