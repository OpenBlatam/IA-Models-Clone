# 🧪 Sistema Completo de Testing - API BUL

## 📦 Suite Completa de Pruebas

### Scripts Disponibles

1. **test_api_responses.py** - Pruebas básicas mejoradas
2. **test_api_advanced.py** - Pruebas avanzadas con métricas
3. **test_security.py** - Pruebas de seguridad
4. **test_dashboard_generator.py** - Generador de dashboard HTML
5. **run_all_tests.bat/sh** - Ejecución automática de todas las pruebas

## 🎯 Tipos de Pruebas

### 1. Pruebas Básicas
- ✅ Validación de endpoints
- ✅ Verificación de respuestas
- ✅ Validación de campos
- ✅ Proceso completo de generación

### 2. Pruebas Avanzadas
- ✅ Pruebas de carga
- ✅ Requests concurrentes
- ✅ Rate limiting
- ✅ WebSocket
- ✅ Exportación de resultados

### 3. Pruebas de Seguridad
- ✅ SQL Injection
- ✅ XSS (Cross-Site Scripting)
- ✅ Validación de inputs
- ✅ Rate limiting como seguridad
- ✅ CORS
- ✅ Manejo de errores

### 4. Dashboard Visual
- ✅ Dashboard HTML interactivo
- ✅ Gráficos y métricas
- ✅ Lista de pruebas
- ✅ Errores detallados

## 🚀 Uso Rápido

### Ejecutar Todas las Pruebas

**Windows:**
```bash
run_all_tests.bat
```

**Linux/Mac:**
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

### Ejecutar Individualmente

```bash
# Pruebas básicas
python test_api_responses.py

# Pruebas avanzadas
python test_api_advanced.py

# Pruebas de seguridad
python test_security.py

# Generar dashboard
python test_dashboard_generator.py
```

## 📊 Resultados Generados

### Archivos de Salida

1. **test_results.json** - Resultados en JSON
2. **test_results.csv** - Resultados en CSV
3. **test_dashboard.html** - Dashboard HTML interactivo

### Dashboard HTML

El dashboard incluye:
- 📊 Estadísticas visuales
- 📋 Lista completa de pruebas
- ❌ Errores detallados
- 📈 Métricas de rendimiento
- 🎨 Diseño moderno y responsive

**Abrir dashboard:**
- Se abre automáticamente al ejecutar `run_all_tests`
- O abrir manualmente: `test_dashboard.html` en el navegador

## 🔒 Pruebas de Seguridad

### Vulnerabilidades Detectadas

El script de seguridad detecta:
- **SQL Injection**: Payloads maliciosos
- **XSS**: Scripts inyectados
- **Input Validation**: Validación de campos
- **Rate Limiting**: Protección contra abuso
- **CORS**: Configuración de seguridad
- **Error Handling**: Exposición de información

### Niveles de Severidad

- 🔴 **CRÍTICA**: Vulnerabilidad grave
- 🟡 **MEDIA**: Vulnerabilidad moderada
- 🟢 **BAJA**: Recomendación de mejora

## 📈 Métricas Generadas

### Pruebas Básicas
- Total de pruebas
- Exitosas vs Fallidas
- Tasa de éxito
- Tiempo total

### Pruebas Avanzadas
- **Carga:**
  - Total de requests
  - Tasa de éxito
  - Tiempo promedio/mínimo/máximo
  - RPS (Requests Per Second)

- **Concurrentes:**
  - Número de requests
  - Tasa de éxito
  - Tiempo total

### Pruebas de Seguridad
- Vulnerabilidades encontradas
- Por tipo y severidad
- Recomendaciones

## 🎨 Dashboard HTML

### Características

- ✅ Diseño moderno y responsive
- ✅ Gráficos visuales
- ✅ Barra de progreso
- ✅ Colores por estado
- ✅ Métricas detalladas
- ✅ Lista de errores
- ✅ Compatible con móviles

### Secciones

1. **Header** - Información general
2. **Stats Grid** - Tarjetas de estadísticas
3. **Progress Bar** - Barra de progreso visual
4. **Test List** - Lista de todas las pruebas
5. **Errors** - Errores encontrados
6. **Metrics** - Métricas detalladas

## 🔧 Configuración

### Modificar Timeouts

En los scripts:
```python
TIMEOUT = 60  # 60 segundos
```

### Modificar Pruebas de Carga

```python
test_load(
    duration_seconds=30,  # 30 segundos
    requests_per_second=5  # 5 RPS
)
```

### Personalizar Dashboard

Editar `test_dashboard_generator.py`:
- Colores
- Estilos
- Layout
- Métricas mostradas

## 📝 Integración CI/CD

### GitHub Actions

```yaml
name: Test API
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python api_frontend_ready.py &
      - run: sleep 5
      - run: python test_api_responses.py
      - run: python test_api_advanced.py
      - run: python test_security.py
      - uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: |
            test_results.json
            test_results.csv
            test_dashboard.html
```

## 🛠️ Troubleshooting

### Error: "No se puede conectar"
- Verificar que el servidor esté corriendo
- Verificar puerto (default: 8000)
- Verificar firewall

### Error: "ModuleNotFoundError"
```bash
pip install requests websockets colorama
```

### Dashboard no se genera
- Verificar que `test_results.json` existe
- Ejecutar pruebas primero
- Verificar permisos de escritura

## 📚 Documentación Adicional

- `GUIDE_TESTING_COMPLETE.md` - Guía completa
- `MEJORAS_SCRIPT_PRUEBAS.md` - Mejoras implementadas
- `README_QUE_GENERA.md` - Qué genera la API

## ✅ Checklist de Pruebas

Antes de un release:
- [ ] Ejecutar pruebas básicas
- [ ] Ejecutar pruebas avanzadas
- [ ] Ejecutar pruebas de seguridad
- [ ] Revisar dashboard HTML
- [ ] Verificar que no hay vulnerabilidades críticas
- [ ] Revisar métricas de rendimiento
- [ ] Exportar resultados para documentación

---

**Estado**: ✅ **Sistema completo de testing listo**
































