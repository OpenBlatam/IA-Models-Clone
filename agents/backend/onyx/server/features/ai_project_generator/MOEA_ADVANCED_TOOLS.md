# 🛠️ MOEA Advanced Tools - Herramientas Avanzadas

Guía completa de las herramientas avanzadas para el proyecto MOEA.

## 🆕 Nuevas Herramientas

### 1. **moea_monitor.py** - Monitor en Tiempo Real 📊

Monitor interactivo para observar el estado del sistema MOEA en tiempo real.

#### Características:
- ✅ Dashboard en tiempo real
- ✅ Actualización automática cada N segundos
- ✅ Muestra estadísticas del sistema
- ✅ Estado de la cola de proyectos
- ✅ Historial de estadísticas (opcional)
- ✅ Limpieza automática de pantalla

#### Uso:
```bash
# Monitor básico (actualiza cada 2 segundos)
python moea_monitor.py

# Intervalo personalizado
python moea_monitor.py --interval 5

# Guardar historial
python moea_monitor.py --save

# URL personalizada
python moea_monitor.py --url http://localhost:8001
```

#### Ejemplo de Salida:
```
======================================================================
            MOEA System Monitor - Dashboard en Tiempo Real
======================================================================
URL: http://localhost:8000 | Intervalo: 2.0s | Presiona Ctrl+C para salir
======================================================================

✅ Servidor: OPERATIVO

📊 Estadísticas del Sistema
   Proyectos procesados: 15
   En cola: 2
   Tiempo promedio: 45.23s

📋 Cola de Proyectos: 2
   1. [pending] A Multi-Objective Evolutionary Algorithm...
   2. [processing] Optimization problem with multiple...

======================================================================
Última actualización: 2025-01-01 12:34:56
```

---

### 2. **moea_export.py** - Exportador Avanzado 📦

Herramienta completa para exportar proyectos, resultados y configuraciones.

#### Características:
- ✅ Exportación de proyectos completos
- ✅ Múltiples formatos (ZIP, TAR, TAR.GZ)
- ✅ Exportación selectiva (con/sin resultados, config)
- ✅ Exportación solo de resultados
- ✅ Exportación de configuración del sistema

#### Uso:
```bash
# Exportar proyecto completo (ZIP)
python moea_export.py project_id

# Formato TAR comprimido
python moea_export.py project_id --format tar.gz

# Sin resultados
python moea_export.py project_id --no-results

# Sin configuración
python moea_export.py project_id --no-config

# Solo resultados
python moea_export.py project_id --results-only
```

#### Formatos Soportados:
- `zip` - Archivo ZIP estándar
- `tar` - Archivo TAR sin comprimir
- `tar.gz` - Archivo TAR comprimido con gzip

---

### 3. **moea_health.py** - Verificador de Salud 🏥

Verificador completo del estado del sistema MOEA.

#### Características:
- ✅ Verificación de múltiples endpoints
- ✅ Health checks automáticos
- ✅ Verificación de documentación
- ✅ Reporte detallado
- ✅ Salida en JSON (opcional)
- ✅ Códigos de salida para scripts

#### Uso:
```bash
# Health check básico
python moea_health.py

# Salida JSON
python moea_health.py --json

# URL personalizada
python moea_health.py --url http://localhost:8001
```

#### Checks Incluidos:
1. ✅ Health Check (`/health`)
2. ✅ API Stats (`/api/v1/stats`)
3. ✅ Queue Status (`/api/v1/queue`)
4. ✅ Swagger UI (`/docs`)
5. ✅ ReDoc (`/redoc`)
6. ✅ OpenAPI Schema (`/openapi.json`)

#### Ejemplo de Salida:
```
======================================================================
                    MOEA Health Check Report
======================================================================
Fecha: 2025-01-01 12:34:56
URL: http://localhost:8000
======================================================================

✅ PASS  Health Check                  Status: 200
✅ PASS  API Stats                     Status: 200
✅ PASS  Queue Status                  Status: 200
✅ PASS  Swagger UI                    Status: 200
✅ PASS  ReDoc                         Status: 200
✅ PASS  OpenAPI Schema                Status: 200

======================================================================
Resumen: 6/6 checks pasaron
✅ Sistema saludable
======================================================================
```

---

### 4. **moea_utils.py** - Utilidades Varias 🔧

Colección de funciones de utilidad y gestor de proyectos.

#### Funciones de Utilidad:
- ✅ Formateo de tiempo y tamaño
- ✅ Carga/guardado de JSON
- ✅ Verificación de servidor
- ✅ Validación de estructura
- ✅ Gestión de proyectos

#### Comandos CLI:

```bash
# Listar todos los proyectos
python moea_utils.py list

# Información de un proyecto
python moea_utils.py info moea_optimization_system

# Crear resumen de proyectos
python moea_utils.py summary --output mi_resumen.json
```

#### Funciones Disponibles:
```python
from moea_utils import (
    format_time,      # Formatear tiempo
    format_size,      # Formatear tamaño
    load_json_file,   # Cargar JSON
    save_json_file,   # Guardar JSON
    check_server_available,  # Verificar servidor
    list_projects,    # Listar proyectos
    find_project,     # Buscar proyecto
    MOEAProjectManager  # Gestor de proyectos
)
```

---

## 🔄 Integración con CLI Unificado

Todas las nuevas herramientas están integradas en `moea_cli.py`:

```bash
# Monitor
python moea_cli.py monitor

# Export
python moea_cli.py export PROJECT_ID --format zip

# Health check
python moea_cli.py health

# Utils
python moea_cli.py utils list
python moea_cli.py utils info PROJECT_NAME
```

---

## 📊 Casos de Uso

### Caso 1: Monitoreo Continuo
```bash
# Terminal 1: Servidor
cd backend && uvicorn app.main:app --reload

# Terminal 2: Monitor
python moea_monitor.py --interval 3 --save
```

### Caso 2: Backup de Proyectos
```bash
# Exportar todos los proyectos
for project in $(python moea_utils.py list | grep name); do
    python moea_export.py $project --format tar.gz
done
```

### Caso 3: Health Check Automatizado
```bash
# En script de CI/CD
python moea_health.py --json > health_report.json
if [ $? -eq 0 ]; then
    echo "Sistema saludable"
else
    echo "Sistema con problemas"
    exit 1
fi
```

### Caso 4: Resumen de Proyectos
```bash
# Crear resumen periódico
python moea_utils.py summary --output daily_summary_$(date +%Y%m%d).json
```

---

## 🎯 Flujo de Trabajo Completo

### 1. Generar y Configurar
```bash
python moea_cli.py generate
python moea_cli.py setup
```

### 2. Verificar Salud
```bash
python moea_health.py
```

### 3. Monitorear
```bash
python moea_monitor.py
```

### 4. Trabajar con Proyectos
```bash
# Listar
python moea_utils.py list

# Info
python moea_utils.py info moea_optimization_system

# Exportar
python moea_export.py project_id
```

### 5. Health Check Final
```bash
python moea_health.py --json
```

---

## 📈 Estadísticas de Herramientas

| Herramienta | Líneas | Funciones | Comandos |
|-------------|--------|-----------|----------|
| moea_monitor.py | ~150 | 8 | 3 |
| moea_export.py | ~200 | 6 | 7 |
| moea_health.py | ~120 | 7 | 3 |
| moea_utils.py | ~250 | 15 | 3 |
| **Total** | **~720** | **36** | **16** |

---

## 🔧 Requisitos

Todas las herramientas requieren:
```bash
pip install requests
```

Para `moea_monitor.py` (actualización de pantalla):
- Terminal que soporte códigos ANSI
- Python 3.8+

---

## 📚 Documentación Relacionada

- **[MOEA_TOOLS.md](MOEA_TOOLS.md)** - Herramientas básicas
- **[MOEA_IMPROVEMENTS.md](MOEA_IMPROVEMENTS.md)** - Mejoras implementadas
- **[MOEA_INDEX.md](MOEA_INDEX.md)** - Índice completo

---

## ✨ Próximas Mejoras

### Corto Plazo:
- [ ] Alertas automáticas en monitor
- [ ] Exportación incremental
- [ ] Health checks programados
- [ ] Métricas históricas

### Mediano Plazo:
- [ ] Dashboard web
- [ ] Notificaciones (email, Slack)
- [ ] Integración con Prometheus
- [ ] Auto-scaling basado en métricas

---

**¡Todas las herramientas avanzadas listas para usar! 🚀**

