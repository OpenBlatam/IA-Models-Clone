# Guía Completa - Sistema de Pipelines

## 📚 Índice

1. [Visión General](#visión-general)
2. [Funcionalidades Principales](#funcionalidades-principales)
3. [Uso Básico](#uso-básico)
4. [Monitoreo y Alertas](#monitoreo-y-alertas)
5. [CLI Tools](#cli-tools)
6. [Migración](#migración)
7. [API Reference](#api-reference)

## 🎯 Visión General

El sistema de pipelines proporciona:
- **Compatibilidad Legacy**: Wrapper para código existente
- **Sistema Modular**: Nuevo sistema modular y extensible
- **Monitoreo**: Sistema de monitoreo y alertas
- **Herramientas CLI**: Utilidades de línea de comandos
- **Migración**: Herramientas para migrar código legacy

## ✨ Funcionalidades Principales

### 1. Compatibilidad Legacy (`pipelines.py`)
- Wrapper de compatibilidad para código existente
- Validación de imports
- Health score y métricas
- Guía de migración integrada

### 2. Monitoreo (`pipelines_monitor.py`)
- Monitoreo automático de salud
- Sistema de alertas
- Historial de métricas
- Callbacks personalizados

### 3. Utilidades (`pipelines_utils.py`)
- Exportación de reportes
- Formateo de reportes
- Validación de salud

### 4. Migración (`pipelines_migration_helper.py`)
- Análisis de código legacy
- Generación de sugerencias
- Reportes de migración

### 5. CLI (`pipelines_cli.py`)
- Herramientas de línea de comandos
- Comandos para todas las funcionalidades

## 🚀 Uso Básico

### Verificar Compatibilidad

```python
from core.architecture.pipelines import check_compatibility

report = check_compatibility()
print(f"Health Score: {report['health_score']}")
print(f"Status: {report['status']}")
```

### Obtener Estadísticas

```python
from core.architecture.pipelines import get_import_statistics

stats = get_import_statistics()
print(f"Coverage: {stats['coverage_percentage']}%")
```

### Exportar Reporte

```python
from pathlib import Path
from core.architecture.pipelines_utils import export_compatibility_report

export_compatibility_report(Path("report.json"), format="json")
```

## 📊 Monitoreo y Alertas

### Monitoreo Básico

```python
from core.architecture.pipelines_monitor import get_monitor

monitor = get_monitor(check_interval=60.0, alert_threshold=0.8)
monitor.start_monitoring()

# Obtener salud actual
health = monitor.get_current_health()
print(f"Health Score: {health.health_score}")

# Obtener alertas activas
alerts = monitor.get_active_alerts()
for alert in alerts:
    print(f"{alert.level.value}: {alert.message}")
```

### Callbacks de Alertas

```python
def on_alert(alert):
    print(f"Alert: {alert.level.value} - {alert.message}")
    # Enviar notificación, log, etc.

monitor = get_monitor()
monitor.add_alert_callback(on_alert)
monitor.start_monitoring()
```

### Quick Health Check

```python
from core.architecture.pipelines_monitor import quick_health_check

result = quick_health_check()
print(result)
```

## 🛠️ CLI Tools

### Verificar Compatibilidad

```bash
# Formato texto
python -m core.architecture.pipelines_cli check

# Formato JSON
python -m core.architecture.pipelines_cli check -f json
```

### Monitoreo

```bash
# Quick check
python -m core.architecture.pipelines_cli monitor --quick

# Monitoreo continuo
python -m core.architecture.pipelines_cli monitor --start --interval 30
```

### Exportar Reportes

```bash
# JSON
python -m core.architecture.pipelines_cli export -o report.json -f json

# Markdown
python -m core.architecture.pipelines_cli export -o report.md -f md
```

### Estadísticas

```bash
# Básico
python -m core.architecture.pipelines_cli stats

# Detallado
python -m core.architecture.pipelines_cli stats --detailed

# JSON
python -m core.architecture.pipelines_cli stats -f json
```

### Análisis de Migración

```bash
# Analizar proyecto actual
python -m core.architecture.pipelines_cli migration

# Especificar directorio
python -m core.architecture.pipelines_cli migration -p /path/to/project

# Guardar reporte
python -m core.architecture.pipelines_cli migration -o migration_report.txt
```

## 🔄 Migración

### Análisis Automático

```python
from pathlib import Path
from core.architecture.pipelines_migration_helper import analyze_project_for_migration

report = analyze_project_for_migration(Path("."))
print(f"Files with legacy code: {report['files_with_legacy']}")
```

### Generar Reporte de Migración

```python
from pathlib import Path
from core.architecture.pipelines_migration_helper import generate_migration_report

report = generate_migration_report(Path("."), Path("migration_report.txt"))
print(report)
```

## 📖 API Reference

### `check_compatibility() -> Dict`
Retorna reporte de compatibilidad con health score y recomendaciones.

### `get_import_statistics() -> Dict`
Retorna estadísticas detalladas de imports por categoría.

### `get_import_status() -> Dict`
Retorna estado detallado de las importaciones.

### `validate_imports() -> bool`
Valida que todas las importaciones críticas estén disponibles.

### `get_migration_guide() -> Dict`
Retorna guía de migración con ejemplos.

### `PipelineMonitor`
Clase para monitoreo continuo con alertas.

### `quick_health_check() -> Dict`
Realiza un check rápido de salud.

## 🎨 Ejemplos Avanzados

### Monitoreo con Alertas Personalizadas

```python
from core.architecture.pipelines_monitor import get_monitor, AlertLevel

def custom_alert_handler(alert):
    if alert.level == AlertLevel.CRITICAL:
        # Enviar email, Slack, etc.
        send_critical_alert(alert)
    elif alert.level == AlertLevel.ERROR:
        # Log error
        logger.error(f"Pipeline error: {alert.message}")

monitor = get_monitor(alert_threshold=0.9)
monitor.add_alert_callback(custom_alert_handler)
monitor.start_monitoring()
```

### Análisis de Tendencias

```python
monitor = get_monitor()
trend = monitor.get_health_trend(minutes=60)

if trend:
    avg_health = sum(m.health_score for m in trend) / len(trend)
    print(f"Average health (last hour): {avg_health:.2f}")
```

### Integración con CI/CD

```python
from core.architecture.pipelines import check_compatibility
import sys

def ci_health_check():
    report = check_compatibility()
    
    if report['health_score'] < 0.8:
        print(f"Health score too low: {report['health_score']}")
        sys.exit(1)
    
    if report['status'] != 'ok':
        print(f"Status not OK: {report['status']}")
        sys.exit(1)
    
    print("Health check passed!")

if __name__ == "__main__":
    ci_health_check()
```

## 📝 Notas

- El wrapper legacy está deprecado pero seguirá funcionando
- Se recomienda migrar gradualmente al nuevo sistema
- El monitoreo puede ejecutarse en background
- Los reportes se pueden exportar en múltiples formatos

## 🔗 Recursos

- [Guía de Migración](PIPELINES_MIGRATION.md)
- [Documentación del Sistema Modular](pipelines/README.md)
- [Tests de Compatibilidad](../tests/test_architecture/test_pipelines_compatibility.py)

---

**Última actualización**: 2024

