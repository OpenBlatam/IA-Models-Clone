# ⚡ MOEA Quick Reference - Referencia Rápida

Guía de referencia rápida con todos los comandos y atajos del sistema MOEA.

## 🚀 Comandos Rápidos

### Generación
```bash
# Generación rápida
python quick_moea.py
# o
moea_quick_commands.bat gen        # Windows
source moea_quick_commands.sh && moea-gen  # Linux/Mac
```

### Configuración
```bash
python moea_setup.py
# o
moea_quick_commands.bat setup
```

### Testing
```bash
python moea_test_api.py
# o
moea_quick_commands.bat test
```

### Monitoreo
```bash
# CLI
python moea_monitor.py

# Web Dashboard
python moea_dashboard.py

# o
moea_quick_commands.bat monitor
moea_quick_commands.bat dash
```

## 📋 Comandos por Categoría

### Generación
| Comando | Descripción |
|---------|-------------|
| `python quick_moea.py` | Generación rápida |
| `python generate_moea.py` | Generación vía API |
| `python moea_wrapper.py generate` | CLI unificado |

### Configuración
| Comando | Descripción |
|---------|-------------|
| `python moea_setup.py` | Setup automático |
| `python verify_moea_project.py` | Verificar estructura |
| `python moea_config.py --show` | Ver configuración |

### Testing
| Comando | Descripción |
|---------|-------------|
| `python moea_test_api.py` | Suite de tests |
| `python moea_health.py` | Health check |
| `python moea_benchmark.py` | Benchmarking |

### Monitoreo
| Comando | Descripción |
|---------|-------------|
| `python moea_monitor.py` | Monitor CLI |
| `python moea_dashboard.py` | Dashboard web |
| `python moea_metrics.py --continuous 60` | Recolección continua |

### Análisis
| Comando | Descripción |
|---------|-------------|
| `python moea_analytics.py --report report.json` | Análisis completo |
| `python moea_performance.py --report perf.json` | Performance analysis |
| `python moea_security.py project_dir` | Auditoría de seguridad |

### Mantenimiento
| Comando | Descripción |
|---------|-------------|
| `python moea_backup.py create project_dir` | Crear backup |
| `python moea_cleanup.py --all` | Limpieza completa |
| `python moea_scheduler.py run` | Ejecutar scheduler |

### Utilidades
| Comando | Descripción |
|---------|-------------|
| `python moea_utils.py list` | Listar proyectos |
| `python moea_docs.py project_dir` | Generar documentación |
| `python moea_export.py project_id` | Exportar proyecto |

### Integración
| Comando | Descripción |
|---------|-------------|
| `python moea_integration.py prometheus` | Exportar a Prometheus |
| `python moea_integration.py grafana` | Dashboard Grafana |
| `python moea_integration.py ci` | GitHub Actions CI |

### Asistente IA
| Comando | Descripción |
|---------|-------------|
| `python moea_ai_assistant.py` | Ver ayuda |
| `python moea_ai_assistant.py --interactive` | Modo interactivo |
| `python moea_ai_assistant.py "generar proyecto"` | Consulta específica |

## 🎯 Comandos Compuestos

### Setup Completo
```bash
# Windows
moea_quick_commands.bat full-setup

# Linux/Mac
moea-full-setup
```

### Test Suite Completo
```bash
# Windows
moea_quick_commands.bat full-test

# Linux/Mac
moea-full-test
```

### Tareas Diarias
```bash
# Windows
moea_quick_commands.bat daily

# Linux/Mac
moea-daily
```

## 🔧 Atajos de Teclado

### En el Wrapper
```bash
python moea_wrapper.py [comando]
```

Comandos disponibles:
- `generate` - Generar proyecto
- `setup` - Configurar
- `test` - Probar
- `monitor` - Monitorear
- `dashboard` - Dashboard web
- `health` - Health check
- `backup` - Backup
- `analytics` - Análisis
- `security` - Seguridad
- `performance` - Performance

## 📊 Flujos Comunes

### Desarrollo Rápido
```bash
python quick_moea.py
python moea_setup.py
python moea_dashboard.py
```

### Producción
```bash
python moea_health.py
python moea_performance.py --report prod.json
python moea_backup.py create project_dir
```

### Troubleshooting
```bash
python moea_ai_assistant.py "servidor no responde"
python moea_health.py
python moea_security.py project_dir
```

## 🆘 Ayuda Rápida

### Ver ayuda de cualquier herramienta
```bash
python [herramienta].py --help
python moea_wrapper.py help
python moea_ai_assistant.py
```

### Asistente Interactivo
```bash
python moea_ai_assistant.py --interactive
```

## 📝 Notas

- Todos los comandos pueden usar `--help` para ver opciones
- El wrapper unifica todos los comandos: `python moea_wrapper.py [comando]`
- Los scripts de comandos rápidos están en `moea_quick_commands.*`
- El asistente IA puede ayudar con cualquier duda

---

**¡Referencia rápida lista para usar! ⚡**

