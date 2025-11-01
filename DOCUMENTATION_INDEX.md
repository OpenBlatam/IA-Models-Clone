# 📚 Índice de Documentación - Blatam Academy Features

## 📖 Documentación Principal

### Documentación General
- **[README.md](README.md)** - Documentación principal del sistema
  - Arquitectura general
  - Instalación y configuración
  - Uso básico
  - Troubleshooting

- **[SUMMARY.md](SUMMARY.md)** - Resumen ejecutivo
  - Estadísticas del proyecto
  - Métricas de rendimiento
  - Capacidades clave
  - Roadmap

- **[QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)** - Guía de inicio rápido
  - Setup en 5 minutos
  - Verificación rápida
  - Comandos esenciales

- **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** - Guía de arquitectura
  - Diagramas de arquitectura
  - Flujos de datos
  - Componentes principales
  - Patrones de diseño

## 🎯 Documentación por Módulo

### BUL (Business Unlimited)
- **[bulk/README.md](bulk/README.md)** - Documentación principal de BUL
  - Características principales
  - Uso básico
  - API endpoints
  - Ultra Adaptive KV Cache Engine

- **[bulk/ADVANCED_USAGE_GUIDE.md](bulk/ADVANCED_USAGE_GUIDE.md)** - Guía de uso avanzado
  - Optimización avanzada
  - Integración con otros sistemas
  - Patrones avanzados
  - Tuning de rendimiento

### Ultra Adaptive KV Cache Engine
- **[bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)** - Documentación completa del KV Cache
  - Características principales
  - Quick start
  - CLI usage
  - Troubleshooting

- **[bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md](bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md)** - Características completas
  - Módulos disponibles
  - Funcionalidades avanzadas
  - Configuración

### Otros Módulos
- **[content_redundancy_detector/README.md](content_redundancy_detector/README.md)** - Detector de redundancia
- **[business_agents/README.md](business_agents/README.md)** - Agentes de negocio
- **[export_ia/README.md](export_ia/README.md)** - Export IA
- **[integration_system/README.md](integration_system/README.md)** - Sistema de integración

## 🔧 Guías Técnicas

- **[BEST_PRACTICES.md](BEST_PRACTICES.md)** - Mejores prácticas completas
  - Optimización del KV Cache
  - Seguridad
  - Rendimiento
  - Desarrollo
  - Escalabilidad

- **[TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)** - Guía completa de troubleshooting
  - Problemas comunes
  - Problemas del KV Cache
  - Problemas de rendimiento
  - Diagnóstico avanzado

- **[bulk/USE_CASES.md](bulk/USE_CASES.md)** - Casos de uso reales
  - Generación masiva de documentos
  - Procesamiento en tiempo real
  - Automatización de flujos
  - Integración empresarial
  - Multi-tenant SaaS

- **[bulk/EXAMPLES.md](bulk/EXAMPLES.md)** - Ejemplos prácticos completos
  - Ejemplos básicos (7 ejemplos)
  - Ejemplos intermedios (7 ejemplos)
  - Ejemplos avanzados (7 ejemplos)
  - Ejemplos de integración (3 ejemplos)
  - Ejemplos de producción (5 ejemplos)
  - Total: 20+ ejemplos de código

### Desarrollo
- **[README.md - Sección Desarrollo](README.md#-desarrollo)** - Guía de desarrollo
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guía completa de contribución
- **Testing**: Ver sección Testing en README principal

### Deployment
- **[README.md - Sección Despliegue](README.md#-despliegue)** - Guía de despliegue
- **Producción**: Configuración de producción
- **Docker**: Uso de Docker Compose

### Monitoreo
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **KV Cache Metrics**: Ver [bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md#monitoring--observability)

## 🛠️ Referencia de API

- **[API_REFERENCE.md](API_REFERENCE.md)** - Referencia completa de API
  - Endpoints principales
  - Schemas
  - Autenticación
  - Métricas
  - Clientes Python

### API Principal
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Servicios Individuales
- **BUL API**: http://localhost:8002/docs
- **Content Redundancy**: http://localhost:8001/docs
- **Business Agents**: http://localhost:8004/docs
- **Export IA**: http://localhost:8005/docs

## 📊 Guías por Tarea

### Para Empezar
1. [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md) - Setup inicial
2. [README.md - Instalación](README.md#-instalación-y-configuración) - Instalación completa
3. [bulk/README.md](bulk/README.md) - Uso de BUL

### Para Desarrollo
1. [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - Entender arquitectura
2. [bulk/ADVANCED_USAGE_GUIDE.md](bulk/ADVANCED_USAGE_GUIDE.md) - Uso avanzado
3. [README.md - Desarrollo](README.md#-desarrollo) - Guía de desarrollo

### Para Optimización
1. [bulk/ADVANCED_USAGE_GUIDE.md#optimización-avanzada](bulk/ADVANCED_USAGE_GUIDE.md#-optimización-avanzada-del-kv-cache)
2. [bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md#-configuration)
3. [README.md - Performance](README.md#-performance)

### Para Troubleshooting
1. [README.md - Troubleshooting](README.md#-troubleshooting) - Problemas comunes
2. [bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md#troubleshooting](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md#-troubleshooting)
3. [QUICK_START_GUIDE.md#troubleshooting-rápido](QUICK_START_GUIDE.md#-troubleshooting-rápido)

## 🔗 Recursos Adicionales

### Documentación Externa
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Redis Documentation](https://redis.io/docs/)
- [Docker Documentation](https://docs.docker.com/)

### Herramientas CLI
- `python start_system.py --help` - Sistema principal
- `python bulk/core/ultra_adaptive_kv_cache_cli.py --help` - KV Cache CLI

## 📝 Estructura de Documentación

```
features/
├── README.md                          # Documentación principal
├── QUICK_START_GUIDE.md              # Inicio rápido
├── ARCHITECTURE_GUIDE.md             # Arquitectura
├── DOCUMENTATION_INDEX.md            # Este archivo
│
├── bulk/
│   ├── README.md                     # BUL principal
│   ├── ADVANCED_USAGE_GUIDE.md       # Uso avanzado
│   └── core/
│       ├── README_ULTRA_ADAPTIVE_KV_CACHE.md
│       └── ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md
│
└── [otros módulos]/
    └── README.md                     # Documentación individual
```

## 🎓 Rutas de Aprendizaje

### Principiante
1. Leer [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md)
2. Seguir [README.md](README.md) - Instalación
3. Probar ejemplos básicos

### Intermedio
1. Leer [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
2. Explorar [bulk/README.md](bulk/README.md)
3. Experimentar con KV Cache

### Avanzado
1. Estudiar [bulk/ADVANCED_USAGE_GUIDE.md](bulk/ADVANCED_USAGE_GUIDE.md)
2. Revisar [bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md](bulk/core/ULTRA_ADAPTIVE_KV_CACHE_COMPLETE_FEATURES.md)
3. Optimizar y customizar

## 🔍 Búsqueda Rápida

### Por Tema
- **Instalación**: [QUICK_START_GUIDE.md](QUICK_START_GUIDE.md), [README.md](README.md)
- **Arquitectura**: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
- **KV Cache**: [bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md](bulk/core/README_ULTRA_ADAPTIVE_KV_CACHE.md)
- **Optimización**: [bulk/ADVANCED_USAGE_GUIDE.md](bulk/ADVANCED_USAGE_GUIDE.md)
- **Troubleshooting**: [README.md - Troubleshooting](README.md#-troubleshooting)
- **API**: http://localhost:8000/docs

### Por Módulo
- **BUL**: [bulk/README.md](bulk/README.md)
- **Integration System**: [integration_system/README.md](integration_system/README.md)
- **Business Agents**: [business_agents/README.md](business_agents/README.md)
- **Export IA**: [export_ia/README.md](export_ia/README.md)

---

## 📝 Documentos Adicionales

- **[CHANGELOG.md](CHANGELOG.md)** - Historial de cambios
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guía de contribución
- **[API_REFERENCE.md](API_REFERENCE.md)** - Referencia completa de API
- **[PERFORMANCE_TUNING.md](PERFORMANCE_TUNING.md)** - Guía de tuning de rendimiento
- **[SECURITY_GUIDE.md](SECURITY_GUIDE.md)** - Guía completa de seguridad
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Checklist de despliegue
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guía de migración entre versiones
- **[bulk/QUICK_REFERENCE.md](bulk/QUICK_REFERENCE.md)** - Referencia rápida BUL

## 🛠️ Documentación de Desarrollo (KV Cache)

- **[bulk/core/DEVELOPMENT_GUIDE.md](bulk/core/DEVELOPMENT_GUIDE.md)** - Guía completa de desarrollo del KV Cache
  - Arquitectura del código
  - Extender el KV Cache
  - Crear nuevas estrategias
  - Debugging y profiling
  
- **[bulk/core/TESTING_GUIDE.md](bulk/core/TESTING_GUIDE.md)** - Guía completa de testing
  - Setup de testing
  - Tests unitarios
  - Tests de integración
  - Tests de performance
  - CI/CD integration
  
- **[bulk/core/API_REFERENCE_COMPLETE.md](bulk/core/API_REFERENCE_COMPLETE.md)** - Referencia completa de API del KV Cache
  - Todas las clases y métodos
  - Parámetros y retornos
  - Ejemplos de uso
  - Advanced features

## 📊 Recursos Visuales y Comparativos

- **[DIAGRAMS.md](DIAGRAMS.md)** - Diagramas visuales del sistema
  - Arquitectura completa
  - Flujos de datos
  - Estrategias de cache
  - Sistemas de monitoreo
  - Escalabilidad
  
- **[FAQ.md](FAQ.md)** - Preguntas frecuentes
  - General
  - KV Cache Engine
  - Configuración
  - Rendimiento
  - Troubleshooting
  - Deployment
  - Desarrollo
  
- **[ROADMAP.md](ROADMAP.md)** - Roadmap del proyecto
  - Versiones futuras
  - Objetivos a largo plazo
  - Prioridades
  - Contribuciones deseadas
  
- **[bulk/COMPARISON.md](bulk/COMPARISON.md)** - Comparación de estrategias BUL
  - LRU vs LFU vs Adaptive
  - Configuraciones preset
  - Modos de operación
  - Técnicas de optimización
  - Casos de uso
  
- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Guía de integración
  - FastAPI, Celery, Django, Flask
  - Redis, PostgreSQL
  - Prometheus, Grafana
  - REST, WebSocket, gRPC
  
- **[bulk/ADVANCED_TROUBLESHOOTING.md](bulk/ADVANCED_TROUBLESHOOTING.md)** - Troubleshooting avanzado BUL
  - Problemas de rendimiento
  - Problemas de memoria
  - Problemas de GPU
  - Problemas de cache
  - Debugging avanzado
  
- **[bulk/PRODUCTION_READY.md](bulk/PRODUCTION_READY.md)** - Guía de producción BUL
  - Checklist de producción
  - Configuración óptima
  - Seguridad
  - Monitoreo
  - Backup y disaster recovery
  - Incident response
  
- **[BENCHMARKING_GUIDE.md](BENCHMARKING_GUIDE.md)** - Guía completa de benchmarking
  - Setup de benchmarking
  - Benchmarks de latencia, throughput, memoria
  - Benchmarks de cache
  - Análisis y visualización de resultados
  
- **[OPTIMIZATION_STRATEGIES.md](OPTIMIZATION_STRATEGIES.md)** - Estrategias de optimización avanzada
  - Optimización de cache (warming, partitioning, prefetching)
  - Optimización de memoria (pools, compresión adaptativa)
  - Optimización de GPU (mixed precision, memory management)
  - Optimización de red y base de datos
  - Optimización combinada

## 📚 Referencias y Recursos

- **[TROUBLESHOOTING_QUICK_REFERENCE.md](TROUBLESHOOTING_QUICK_REFERENCE.md)** - Referencia rápida de troubleshooting
  - Problemas comunes y soluciones rápidas
  - Comandos de diagnóstico
  - Presets de configuración
  - Checklist de verificación
  
- **[GLOSSARY.md](GLOSSARY.md)** - Glosario completo de términos
  - Definiciones de conceptos clave
  - Términos técnicos
  - Acrónimos y abreviaciones
  
- **[CHANGELOG_DETAILED.md](CHANGELOG_DETAILED.md)** - Changelog detallado
  - Historial completo de versiones
  - Features por categoría
  - Estadísticas de desarrollo
  - Roadmap de versiones
  
- **[COMMANDS_CHEATSHEET.md](COMMANDS_CHEATSHEET.md)** - Cheatsheet de comandos
  - Docker commands
  - Python/KV Cache CLI
  - Configuración
  - Monitoreo y debugging
  - Backup y restore
  - Deployment
  
- **[EXAMPLES_COOKBOOK.md](EXAMPLES_COOKBOOK.md)** - Cookbook de ejemplos
  - Ejemplos básicos, intermedios y avanzados
  - Patrones comunes
  - Recetas de integración
  
- **[ANTI_PATTERNS.md](ANTI_PATTERNS.md)** - Anti-patrones y qué evitar
  - Anti-patrones de cache y configuración
  - Anti-patrones de performance y seguridad
  - Anti-patrones de código
  - Mejores prácticas
  
- **[SECURITY_CHECKLIST.md](SECURITY_CHECKLIST.md)** - Checklist de seguridad
  - Pre-deployment security
  - Production hardening
  - Security audit
  - Incident response
  
- **[PERFORMANCE_CHECKLIST.md](PERFORMANCE_CHECKLIST.md)** - Checklist de rendimiento
  - Pre-deployment performance
  - Métricas a monitorear
  - Performance tuning
  - Targets y objetivos
  
- **[TROUBLESHOOTING_BY_SYMPTOM.md](TROUBLESHOOTING_BY_SYMPTOM.md)** - Troubleshooting por síntomas
  - Diagnóstico por síntomas comunes
  - Soluciones rápidas
  - Flujo de diagnóstico
  - Cuando buscar más ayuda
  
- **[QUICK_SETUP_GUIDES.md](QUICK_SETUP_GUIDES.md)** - Guías de setup rápido
  - Setup por caso de uso (desarrollo, producción, testing, etc.)
  - Presets de configuración
  - Scripts de setup automático
  - Verificación post-setup
  
- **[QUICK_WINS.md](QUICK_WINS.md)** - Quick Wins de rendimiento
  - Mejoras en 5 minutos
  - Mejoras en 15 minutos
  - Mejoras en 30 minutos
  - Impacto esperado por mejora
  - Combinaciones de quick wins
  - Scripts de aplicación automática
  
- **[BEST_PRACTICES_SUMMARY.md](BEST_PRACTICES_SUMMARY.md)** - Resumen de mejores prácticas
  - Principios fundamentales
  - Checklist por categoría
  - Mejores prácticas por escenario
  - Patrones recomendados
  - Qué NO hacer (resumen)
  - Métricas de éxito
  
- **[COMMON_WORKFLOWS.md](COMMON_WORKFLOWS.md)** - Workflows comunes
  - Desarrollo diario
  - Deployment
  - Troubleshooting
  - Optimización
  - Mantenimiento
  - Monitoreo
  
- **[DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md)** - Mapa de documentación
  - Navegación por nivel de experiencia
  - Navegación por tarea
  - Navegación por componente
  - Rutas de aprendizaje
  - Matriz de documentación
  - Enlaces rápidos por situación
  
- **[CONFIGURATION_DECISION_TREE.md](CONFIGURATION_DECISION_TREE.md)** - Árbol de decisión de configuración
  - Decision tree visual
  - Presets por caso de uso
  - Configuración por prioridad
  - Configuración por recurso
  - Matriz de decisión rápida
  
- **[COST_OPTIMIZATION.md](COST_OPTIMIZATION.md)** - Optimización de costos
  - Análisis de componentes de costo
  - Estrategias de reducción
  - ROI de optimizaciones
  - Presupuesto por escenario
  - Monitoring de costos
  
- **[SCALING_GUIDE.md](SCALING_GUIDE.md)** - Guía de escalabilidad
  - Escalado vertical vs horizontal
  - Planificación por fase
  - Auto-scaling inteligente
  - Escalado multi-región
  - Monitoreo de escalado
  - Estrategias por escenario
  
- **[ERROR_CODES_REFERENCE.md](ERROR_CODES_REFERENCE.md)** - Referencia de códigos de error
  - Errores KV Cache Engine (E001-E005)
  - Errores del Sistema (S001-S004)
  - Errores de API (A001-A003)
  - Búsqueda rápida por error
  - Herramientas de diagnóstico automático
  - Tabla de referencia rápida
  
- **[QUICK_DIAGNOSTICS.md](QUICK_DIAGNOSTICS.md)** - Diagnóstico rápido
  - Health check completo
  - Performance snapshot
  - Configuration validator
  - Resource usage check
  - Diagnóstico por categoría
  - Checklists de diagnóstico
  
- **[API_VERSIONING.md](API_VERSIONING.md)** - Versionado de API
  - Estrategias de versionado (URL, Header, Query)
  - Implementación recomendada
  - Semantic versioning
  - Migración entre versiones
  - Deprecation policy
  - Changelog de API
  
- **[GETTING_HELP.md](GETTING_HELP.md)** - Cómo obtener ayuda
  - Ruta de ayuda por situación
  - Cómo buscar en documentación
  - Recursos por nivel
  - Cuando necesitas más ayuda
  - Recopilar información para soporte
  
- **[DISASTER_RECOVERY.md](DISASTER_RECOVERY.md)** - Disaster Recovery
  - Plan de recuperación ante desastres
  - Estrategias de backup
  - Procedimientos de restore
  - Backup encriptado
  - RTO y RPO objetivos
  - Test de disaster recovery
  
- **[ONBOARDING_GUIDE.md](ONBOARDING_GUIDE.md)** - Guía de onboarding
  - Primer día (30 min)
  - Primera semana
  - Primer mes
  - Checklist de onboarding
  - Recursos por rol
  - Objetivos de aprendizaje
  - Notas para mentores
  
- **[OPERATIONAL_RUNBOOK.md](OPERATIONAL_RUNBOOK.md)** - Runbook operacional
  - Incidentes críticos y procedimientos
  - Operaciones diarias (morning/evening routine)
  - Mantenimiento regular (diario, semanal, mensual)
  - Escalación
  - Métricas de monitoreo
  
- **[QUICK_LINKS.md](QUICK_LINKS.md)** - Quick Links
  - Links más usados
  - Por situación común
  - Búsqueda rápida
  
- **[CONFIGURATION_RECIPES.md](CONFIGURATION_RECIPES.md)** - Recetas de configuración
  - Configuraciones listas para usar (8+ recetas)
  - Recetas de integración (FastAPI, Django, Celery)
  - Recetas de .env
  - Recetas por objetivo
  
- **[ALERTING_CONFIGURATION.md](ALERTING_CONFIGURATION.md)** - Configuración de alertas
  - Alertas críticas (severidad alta)
  - Alertas importantes (severidad media)
  - Alertas informativas (severidad baja)
  - Configuración Prometheus
  - Notificaciones (Slack, Email, PagerDuty)
  - Dashboard de alertas
  
- **[LOGGING_BEST_PRACTICES.md](LOGGING_BEST_PRACTICES.md)** - Mejores prácticas de logging
  - Estructura de logs
  - Logging estructurado (JSON)
  - Niveles de logging
  - Logging seguro (redacción)
  - Logging de auditoría
  - Logging de performance
  - Configuración por entorno
  - ELK Stack integration
  
- **[CI_CD_SETUP.md](CI_CD_SETUP.md)** - Setup de CI/CD
  - GitHub Actions workflow
  - Pre-commit hooks
  - Dockerfile para CI/CD
  - Pipeline completo (lint, test, build, deploy)
  - Checklist de CI/CD
  
- **[ADVANCED_DEBUGGING.md](ADVANCED_DEBUGGING.md)** - Debugging avanzado
  - Estrategias de debugging
  - Debugging interactivo (IPython)
  - Debugging asíncrono
  - Memory leak debugging
  - GPU memory debugging
  - Distributed systems debugging
  - Network debugging
  - Performance debugging
  
- **[PROFILING_GUIDE.md](PROFILING_GUIDE.md)** - Guía de profiling
  - Tipos de profiling (CPU, memory, I/O, GPU)
  - Herramientas (py-spy, cProfile, line_profiler)
  - Memory profiling avanzado
  - GPU profiling (CUDA)
  - KV Cache profiling
  - Análisis de resultados
  - Flame graphs
  
- **[NETWORKING_GUIDE.md](NETWORKING_GUIDE.md)** - Guía de networking
  - Configuración de red Docker
  - Network policies (Kubernetes)
  - Comunicación REST/gRPC/WebSocket
  - Security (TLS/SSL)
  - Load balancing
  - Network monitoring
  - Health checks
  
- **[E2E_TESTING.md](E2E_TESTING.md)** - Testing end-to-end
  - Estrategia de E2E testing
  - Setup y configuración
  - Tests de flujos completos
  - Tests de integración
  - Performance testing E2E
  - Stress testing
  
- **[ERROR_HANDLING_PATTERNS.md](ERROR_HANDLING_PATTERNS.md)** - Patrones de error handling
  - Principios de error handling
  - Patrones de retry (exponential backoff)
  - Circuit breaker pattern
  - Error wrapping y jerarquía
  - Error logging estructurado
  - Graceful degradation
  - Partial success handling
  
- **[SECURITY_AUDITING.md](SECURITY_AUDITING.md)** - Auditoría de seguridad
  - Checklist completo de seguridad
  - Auditoría automatizada
  - Security scanning tools
  - Security testing
  - Security metrics y monitoring
  - Checklist por componente

## 🛠️ Scripts y Utilidades

- **[scripts/setup_complete.sh](scripts/setup_complete.sh)** - Script de setup completo
- **[scripts/health_check.sh](scripts/health_check.sh)** - Health check del sistema
- **[scripts/benchmark.sh](scripts/benchmark.sh)** - Script de benchmarking

## 📋 Plantillas de Configuración

- **[config/templates/production.env.template](config/templates/production.env.template)** - Template .env para producción
- **[config/templates/kv_cache_production.yaml](config/templates/kv_cache_production.yaml)** - Config KV Cache producción

---

**Última actualización**: Documentación completa del ecosistema Blatam Academy Features

**Estadísticas de Documentación:**
- ✅ 40+ módulos con README
- ✅ 10+ guías técnicas completas
- ✅ 20+ ejemplos prácticos
- ✅ 5+ casos de uso reales
- ✅ Documentación completa de KV Cache
- ✅ Referencia completa de API

Para sugerencias o mejoras en la documentación, por favor crear un issue o consultar [CONTRIBUTING.md](CONTRIBUTING.md).

