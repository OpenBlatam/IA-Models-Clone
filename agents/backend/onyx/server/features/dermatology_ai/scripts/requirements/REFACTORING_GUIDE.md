# Refactoring Guide - Requirements Scripts

## 📁 Nueva Estructura Organizada

Los scripts de requirements han sido reorganizados en una estructura modular:

```
scripts/
├── requirements/
│   ├── __init__.py
│   ├── common.sh                    # Funciones comunes
│   ├── REFACTORING_GUIDE.md         # Esta guía
│   │
│   ├── analysis/                    # Scripts de análisis
│   │   ├── analyze-dependencies.py
│   │   ├── visualize-dependencies.py
│   │   ├── compare-requirements.sh
│   │   ├── requirements-diff.py
│   │   ├── requirements-stats.sh
│   │   ├── requirements-health-score.py
│   │   ├── dependency-tree.py
│   │   ├── dependency-dashboard.py
│   │   ├── requirements-size-analysis.sh
│   │   └── requirements-dependency-conflicts.py
│   │
│   ├── validation/                  # Scripts de validación
│   │   ├── validate-requirements.py
│   │   ├── all-checks.sh
│   │   ├── check-dependencies.sh
│   │   ├── requirements-quick-scan.sh
│   │   ├── requirements-version-check.sh
│   │   ├── requirements-check-compatibility.py
│   │   └── requirements-integration-test.sh
│   │
│   ├── management/                  # Scripts de gestión
│   │   ├── update-dependencies.sh
│   │   ├── backup-requirements.sh
│   │   ├── restore-requirements.sh
│   │   ├── monitor-dependencies.sh
│   │   ├── cleanup-requirements.sh
│   │   ├── requirements-sync.sh
│   │   ├── requirements-audit.sh
│   │   ├── requirements-notify.sh
│   │   └── requirements-auto-fix.sh
│   │
│   └── utils/                       # Utilidades
│       ├── optimize-requirements.py
│       ├── migrate-requirements.py
│       ├── benchmark-install.sh
│       ├── requirements-export.sh
│       ├── requirements-deps-graph.py
│       ├── ci-check-dependencies.sh
│       └── setup-dev-environment.sh
│
└── (otros scripts del proyecto)
```

## 🔄 Migración

### Scripts Legacy

Los scripts originales en `scripts/` siguen funcionando pero se recomienda usar la nueva estructura.

### Nuevos Comandos

```bash
# Análisis
make analyze          # Análisis completo
make visualize        # Visualización
make compare          # Comparar archivos
make health           # Health score
make stats            # Estadísticas

# Validación
make validate         # Validar formato
make check-all        # Todos los checks
make quick-scan       # Escaneo rápido

# Gestión
make update           # Actualizar
make backup           # Backup
make monitor          # Monitoreo
```

## 📊 Beneficios de la Refactorización

1. **Organización**: Scripts agrupados por función
2. **Mantenibilidad**: Más fácil de mantener
3. **Reutilización**: Funciones comunes compartidas
4. **Escalabilidad**: Fácil agregar nuevos scripts
5. **Documentación**: Mejor documentación por categoría

## 🚀 Uso

Los scripts mantienen la misma funcionalidad, pero ahora están mejor organizados:

```bash
# Antes
./scripts/analyze-dependencies.py

# Ahora (mismo comando, mejor organizado)
./scripts/requirements/analysis/analyze-dependencies.py

# O usar Makefile (recomendado)
make analyze
```



