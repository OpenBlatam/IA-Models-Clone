# 🚀 Quick Start - Refactoring

## Inicio Rápido (5 minutos)

### 1. Setup Inicial

```bash
# Windows
scripts\refactor\setup-structure.bat

# Linux/Mac
mkdir -p docs/{architecture,dependencies,features,guides,api}
mkdir -p config/{environments,schemas,models}
mkdir -p services/{analysis,recommendations,tracking,products,ml,notifications,integrations,reporting,social,shared}
mkdir -p utils/{logging,caching,validation,security,performance,database,async,monitoring,helpers}
```

### 2. Refactorización Completa

```bash
# Todo en uno
make refactor-complete
```

### 3. Verificar

```bash
# Validar refactorización
make validate-refactoring

# Generar reporte
make generate-report
```

## Comandos Esenciales

```bash
# Refactorizar todo
make refactor-complete

# Solo documentación
make refactor-docs

# Solo código
make organize-code

# Validar
make validate-refactoring
```

## Estructura Resultante

```
✅ services/          → 10 categorías
✅ utils/             → 9 categorías
✅ docs/              → 5 categorías
✅ scripts/           → Organizados
✅ config/            → Por ambiente
```

## Próximos Pasos

1. Revisar reporte: `docs/REFACTORING_REPORT.md`
2. Actualizar imports: `make update-imports-dry-run`
3. Validar cambios: `make validate-refactoring`

---

**Tiempo estimado**: 5-10 minutos  
**Complejidad**: Baja  
**Riesgo**: Bajo (compatible hacia atrás)



