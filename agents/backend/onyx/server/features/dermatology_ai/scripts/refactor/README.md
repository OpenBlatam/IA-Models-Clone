# 🔄 Refactoring Scripts

Scripts para refactorizar y organizar el proyecto Dermatology AI.

## 📁 Scripts Disponibles

### Setup y Estructura

1. **setup-structure.bat** - Crea estructura de directorios
   ```bash
   scripts\refactor\setup-structure.bat
   ```

### Migración

2. **move-docs.sh** - Mueve documentación a estructura organizada
   ```bash
   bash scripts/refactor/move-docs.sh
   # o
   make refactor-docs
   ```

3. **organize-config.sh** - Organiza archivos de configuración
   ```bash
   bash scripts/refactor/organize-config.sh
   # o
   make refactor-config
   ```

### Análisis

4. **analyze-structure.py** - Analiza estructura del proyecto
   ```bash
   python scripts/refactor/analyze-structure.py
   # o
   make analyze-structure
   ```

5. **check-refactoring.sh** - Verifica estado de refactorización
   ```bash
   bash scripts/refactor/check-refactoring.sh
   # o
   make check-refactoring
   ```

6. **create-index.py** - Crea índice de documentación
   ```bash
   python scripts/refactor/create-index.py
   # o
   make create-docs-index
   ```

## 🚀 Uso Rápido

### Setup Inicial

```bash
# Windows
scripts\refactor\setup-structure.bat

# Linux/Mac
mkdir -p docs/{architecture,dependencies,features,guides,api}
mkdir -p config/{environments,schemas,models}
```

### Refactorización Completa

```bash
# Verificar estado
make check-refactoring

# Analizar estructura
make analyze-structure

# Refactorizar todo
make refactor-all

# Crear índice
make create-docs-index
```

## 📊 Comandos Make

```bash
make refactor-docs        # Mover documentación
make refactor-config      # Organizar configuración
make refactor-all         # Refactorizar todo
make analyze-structure    # Analizar estructura
make check-refactoring    # Verificar estado
make create-docs-index    # Crear índice
```

## 📚 Documentación Relacionada

- [QUICK_START.md](QUICK_START.md) - Inicio rápido
- [REFACTORING_PLAN.md](../../REFACTORING_PLAN.md) - Plan completo
- [REFACTORING_COMPLETE.md](../../REFACTORING_COMPLETE.md) - Resumen
- [REFACTORING_FINAL.md](../../REFACTORING_FINAL.md) - Resumen final
- [docs/README.md](../../docs/README.md) - Documentación organizada

## 🚀 Quick Start

Para refactorización rápida:

```bash
# Setup y refactorización completa
make refactor-complete

# Verificar
make validate-refactoring
```

Ver [QUICK_START.md](QUICK_START.md) para más detalles.

