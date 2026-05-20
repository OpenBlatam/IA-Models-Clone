# Execution Guide - Test Suite

## 🚀 Guía de Ejecución de Tests

### Ejecución Básica

```bash
# Todos los tests
npm test

# Con cobertura
npm run test:coverage

# Modo watch
npm run test:watch
```

### Ejecución por Categoría

```bash
# Solo unitarios
npm test -- --testPathIgnorePatterns=e2e

# Solo E2E
npm test -- e2e

# Solo componentes
npm test -- components

# Solo lib
npm test -- lib

# Solo integración
npm test -- integration

# Solo regresión
npm test -- regression

# Solo performance
npm test -- performance

# Solo accesibilidad
npm test -- accessibility
```

### Ejecución con Opciones

```bash
# Verbose
npm test -- --verbose

# Solo failures
npm test -- --onlyFailures

# Con coverage específico
npm test -- --collectCoverageFrom="components/**/*.tsx"

# Con timeout personalizado
npm test -- --testTimeout=10000

# Con workers limitados
npm test -- --maxWorkers=2

# Sin cache
npm test -- --no-cache
```

### Ejecución en CI/CD

```bash
# Modo CI
npm test -- --ci --coverage --maxWorkers=2

# Con thresholds
npm test -- --coverageThreshold='{"global":{"branches":90}}'
```

## 📊 Interpretación de Resultados

### Tests Passing
✅ Todos los tests pasan - código listo

### Tests Failing
❌ Revisar errores y corregir

### Coverage Bajo
⚠️ Agregar más tests para aumentar cobertura

## 🔍 Debugging

```bash
# Debug específico
npm test -- --testNamePattern="nombre del test"

# Con Node debugger
node --inspect-brk node_modules/.bin/jest --runInBand
```

## ✨ Mejores Prácticas

1. ✅ Ejecutar tests antes de commit
2. ✅ Verificar cobertura regularmente
3. ✅ Mantener tests actualizados
4. ✅ Revisar tests fallidos inmediatamente

## 📝 Notas

- Tests deben ejecutarse regularmente
- Coverage debe mantenerse >90%
- Tests deben ser rápidos (<5min total)

