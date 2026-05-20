# CI/CD Test Scripts

## 🚀 Scripts para CI/CD

### Scripts Disponibles

```bash
# Ejecutar todos los tests
npm test

# Tests con cobertura
npm run test:coverage

# Tests en modo watch
npm run test:watch

# Tests específicos
npm test -- nombre-del-test

# Tests con verbose
npm test -- --verbose

# Tests con solo failures
npm test -- --onlyFailures
```

## 📊 Scripts de Cobertura

```bash
# Generar reporte de cobertura
npm run test:coverage

# Cobertura con umbral mínimo
npm run test:coverage -- --coverageThreshold='{"global":{"branches":90,"functions":90,"lines":90,"statements":90}}'

# Cobertura HTML
npm run test:coverage && open coverage/lcov-report/index.html
```

## 🔍 Scripts de Debugging

```bash
# Debug con Node
node --inspect-brk node_modules/.bin/jest --runInBand

# Tests con más información
npm test -- --verbose --no-coverage

# Tests con detección de leaks
npm test -- --detectOpenHandles --forceExit
```

## ⚡ Scripts de Performance

```bash
# Tests con cache
npm test -- --cache

# Tests sin cache
npm test -- --no-cache

# Tests con workers limitados
npm test -- --maxWorkers=2
```

## 🎯 Scripts Específicos

```bash
# Solo tests unitarios
npm test -- --testPathIgnorePatterns=e2e

# Solo tests E2E
npm test -- e2e

# Solo tests de componentes
npm test -- components

# Solo tests de lib
npm test -- lib

# Solo tests de integración
npm test -- integration
```

## 📝 Scripts Personalizados

Agregar a `package.json`:

```json
{
  "scripts": {
    "test:unit": "jest --testPathIgnorePatterns=e2e",
    "test:e2e": "jest e2e",
    "test:components": "jest components",
    "test:lib": "jest lib",
    "test:integration": "jest integration",
    "test:watch:unit": "jest --watch --testPathIgnorePatterns=e2e",
    "test:ci": "jest --ci --coverage --maxWorkers=2"
  }
}
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run test:ci
      - uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
```

## ✨ Mejores Prácticas CI/CD

1. ✅ Ejecutar tests en cada push
2. ✅ Requerir cobertura mínima
3. ✅ Falla si tests fallan
4. ✅ Generar reportes de cobertura
5. ✅ Cache de dependencias

