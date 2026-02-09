# Test Suite - Music Analyzer AI Frontend

## 📚 Guía Completa de Tests

Esta suite de tests proporciona cobertura completa (~95%) del frontend de Music Analyzer AI.

## 🏗️ Estructura de Tests

```
__tests__/
├── components/          # Tests de componentes React
│   ├── music/          # Componentes de música
│   └── ui/             # Componentes UI base
├── lib/                 # Tests de librerías y utilidades
│   ├── hooks/          # Tests de hooks personalizados
│   ├── store/          # Tests de Zustand store
│   ├── api/            # Tests de servicios API
│   ├── utils/          # Tests de utilidades
│   ├── validations/    # Tests de validaciones
│   ├── constants/      # Tests de constantes
│   ├── config/         # Tests de configuración
│   └── types/          # Tests de tipos
├── e2e/                # Tests end-to-end
├── integration/        # Tests de integración
├── regression/         # Tests de regresión
├── snapshots/          # Snapshot tests
├── performance/        # Tests de performance
└── helpers/            # Helpers para tests
```

## 🚀 Comandos Rápidos

```bash
# Ejecutar todos los tests
npm test

# Modo watch
npm run test:watch

# Con cobertura
npm run test:coverage

# Test específico
npm test -- nombre-del-test

# Solo tests unitarios
npm test -- --testPathIgnorePatterns=e2e

# Solo tests E2E
npm test -- e2e
```

## 📊 Cobertura

### Cobertura Total: ~95%

- **Unit Tests**: ~94%
- **Integration Tests**: ~85%
- **E2E Tests**: ~92%
- **Store Tests**: ~100%
- **UI Component Tests**: ~100%
- **Type Tests**: ~100%

Ver [coverage-report.md](./coverage/coverage-report.md) para detalles completos.

## 🧪 Tipos de Tests

### 1. Unit Tests
Tests individuales de componentes, hooks, utilidades, etc.

```bash
npm test -- components
npm test -- lib
```

### 2. Integration Tests
Tests de integración entre componentes y servicios.

```bash
npm test -- integration
```

### 3. E2E Tests
Tests end-to-end de flujos completos de usuario.

```bash
npm test -- e2e
```

### 4. Regression Tests
Tests para prevenir regresiones.

```bash
npm test -- regression
```

### 5. Performance Tests
Tests de rendimiento y optimización.

```bash
npm test -- performance
```

## 📝 Escribir Nuevos Tests

### Estructura Básica

```typescript
import { render, screen } from '@testing-library/react';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should render correctly', () => {
    render(<Component />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### Usando Helpers

```typescript
import { renderWithQueryClient, createMockTrack } from '@/__tests__/helpers/test-helpers';

describe('Component', () => {
  it('should work with QueryClient', () => {
    const track = createMockTrack();
    renderWithQueryClient(<Component track={track} />);
    // ...
  });
});
```

## 🎯 Mejores Prácticas

1. **AAA Pattern**: Arrange, Act, Assert
2. **Tests independientes**: Cada test debe ser aislado
3. **Nombres descriptivos**: Nombres claros y comprensibles
4. **Casos edge**: Cubrir casos límite
5. **Mocks apropiados**: Mockear dependencias externas
6. **Setup/Teardown**: Usar beforeEach/afterEach

## 🔍 Debugging Tests

```bash
# Modo watch con filtro
npm run test:watch -- --testNamePattern="nombre del test"

# Con verbose
npm test -- --verbose

# Con coverage específico
npm run test:coverage -- --collectCoverageFrom="components/**/*.tsx"
```

## 📚 Recursos

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library](https://testing-library.com/react)
- [User Event](https://testing-library.com/docs/user-event/intro)
- [Coverage Report](./coverage/coverage-report.md)
- [Final Summary](./COMPLETE_FINAL_SUMMARY.md)

## ✨ Contribuir

Al agregar nuevas funcionalidades:

1. ✅ Escribir tests primero (TDD)
2. ✅ Mantener cobertura >90%
3. ✅ Agregar casos edge
4. ✅ Actualizar documentación
5. ✅ Ejecutar todos los tests antes de commit

## 🎉 Estado Actual

- ✅ **82+ archivos** de tests
- ✅ **550+ tests** individuales
- ✅ **50+ flujos E2E**
- ✅ **~95% cobertura total**
- ✅ **100% hooks testeados**
- ✅ **100% store testeados**
- ✅ **100% UI components testeados**

¡La suite de tests está completa y lista para producción! 🚀
