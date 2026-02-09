# Quick Reference - Test Suite

## 🚀 Comandos Esenciales

```bash
# Todos los tests
npm test

# Watch mode
npm run test:watch

# Con cobertura
npm run test:coverage

# Test específico
npm test -- nombre-del-archivo

# Solo unitarios
npm test -- --testPathIgnorePatterns=e2e

# Solo E2E
npm test -- e2e
```

## 📁 Estructura Rápida

```
__tests__/
├── components/     # Componentes React
├── lib/           # Librerías y utilidades
├── e2e/           # Tests E2E
├── integration/   # Tests de integración
├── regression/    # Tests de regresión
├── snapshots/     # Snapshot tests
├── performance/   # Performance tests
└── helpers/       # Test helpers
```

## 📊 Cobertura

- **Total**: ~95%
- **Unit**: ~94%
- **E2E**: ~92%
- **Store**: ~100%
- **Hooks**: ~100%

## 🎯 Tests por Categoría

### Componentes
- `components/music/*` - Componentes de música
- `components/ui/*` - Componentes UI base
- `components/api-status.test.tsx` - Estado de API
- `components/error-boundary.test.tsx` - Error boundary
- `components/navigation.test.tsx` - Navegación

### Hooks
- `lib/hooks/use-debounce.test.ts`
- `lib/hooks/use-local-storage.test.ts`
- `lib/hooks/use-api-health.test.ts`
- `lib/hooks/use-form-validation.test.ts`
- `lib/hooks/use-media-query.test.ts`

### Store
- `lib/store/music-store.test.ts` - Store completo

### API
- `lib/api/client.test.ts`
- `lib/api/music-api.test.ts`
- `lib/api/favorites.test.ts`
- `lib/api/recommendations.test.ts`
- `lib/api/connection-utils.test.ts`

### Utils
- `lib/utils.test.ts`
- `lib/utils/validation.test.ts`
- `lib/validations/music.test.ts`

### E2E
- `e2e/user-flows.test.tsx`
- `e2e/music-workflow.test.tsx`
- `e2e/accessibility.test.tsx`
- `e2e/performance.test.tsx`
- `e2e/advanced-workflows.test.tsx`
- `e2e/edge-cases.test.tsx`
- `e2e/stress-test.test.tsx`
- `e2e/cross-component.test.tsx`

## 🔧 Helpers Útiles

```typescript
// Test helpers
import {
  createTestQueryClient,
  renderWithQueryClient,
  createMockTrack,
  createMockApiResponse,
} from '@/__tests__/helpers/test-helpers';
```

## 📝 Patrón Básico

```typescript
describe('Component', () => {
  beforeEach(() => {
    // Setup
  });

  it('should do something', () => {
    // Arrange
    // Act
    // Assert
  });
});
```

## ✅ Checklist para Nuevos Tests

- [ ] Test unitario básico
- [ ] Casos edge
- [ ] Manejo de errores
- [ ] Estados de carga
- [ ] Interacciones de usuario
- [ ] Accesibilidad
- [ ] Performance (si aplica)

## 🎉 Estadísticas

- **82+ archivos** de tests
- **550+ tests** individuales
- **50+ flujos E2E**
- **~95% cobertura**

¡Tests completos y listos! 🚀

