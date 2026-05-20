# Legacy Tests - Migration Guide

## 📜 Guía de Migración de Tests Legacy

### Tests Legacy Identificados

Si encuentras tests legacy que necesitan migración, sigue esta guía.

## 🔄 Proceso de Migración

### 1. Identificar Tests Legacy

Tests legacy pueden tener:
- ❌ Nombres no descriptivos
- ❌ Dependencias entre tests
- ❌ Mocks no apropiados
- ❌ Sin casos edge
- ❌ Sin documentación

### 2. Migrar a Nuevos Estándares

#### Antes (Legacy)
```typescript
it('test 1', () => {
  // Test sin estructura
});
```

#### Después (Nuevo)
```typescript
it('should render component correctly', () => {
  // Arrange
  const props = { name: 'Test' };
  
  // Act
  render(<Component {...props} />);
  
  // Assert
  expect(screen.getByText('Test')).toBeInTheDocument();
});
```

### 3. Usar Nuevos Utilities

#### Antes (Legacy)
```typescript
const queryClient = new QueryClient();
render(
  <QueryClientProvider client={queryClient}>
    <Component />
  </QueryClientProvider>
);
```

#### Después (Nuevo)
```typescript
import { renderWithQueryClient } from '@/__tests__/setup/test-utils';

renderWithQueryClient(<Component />);
```

### 4. Agregar Casos Edge

#### Antes (Legacy)
```typescript
it('should work', () => {
  // Solo caso normal
});
```

#### Después (Nuevo)
```typescript
it('should work with normal input', () => {});
it('should handle empty input', () => {});
it('should handle null input', () => {});
it('should handle invalid input', () => {});
```

## ✅ Checklist de Migración

- [ ] Nombre descriptivo
- [ ] Estructura AAA
- [ ] Tests independientes
- [ ] Mocks apropiados
- [ ] Casos edge incluidos
- [ ] Documentación clara
- [ ] Usa test utilities
- [ ] Sigue mejores prácticas

## 📚 Recursos

- [best-practices.md](./best-practices.md) - Mejores prácticas
- [example-tests.md](./examples/example-tests.md) - Ejemplos
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Guía de contribución

## ✨ Conclusión

Migrar tests legacy mejora:
- ✅ Mantenibilidad
- ✅ Legibilidad
- ✅ Confiabilidad
- ✅ Consistencia

¡Mantén la suite actualizada! 🔄

