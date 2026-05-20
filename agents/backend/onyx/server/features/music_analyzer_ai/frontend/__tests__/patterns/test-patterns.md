# Test Patterns - Test Suite

## 🎨 Patrones de Testing

### 1. AAA Pattern (Arrange, Act, Assert)

```typescript
it('should do something', () => {
  // Arrange - Preparar
  const props = { name: 'Test' };
  
  // Act - Ejecutar
  render(<Component {...props} />);
  
  // Assert - Verificar
  expect(screen.getByText('Test')).toBeInTheDocument();
});
```

### 2. Given-When-Then Pattern

```typescript
it('should handle user interaction', () => {
  // Given - Dado
  const user = userEvent.setup();
  render(<Component />);
  
  // When - Cuando
  await user.click(button);
  
  // Then - Entonces
  expect(result).toBe(expected);
});
```

### 3. Setup/Teardown Pattern

```typescript
describe('Component', () => {
  beforeEach(() => {
    // Setup antes de cada test
  });

  afterEach(() => {
    // Cleanup después de cada test
  });
});
```

### 4. Factory Pattern

```typescript
// Test data factories
const track = createMockTrack();
const tracks = createMockTracks(10);
```

### 5. Builder Pattern

```typescript
// Test builders
renderWithQueryClient(<Component />);
renderWithProviders(<Component />);
```

### 6. Page Object Pattern (para E2E)

```typescript
class SearchPage {
  async search(query: string) {
    await user.type(input, query);
  }
  
  async selectTrack(name: string) {
    await user.click(getByText(name));
  }
}
```

### 7. Test Doubles Pattern

```typescript
// Mocks, Stubs, Spies
jest.mock('@/lib/api/music-api');
const mockFn = jest.fn();
```

## 🎯 Patrones por Tipo de Test

### Unit Tests
- ✅ AAA Pattern
- ✅ Factory Pattern
- ✅ Mock Pattern

### Integration Tests
- ✅ Setup/Teardown Pattern
- ✅ Builder Pattern
- ✅ Test Doubles Pattern

### E2E Tests
- ✅ Given-When-Then Pattern
- ✅ Page Object Pattern
- ✅ Flow Pattern

## ✨ Mejores Prácticas

1. ✅ Usar patrones consistentemente
2. ✅ Elegir patrón apropiado
3. ✅ Documentar patrones usados
4. ✅ Mantener patrones simples

## 🎊 Conclusión

Patrones bien aplicados aseguran:
- ✅ Consistencia
- ✅ Legibilidad
- ✅ Mantenibilidad
- ✅ Escalabilidad

¡Patrones profesionales! 🎨
