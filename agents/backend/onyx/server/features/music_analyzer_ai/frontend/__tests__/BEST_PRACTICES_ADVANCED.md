# Advanced Best Practices - Test Suite

## 🎯 Mejores Prácticas Avanzadas

### 1. Test Organization

#### Estructura de Tests
```typescript
describe('ComponentName', () => {
  describe('Rendering', () => {
    // Tests de renderizado
  });

  describe('Interactions', () => {
    // Tests de interacciones
  });

  describe('Edge Cases', () => {
    // Tests de casos extremos
  });
});
```

#### Naming Conventions
- ✅ `should [expected behavior] when [condition]`
- ✅ `should handle [scenario] correctly`
- ✅ `should not [unexpected behavior]`

### 2. Test Data Management

#### Fixtures
```typescript
// __tests__/fixtures/test-data.ts
export const mockTrack = {
  id: '123',
  name: 'Test Track',
  // ...
};
```

#### Factories
```typescript
export const createMockTrack = (overrides = {}) => ({
  id: '123',
  name: 'Test Track',
  ...overrides,
});
```

### 3. Mocking Strategies

#### API Mocks
```typescript
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));
```

#### Component Mocks
```typescript
jest.mock('@/components/SomeComponent', () => ({
  SomeComponent: () => <div>Mocked</div>,
}));
```

### 4. Async Testing

#### Wait for Updates
```typescript
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument();
});
```

#### Async Utilities
```typescript
await waitForElementToBeRemoved(() =>
  screen.queryByText('Loading')
);
```

### 5. Error Testing

#### Error Boundaries
```typescript
it('should catch and display errors', () => {
  const ThrowError = () => {
    throw new Error('Test error');
  };

  render(
    <ErrorBoundary>
      <ThrowError />
    </ErrorBoundary>
  );

  expect(screen.getByText(/error/i)).toBeInTheDocument();
});
```

### 6. Performance Testing

#### Render Performance
```typescript
it('should render within acceptable time', () => {
  const start = performance.now();
  render(<Component />);
  const end = performance.now();

  expect(end - start).toBeLessThan(100);
});
```

### 7. Accessibility Testing

#### ARIA Attributes
```typescript
it('should have proper ARIA labels', () => {
  render(<Button aria-label="Close" />);
  expect(screen.getByLabelText('Close')).toBeInTheDocument();
});
```

#### Keyboard Navigation
```typescript
it('should be keyboard navigable', () => {
  const { container } = render(<Component />);
  const firstButton = container.querySelector('button');
  firstButton?.focus();
  expect(document.activeElement).toBe(firstButton);
});
```

### 8. Security Testing

#### XSS Prevention
```typescript
it('should sanitize user input', () => {
  const malicious = '<script>alert("XSS")</script>';
  const sanitized = sanitizeInput(malicious);
  expect(sanitized).not.toContain('<script>');
});
```

### 9. Edge Cases

#### Boundary Conditions
```typescript
it('should handle empty arrays', () => {
  expect(processArray([])).toEqual([]);
});

it('should handle null values', () => {
  expect(processValue(null)).toBeNull();
});
```

### 10. Test Maintenance

#### DRY Principle
```typescript
// Reusable test utilities
const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient();
  return render(
    <QueryClientProvider client={queryClient}>
      {component}
    </QueryClientProvider>
  );
};
```

#### Regular Updates
- ✅ Actualizar tests cuando cambia el código
- ✅ Eliminar tests obsoletos
- ✅ Refactorizar tests duplicados

## ✨ Conclusión

Seguir estas mejores prácticas asegura:
- ✅ Tests mantenibles
- ✅ Tests confiables
- ✅ Tests rápidos
- ✅ Tests completos

¡Mejores prácticas avanzadas implementadas! 🎯

