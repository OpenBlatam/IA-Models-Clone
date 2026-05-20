# Advanced Troubleshooting - Test Suite

## 🔧 Troubleshooting Avanzado

### Problemas Comunes y Soluciones

#### 1. Tests Flaky (Inconsistentes)

**Problema**: Tests que pasan a veces y fallan otras veces.

**Soluciones**:
```typescript
// ❌ Evitar timeouts fijos
await new Promise(resolve => setTimeout(resolve, 1000));

// ✅ Usar waitFor
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument();
});

// ✅ Usar fake timers
jest.useFakeTimers();
// ... test code
jest.useRealTimers();
```

#### 2. Memory Leaks

**Problema**: Tests que consumen mucha memoria.

**Soluciones**:
```typescript
// ✅ Limpiar después de cada test
afterEach(() => {
  cleanup();
  jest.clearAllMocks();
});

// ✅ Limpiar event listeners
afterEach(() => {
  window.removeEventListener('resize', handler);
});
```

#### 3. Async Issues

**Problema**: Tests que no esperan operaciones asíncronas.

**Soluciones**:
```typescript
// ✅ Usar async/await
it('should load data', async () => {
  await waitFor(() => {
    expect(screen.getByText('Data')).toBeInTheDocument();
  });
});

// ✅ Esperar queries
const { waitFor } = renderHook(() => useQuery(...));
await waitFor(() => result.current.isSuccess);
```

#### 4. Mock Issues

**Problema**: Mocks que no funcionan correctamente.

**Soluciones**:
```typescript
// ✅ Resetear mocks
beforeEach(() => {
  jest.clearAllMocks();
});

// ✅ Verificar llamadas
expect(mockFn).toHaveBeenCalledWith(expectedArgs);

// ✅ Mockear módulos correctamente
jest.mock('@/lib/api', () => ({
  apiService: {
    fetch: jest.fn(),
  },
}));
```

#### 5. State Management Issues

**Problema**: Estado que no se resetea entre tests.

**Soluciones**:
```typescript
// ✅ Resetear store
beforeEach(() => {
  useMusicStore.getState().reset();
});

// ✅ Crear nuevo store para cada test
const createTestStore = () => {
  // ...
};
```

#### 6. Component Rendering Issues

**Problema**: Componentes que no se renderizan correctamente.

**Soluciones**:
```typescript
// ✅ Verificar que el componente existe
expect(container.firstChild).toBeInTheDocument();

// ✅ Verificar props
expect(component.props.value).toBe(expectedValue);

// ✅ Verificar eventos
fireEvent.click(button);
expect(onClick).toHaveBeenCalled();
```

#### 7. Query Client Issues

**Problema**: React Query no funciona en tests.

**Soluciones**:
```typescript
// ✅ Crear nuevo QueryClient para cada test
const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

// ✅ Wrappear con QueryClientProvider
render(
  <QueryClientProvider client={queryClient}>
    <Component />
  </QueryClientProvider>
);
```

#### 8. Router Issues

**Problema**: Next.js router no funciona en tests.

**Soluciones**:
```typescript
// ✅ Mockear next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
  }),
}));
```

#### 9. Performance Issues

**Problema**: Tests que tardan mucho.

**Soluciones**:
```typescript
// ✅ Usar fake timers
jest.useFakeTimers();

// ✅ Mockear operaciones pesadas
jest.mock('heavy-library', () => ({
  heavyOperation: jest.fn(),
}));

// ✅ Limitar datos de test
const limitedData = largeArray.slice(0, 10);
```

#### 10. Coverage Issues

**Problema**: Cobertura que no aumenta.

**Soluciones**:
```typescript
// ✅ Testear todas las ramas
if (condition) {
  // Test case 1
} else {
  // Test case 2
}

// ✅ Testear edge cases
it('should handle null', () => {
  expect(process(null)).toBeNull();
});

// ✅ Testear error cases
it('should handle errors', () => {
  expect(() => process(invalid)).toThrow();
});
```

## 🎯 Debugging Tips

### 1. Usar screen.debug()
```typescript
screen.debug(); // Ver el DOM completo
screen.debug(container); // Ver un elemento específico
```

### 2. Usar console.log
```typescript
console.log('State:', state);
console.log('Props:', props);
```

### 3. Usar breakpoints
```typescript
debugger; // Pausar ejecución
```

### 4. Verificar mocks
```typescript
console.log(mockFn.mock.calls); // Ver todas las llamadas
```

## ✨ Conclusión

Estas soluciones ayudan a:
- ✅ Resolver problemas comunes
- ✅ Mejorar estabilidad de tests
- ✅ Acelerar debugging
- ✅ Mantener tests confiables

¡Troubleshooting avanzado documentado! 🔧

