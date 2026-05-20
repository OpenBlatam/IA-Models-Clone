# Troubleshooting Guide - Test Suite

## 🔧 Guía de Solución de Problemas

### Problemas Comunes y Soluciones

## 1. Tests Failing - "Cannot find module"

### Problema
```
Cannot find module '@/lib/utils' from 'component.test.tsx'
```

### Solución
```bash
# Verificar que el path alias está configurado en jest.config.js
moduleNameMapper: {
  '^@/(.*)$': '<rootDir>/$1',
}

# Verificar que el archivo existe
ls lib/utils.ts
```

## 2. Tests Failing - "useQuery must be used within QueryClientProvider"

### Problema
```
useQuery must be used within QueryClientProvider
```

### Solución
```typescript
import { renderWithQueryClient } from '@/__tests__/setup/test-utils';

// Usar renderWithQueryClient en lugar de render
renderWithQueryClient(<Component />);
```

## 3. Tests Failing - "localStorage is not defined"

### Problema
```
localStorage is not defined
```

### Solución
```typescript
// En jest.setup.js o en el test
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(),
    setItem: jest.fn(),
    removeItem: jest.fn(),
    clear: jest.fn(),
  },
});
```

## 4. Tests Failing - "matchMedia is not defined"

### Problema
```
matchMedia is not defined
```

### Solución
```typescript
// En jest.setup.js
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
```

## 5. Tests Failing - "Audio is not defined"

### Problema
```
Audio is not defined
```

### Solución
```typescript
// En jest.setup.js
global.Audio = jest.fn().mockImplementation(() => ({
  play: jest.fn().mockResolvedValue(undefined),
  pause: jest.fn(),
  load: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  currentTime: 0,
  duration: 100,
  volume: 1,
  paused: true,
}));
```

## 6. Tests Failing - Async Operations

### Problema
```
Warning: An update to Component inside a test was not wrapped in act(...)
```

### Solución
```typescript
import { act, waitFor } from '@testing-library/react';

it('should update', async () => {
  render(<Component />);
  
  await act(async () => {
    await waitFor(() => {
      expect(screen.getByText('Updated')).toBeInTheDocument();
    });
  });
});
```

## 7. Tests Failing - Mock Not Working

### Problema
```
Mock function not being called
```

### Solución
```typescript
// Verificar que el mock está configurado antes del render
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

// Limpiar mocks antes de cada test
beforeEach(() => {
  jest.clearAllMocks();
});
```

## 8. Tests Failing - Timer Issues

### Problema
```
Timer mocks not working correctly
```

### Solución
```typescript
beforeEach(() => {
  jest.useFakeTimers();
});

afterEach(() => {
  jest.useRealTimers();
});

it('should debounce', () => {
  // Usar fake timers
  jest.advanceTimersByTime(500);
});
```

## 9. Tests Failing - State Not Resetting

### Problema
```
State from previous test affecting current test
```

### Solución
```typescript
import { resetMusicStore } from '@/__tests__/setup/test-utils';

beforeEach(() => {
  resetMusicStore();
  jest.clearAllMocks();
});
```

## 10. Tests Slow - Performance Issues

### Problema
```
Tests taking too long to run
```

### Solución
```typescript
// Usar fake timers para operaciones async
jest.useFakeTimers();

// Limitar timeouts
jest.setTimeout(5000);

// Usar mocks en lugar de implementaciones reales
jest.mock('heavy-library');
```

## 🔍 Debugging Tips

### 1. Ver Output Completo

```bash
npm test -- --verbose
```

### 2. Ver Solo Tests Failing

```bash
npm test -- --onlyFailures
```

### 3. Debug Específico

```typescript
it('should work', () => {
  render(<Component />);
  screen.debug(); // Ver el DOM renderizado
});
```

### 4. Ver Queries Disponibles

```typescript
import { screen } from '@testing-library/react';

it('should work', () => {
  render(<Component />);
  screen.logTestingPlaygroundURL(); // Ver queries sugeridas
});
```

## 📚 Recursos Adicionales

- [Jest Troubleshooting](https://jestjs.io/docs/troubleshooting)
- [React Testing Library Common Mistakes](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
- [Testing Library Queries](https://testing-library.com/docs/queries/about)

## ✨ Conclusión

La mayoría de problemas se resuelven con:
1. ✅ Verificar configuración de Jest
2. ✅ Limpiar mocks y estado
3. ✅ Usar helpers apropiados
4. ✅ Manejar async correctamente
5. ✅ Verificar que los mocks están configurados

