# Best Practices - Testing Guide

## 📚 Guía de Mejores Prácticas para Tests

### 1. Estructura de Tests (AAA Pattern)

```typescript
describe('Component', () => {
  it('should do something', () => {
    // Arrange - Preparar el estado inicial
    const props = { name: 'Test' };
    
    // Act - Ejecutar la acción
    render(<Component {...props} />);
    
    // Assert - Verificar el resultado
    expect(screen.getByText('Test')).toBeInTheDocument();
  });
});
```

### 2. Nombres Descriptivos

✅ **Bueno:**
```typescript
it('should display error message when API call fails', () => {});
it('should disable submit button when form is invalid', () => {});
```

❌ **Malo:**
```typescript
it('test 1', () => {});
it('works', () => {});
```

### 3. Tests Independientes

✅ **Bueno:**
```typescript
describe('Component', () => {
  beforeEach(() => {
    // Setup limpio para cada test
    jest.clearAllMocks();
  });

  it('test 1', () => {
    // Test independiente
  });
});
```

❌ **Malo:**
```typescript
it('test 1', () => {
  // Modifica estado global
});

it('test 2', () => {
  // Depende de test 1
});
```

### 4. Casos Edge

Siempre incluir casos edge:

```typescript
describe('formatDuration', () => {
  it('should handle normal values', () => {});
  it('should handle zero', () => {});
  it('should handle negative values', () => {});
  it('should handle very large values', () => {});
  it('should handle NaN', () => {});
  it('should handle Infinity', () => {});
});
```

### 5. Mocks Apropiados

✅ **Bueno:**
```typescript
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));
```

❌ **Malo:**
```typescript
// Mockear todo el módulo cuando solo necesitas una función
jest.mock('@/lib/api/music-api');
```

### 6. User Events

Usar `userEvent` para interacciones realistas:

```typescript
const user = userEvent.setup();
await user.click(button);
await user.type(input, 'text');
```

### 7. Async Handling

Manejar operaciones async correctamente:

```typescript
it('should load data', async () => {
  render(<Component />);
  
  await waitFor(() => {
    expect(screen.getByText('Loaded')).toBeInTheDocument();
  });
});
```

### 8. Cleanup

Limpiar después de cada test:

```typescript
afterEach(() => {
  jest.clearAllMocks();
  cleanup();
});
```

### 9. Test Helpers

Usar helpers reutilizables:

```typescript
import { renderWithQueryClient, createMockTrack } from '@/__tests__/setup/test-utils';

it('should work', () => {
  const track = createMockTrack();
  renderWithQueryClient(<Component track={track} />);
});
```

### 10. Coverage Goals

- **Mínimo**: 80% cobertura
- **Objetivo**: 90%+ cobertura
- **Ideal**: 95%+ cobertura

## 🎯 Checklist para Nuevos Tests

- [ ] Test básico de renderizado
- [ ] Test de interacciones de usuario
- [ ] Test de estados de carga
- [ ] Test de manejo de errores
- [ ] Test de casos edge
- [ ] Test de accesibilidad (si aplica)
- [ ] Test de performance (si aplica)

## 📝 Ejemplos Completos

### Test de Componente Simple

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

### Test con QueryClient

```typescript
import { renderWithQueryClient } from '@/__tests__/setup/test-utils';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should work with QueryClient', () => {
    renderWithQueryClient(<Component />);
    // ...
  });
});
```

### Test con User Events

```typescript
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Component } from '@/components/Component';

describe('Component', () => {
  it('should handle user interaction', async () => {
    const user = userEvent.setup();
    render(<Component />);
    
    const button = screen.getByRole('button');
    await user.click(button);
    
    expect(screen.getByText('Clicked')).toBeInTheDocument();
  });
});
```

### Test con Mocks

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { Component } from '@/components/Component';
import { musicApiService } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api');

describe('Component', () => {
  it('should load data', async () => {
    (musicApiService.searchTracks as jest.Mock).mockResolvedValue({
      success: true,
      results: [],
    });

    render(<Component />);
    
    await waitFor(() => {
      expect(musicApiService.searchTracks).toHaveBeenCalled();
    });
  });
});
```

## 🚫 Errores Comunes

### 1. Tests Dependientes

❌ **Malo:**
```typescript
it('test 1', () => {
  globalState.value = 'test';
});

it('test 2', () => {
  expect(globalState.value).toBe('test'); // Depende de test 1
});
```

### 2. Mocks No Limpiados

❌ **Malo:**
```typescript
it('test 1', () => {
  jest.spyOn(api, 'call').mockResolvedValue('result');
});

it('test 2', () => {
  // Mock todavía activo de test 1
});
```

### 3. Async No Manejado

❌ **Malo:**
```typescript
it('should load', () => {
  render(<Component />);
  expect(screen.getByText('Loaded')).toBeInTheDocument(); // Falla
});
```

✅ **Bueno:**
```typescript
it('should load', async () => {
  render(<Component />);
  await waitFor(() => {
    expect(screen.getByText('Loaded')).toBeInTheDocument();
  });
});
```

## ✨ Conclusión

Seguir estas mejores prácticas asegura:
- ✅ Tests confiables
- ✅ Tests mantenibles
- ✅ Tests rápidos
- ✅ Alta cobertura
- ✅ Código de calidad

