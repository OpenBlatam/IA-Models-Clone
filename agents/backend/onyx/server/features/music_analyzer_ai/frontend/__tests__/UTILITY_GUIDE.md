# Test Utilities Guide

## 🛠️ Guía Completa de Test Utilities

### Importación

```typescript
import {
  createTestQueryClient,
  renderWithQueryClient,
  resetMusicStore,
  createMockTrack,
  createMockTracks,
  createMockApiResponse,
  createMockPaginatedResponse,
  wait,
  createMockError,
  createMockNetworkError,
} from '@/__tests__/setup/test-utils';
```

## 📚 Utilities Disponibles

### 1. QueryClient Helpers

#### `createTestQueryClient()`
Crea un QueryClient configurado para tests.

```typescript
const queryClient = createTestQueryClient();
```

#### `renderWithQueryClient(ui, options?)`
Renderiza un componente con QueryClient provider.

```typescript
renderWithQueryClient(<Component />);
```

### 2. Store Helpers

#### `resetMusicStore()`
Resetea el store de música a estado inicial.

```typescript
beforeEach(() => {
  resetMusicStore();
});
```

### 3. Mock Factories

#### `createMockTrack(overrides?)`
Crea un track mock.

```typescript
const track = createMockTrack({ name: 'Custom Track' });
```

#### `createMockTracks(count)`
Crea múltiples tracks mock.

```typescript
const tracks = createMockTracks(10);
```

#### `createMockApiResponse(data, success?)`
Crea una respuesta API mock.

```typescript
const response = createMockApiResponse({ data: 'test' });
```

#### `createMockPaginatedResponse(items, page?, pageSize?)`
Crea una respuesta paginada mock.

```typescript
const response = createMockPaginatedResponse(['item1', 'item2'], 1, 20);
```

### 4. Error Helpers

#### `createMockError(message?)`
Crea un error mock.

```typescript
const error = createMockError('Custom error');
```

#### `createMockNetworkError()`
Crea un error de red mock.

```typescript
const error = createMockNetworkError();
```

### 5. Async Helpers

#### `wait(ms)`
Espera un tiempo específico.

```typescript
await wait(1000);
```

## 💡 Ejemplos de Uso

### Ejemplo 1: Test con QueryClient

```typescript
import { renderWithQueryClient } from '@/__tests__/setup/test-utils';

it('should work with QueryClient', () => {
  renderWithQueryClient(<Component />);
  // ...
});
```

### Ejemplo 2: Test con Mock Track

```typescript
import { createMockTrack } from '@/__tests__/setup/test-utils';

it('should display track', () => {
  const track = createMockTrack({ name: 'Test Track' });
  render(<Component track={track} />);
  // ...
});
```

### Ejemplo 3: Test con Store Reset

```typescript
import { resetMusicStore } from '@/__tests__/setup/test-utils';

beforeEach(() => {
  resetMusicStore();
});
```

### Ejemplo 4: Test con Mock API Response

```typescript
import { createMockApiResponse } from '@/__tests__/setup/test-utils';

it('should handle API response', () => {
  const response = createMockApiResponse({ data: 'test' });
  // ...
});
```

## 🎯 Mejores Prácticas

1. ✅ Usar `renderWithQueryClient` para componentes con React Query
2. ✅ Usar `resetMusicStore` en `beforeEach` cuando uses el store
3. ✅ Usar factories de mocks para datos consistentes
4. ✅ Usar helpers de error para tests de error handling

## 📝 Notas

- Todos los helpers están testeados
- Los helpers son reutilizables
- Los helpers siguen mejores prácticas
- Los helpers están documentados

## 🔗 Ver También

- [test-utils.tsx](./setup/test-utils.tsx) - Implementación
- [test-utils.test.ts](./setup/test-utils.test.ts) - Tests
- [best-practices.md](./best-practices.md) - Mejores prácticas

