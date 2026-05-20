# Guía de Refactorización

## Patrones de Refactorización Aplicados

### 1. Centralización de Lógica

#### Antes
```typescript
// En cada componente
try {
  await apiCall();
  toast.success('Éxito');
} catch (error) {
  console.error(error);
  toast.error('Error');
}
```

#### Después
```typescript
import { handleApiError, handleFormSubmit } from '@/lib/utils';

// Opción 1: Con helper
await handleFormSubmit(async () => {
  await apiCall();
}, { successMessage: 'Éxito' });

// Opción 2: Con hook
const { request, loading, error } = useApi();
await request('/api/endpoint');
```

### 2. Eliminación de Duplicación

#### Antes
```typescript
// En múltiples componentes
const [data, setData] = useState([]);
useEffect(() => {
  const saved = localStorage.getItem('key');
  if (saved) {
    setData(JSON.parse(saved));
  }
}, []);
```

#### Después
```typescript
import { useLocalStorageState } from '@/lib/hooks';

const [data, setData] = useLocalStorageState('key', []);
```

### 3. Uso de Hooks Personalizados

#### Antes
```typescript
const messagesEndRef = useRef<HTMLDivElement>(null);
useEffect(() => {
  messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
}, [messages]);
```

#### Después
```typescript
import { useScrollToBottom } from '@/lib/hooks';

const { containerRef, scrollToBottom } = useScrollToBottom([messages]);
```

### 4. Utilidades Funcionales

#### Antes
```typescript
const filtered = items.filter(item => item.status === 'active');
const sorted = filtered.sort((a, b) => a.name.localeCompare(b.name));
```

#### Después
```typescript
import { filterBy, sortBy, propertyEquals } from '@/lib/utils';

const filtered = filterBy(items, 'status', 'active');
const sorted = sortBy(filtered, 'name');
// O con predicados
const filtered = items.filter(propertyEquals('status', 'active'));
```

### 5. Manejo de Errores Centralizado

#### Antes
```typescript
try {
  await operation();
} catch (error) {
  console.error(error);
  toast.error('Error genérico');
}
```

#### Después
```typescript
import { handleApiError, withErrorHandling } from '@/lib/utils';

// Opción 1: Manual
try {
  await operation();
} catch (error) {
  handleApiError(error, 'Mensaje personalizado');
}

// Opción 2: Wrapper
const safeOperation = withErrorHandling(operation, {
  fallbackMessage: 'Mensaje personalizado'
});
await safeOperation();
```

## Mejores Prácticas

### 1. Imports Organizados
```typescript
// ✅ Bueno - Imports centralizados
import { formatNumber, handleApiError, useLocalStorageState } from '@/lib';

// ✅ También bueno - Imports directos para tree-shaking
import { formatNumber } from '@/lib/utils/format';
import { useLocalStorageState } from '@/lib/hooks/useLocalStorageState';
```

### 2. Uso de Hooks
```typescript
// ✅ Bueno - Hooks personalizados
const [data, setData] = useLocalStorageState('key', []);

// ❌ Evitar - Lógica duplicada
const [data, setData] = useState([]);
useEffect(() => {
  // ... lógica de localStorage
}, []);
```

### 3. Manejo de Errores
```typescript
// ✅ Bueno - Helpers centralizados
handleApiError(error, 'Mensaje personalizado');

// ❌ Evitar - Manejo manual repetitivo
try {
  // ...
} catch (error) {
  console.error(error);
  toast.error('Error');
}
```

### 4. Utilidades Funcionales
```typescript
// ✅ Bueno - Utilidades reutilizables
const result = pipe(
  filterByStatus('active'),
  sortBy('name'),
  take(10)
)(items);

// ❌ Evitar - Lógica inline repetida
const result = items
  .filter(i => i.status === 'active')
  .sort((a, b) => a.name.localeCompare(b.name))
  .slice(0, 10);
```

## Checklist de Refactorización

- [ ] Reemplazar `localStorage` directo con `useLocalStorageState`
- [ ] Reemplazar `toast.error` manual con `handleApiError`
- [ ] Usar `handleFormSubmit` para formularios
- [ ] Usar hooks personalizados para scroll, focus, etc.
- [ ] Usar utilidades funcionales para transformaciones
- [ ] Eliminar código duplicado
- [ ] Centralizar lógica común
- [ ] Mejorar type safety con type guards
- [ ] Usar helpers de renderizado condicional
- [ ] Optimizar imports para tree-shaking



