# Mejoras V16 - Hooks Composable y Utilidades Avanzadas

## 🎯 Objetivos

Esta versión se enfoca en crear hooks composables adicionales, mejorar hooks existentes, y agregar utilidades avanzadas para manipulación de datos.

## ✅ Mejoras Implementadas

### 1. **Nuevos Hooks Composable**

#### useToggle:
```typescript
const [isOpen, toggle, setTrue, setFalse, setValue] = useToggle(false);
```

#### usePrevious:
```typescript
const previousValue = usePrevious(currentValue);
```

#### useAsync:
```typescript
const { data, error, loading, execute, reset } = useAsync(
  () => fetchData(),
  { immediate: true, onSuccess: (data) => console.log(data) }
);
```

#### useInterval:
```typescript
useInterval(() => {
  // Code to run
}, 1000); // Run every second
```

#### useTimeout:
```typescript
useTimeout(() => {
  // Code to run after delay
}, 5000); // Run after 5 seconds
```

#### useIsomorphicLayoutEffect:
```typescript
// Automatically uses useLayoutEffect on client, useEffect on server
useIsomorphicLayoutEffect(() => {
  // Code
}, [deps]);
```

### 2. **Hooks Mejorados**

#### useLocalStorage:
- ✅ Agregado `removeValue` function
- ✅ Mejor manejo de errores
- ✅ Sincronización entre tabs
- ✅ Documentación JSDoc

#### useMediaQuery:
- ✅ Mejor soporte para navegadores antiguos
- ✅ Inicialización correcta en SSR
- ✅ Manejo de eventos mejorado
- ✅ Documentación completa

#### useCopyToClipboard:
- ✅ Fallback para navegadores antiguos
- ✅ Opciones configurables (timeout, callbacks)
- ✅ Función `reset` para limpiar estado
- ✅ Mejor manejo de errores

#### useClickOutside:
- ✅ Mejor tipado con generics
- ✅ Opción `enabled` para controlar el hook
- ✅ Soporte para eventos touch
- ✅ Documentación mejorada

### 3. **Utilidades de Array**

#### Funciones Implementadas:
- `unique()` - Elimina duplicados
- `groupBy()` - Agrupa por clave
- `chunk()` - Divide en chunks
- `flatten()` - Aplana arrays anidados
- `difference()` - Diferencia entre arrays
- `intersection()` - Intersección de arrays
- `shuffle()` - Mezcla aleatoria (Fisher-Yates)
- `sample()` - Obtiene elemento aleatorio
- `last()` - Último elemento
- `first()` - Primer elemento

**Ejemplo:**
```typescript
import { groupBy, chunk, unique } from '@/lib/utils';

const grouped = groupBy(users, (user) => user.role);
const chunks = chunk(items, 10);
const uniqueItems = unique([1, 2, 2, 3, 3, 3]);
```

### 4. **Utilidades de Object**

#### Funciones Implementadas:
- `omit()` - Omite keys de un objeto
- `pick()` - Selecciona keys de un objeto
- `isEmpty()` - Verifica si objeto está vacío
- `deepMerge()` - Merge profundo de objetos
- `get()` - Obtiene valor con dot notation
- `set()` - Establece valor con dot notation

**Ejemplo:**
```typescript
import { omit, pick, get, set } from '@/lib/utils';

const user = { id: 1, name: 'John', email: 'john@example.com' };
const withoutEmail = omit(user, ['email']);
const onlyName = pick(user, ['name']);
const nestedValue = get(obj, 'user.profile.name');
const updated = set(obj, 'user.profile.name', 'New Name');
```

### 5. **Utilidades de String**

#### Funciones Implementadas:
- `capitalize()` - Capitaliza primera letra
- `camelCase()` - Convierte a camelCase
- `kebabCase()` - Convierte a kebab-case
- `snakeCase()` - Convierte a snake_case
- `pascalCase()` - Convierte a PascalCase
- `stripHtml()` - Elimina tags HTML
- `escapeHtml()` - Escapa caracteres HTML
- `randomString()` - Genera string aleatorio
- `isValidEmail()` - Valida email
- `isValidUrl()` - Valida URL
- `truncate()` - Trunca string
- `cleanWhitespace()` - Limpia espacios

**Ejemplo:**
```typescript
import { capitalize, kebabCase, isValidEmail, truncate } from '@/lib/utils';

const title = capitalize('hello world'); // "Hello world"
const slug = kebabCase('Hello World'); // "hello-world"
const isValid = isValidEmail('test@example.com'); // true
const short = truncate('Long text...', 10); // "Long te..."
```

### 6. **Barrel Exports**

#### hooks/index.ts:
- Exportaciones centralizadas de todos los hooks
- Imports más limpios
- Mejor organización

#### lib/utils/index.ts:
- Exportaciones centralizadas de utilidades
- Organización por categoría (array, object, string, image)
- Fácil descubrimiento de funciones

## 📊 Beneficios

### Desarrollo:
- **Hooks reutilizables:** Menos código duplicado
- **Utilidades comunes:** Funciones listas para usar
- **Type-safe:** Todo tipado correctamente
- **Documentado:** JSDoc en todas las funciones

### Mantenibilidad:
- **Código centralizado:** Fácil de mantener
- **Consistencia:** Mismos patrones en todo el código
- **Testeable:** Funciones puras fáciles de testear

### Performance:
- **Optimizado:** Hooks con memoización donde es necesario
- **Lazy evaluation:** Cálculos solo cuando se necesitan

## 🎨 Ejemplos de Uso

### useToggle:
```typescript
function Modal() {
  const [isOpen, toggle, open, close] = useToggle(false);
  
  return (
    <>
      <button onClick={toggle}>Toggle</button>
      <button onClick={open}>Open</button>
      <button onClick={close}>Close</button>
      {isOpen && <div>Modal Content</div>}
    </>
  );
}
```

### useAsync:
```typescript
function DataFetcher() {
  const { data, error, loading, execute } = useAsync(
    () => fetch('/api/data').then(r => r.json()),
    { immediate: true }
  );
  
  if (loading) return <Loading />;
  if (error) return <Error message={error.message} />;
  return <DataDisplay data={data} />;
}
```

### Utilidades de Array:
```typescript
// Agrupar posts por plataforma
const postsByPlatform = groupBy(posts, (post) => post.platform);

// Dividir en páginas
const pages = chunk(allItems, 20);

// Obtener elementos únicos
const uniqueTags = unique(allTags);
```

### Utilidades de Object:
```typescript
// Omitir campos sensibles
const publicUser = omit(user, ['password', 'token']);

// Seleccionar solo campos necesarios
const userSummary = pick(user, ['id', 'name', 'email']);

// Obtener valor anidado
const userName = get(data, 'user.profile.name');
```

## 🚀 Próximos Pasos

1. **Aplicar hooks a componentes:**
   - Reemplazar useState/useEffect con hooks composables
   - Usar utilidades en lugar de código duplicado

2. **Testing:**
   - Unit tests para hooks
   - Unit tests para utilidades
   - Integration tests

3. **Documentación:**
   - Ejemplos de uso en Storybook
   - Guías de mejores prácticas
   - Type definitions mejoradas

## 📚 Referencias

- [React Hooks Best Practices](https://react.dev/reference/react)
- [Functional Programming Utilities](https://lodash.com/docs)
- [TypeScript Utility Types](https://www.typescriptlang.org/docs/handbook/utility-types.html)

---

**Versión:** 16
**Fecha:** 2024
**Estado:** ✅ Completo


