# Modular Utils - Guía Completa de Uso

Esta estructura modular permite importar utilidades de manera organizada y eficiente, mejorando el tree-shaking y la claridad del código.

## 🎯 Módulos Principales

### Core Module (`modules/core.ts`)
Utilidades esenciales de manipulación de datos:

```ts
import { unique, deepMerge, camelCase } from '@/app/continuous-agent/utils/modules/core';

// O importación por namespace
import { Array, Object, String } from '@/app/continuous-agent/utils/modules/core';

const unique = Array.unique([1, 2, 2]);
const merged = Object.deepMerge({ a: 1 }, { b: 2 });
const slug = String.slugify("Hello World!");
```

**Funciones incluidas:**
- **Array**: chunk, groupBy, unique, flatten, partition, sortBy, zip, intersection, etc.
- **Object**: pick, omit, deepMerge, mapValues, mapKeys, invert, compact, etc.
- **String**: truncate, capitalize, camelCase, kebabCase, slugify, escapeHtml, etc.
- **Type**: isString, isNumber, isObject, isArray, isEmail, isUrl, etc.

### Async Module (`modules/async.ts`)
Operaciones asíncronas y promesas:

```ts
import { delay, retry, debounce, throttle } from '@/app/continuous-agent/utils/modules/async';

// O importación por namespace
import { Async, Promise, Performance } from '@/app/continuous-agent/utils/modules/async';

await Async.delay(1000);
const result = await Promise.retry(() => fetchData());
```

**Funciones incluidas:**
- **Async**: delay, retry, timeout, waitFor, sleep
- **Promise**: allSettled, promiseRace, promiseAny, map, filter, reduce
- **Performance**: throttle, requestIdleCallback, measurePerformance
- **Debounce**: debounce function

### Format Module (`modules/format.ts`)
Formateo de fechas, números y texto:

```ts
import { formatDate, formatNumber, formatFrequency } from '@/app/continuous-agent/utils/modules/format';

// O importación por namespace
import { Date, Formatting, Formatters } from '@/app/continuous-agent/utils/modules/format';

const formatted = Date.formatDateTime(new Date());
const credits = Formatting.formatCredits(100);
```

**Funciones incluidas:**
- **Date**: formatRelativeTime, formatDateTime, formatDate, formatTime
- **Formatting**: formatNumber, formatCredits, getCreditsStatusClass
- **Formatters**: formatFrequency, formatJSONError, getJSONErrorPosition

### Validation Module (`modules/validation.ts`)
Validación de entrada y manejo de errores:

```ts
import { validateName, validateJSON, getApiErrorMessage } from '@/app/continuous-agent/utils/modules/validation';

// O importación por namespace
import { Validation, ApiError } from '@/app/continuous-agent/utils/modules/validation';

const isValid = Validation.validateName("Agent Name");
const error = ApiError.getApiErrorMessage(err);
```

**Funciones incluidas:**
- **Validation**: validateName, validateDescription, validateFrequency, validateJSON, parseJSON
- **ApiError**: getApiErrorMessage, handleApiError

### Storage Module (`modules/storage.ts`)
Utilidades de almacenamiento local:

```ts
import { getStorageItem, setStorageItem } from '@/app/continuous-agent/utils/modules/storage';

// O importación por namespace
import { Storage } from '@/app/continuous-agent/utils/modules/storage';

Storage.setStorageItem("key", value);
const value = Storage.getStorageItem("key");
```

**Funciones incluidas:**
- getStorageItem, setStorageItem, removeStorageItem
- clearStorage, getStorageKeys, hasStorageItem, getStorageSize

### UI Module (`modules/ui.ts`)
Utilidades de interfaz de usuario:

```ts
import { cn } from '@/app/continuous-agent/utils/modules/ui';
```

**Funciones incluidas:**
- cn: Utility para combinar clases CSS (similar a clsx)

## 📦 Patrones de Importación

### 1. Importación Directa (Recomendado para funciones individuales)
```ts
import { unique, deepMerge, camelCase } from '@/app/continuous-agent/utils/modules/core';
import { delay, retry } from '@/app/continuous-agent/utils/modules/async';
```

### 2. Importación por Namespace (Recomendado para múltiples funciones)
```ts
import { Array, Object, String } from '@/app/continuous-agent/utils/modules/core';
import { Async, Promise } from '@/app/continuous-agent/utils/modules/async';

const unique = Array.unique([1, 2, 2]);
const merged = Object.deepMerge({ a: 1 }, { b: 2 });
```

### 3. Importación desde el Index Principal
```ts
import { Array, Object, Async, Format } from '@/app/continuous-agent/utils/modules';
```

## 🚀 Beneficios

1. **Mejor Tree-shaking**: Solo importas lo que necesitas
2. **Carga más rápida**: Módulos más pequeños se cargan más rápido
3. **Organización clara**: Fácil encontrar dónde está cada función
4. **Autocomplete mejorado**: Mejor experiencia de desarrollo
5. **Mantenibilidad**: Código más fácil de mantener y entender

## 📝 Ejemplos de Uso

### Ejemplo 1: Manipulación de Arrays
```ts
import { Array } from '@/app/continuous-agent/utils/modules/core';

const agents = [...];
const grouped = Array.groupBy(agents, (a) => a.status);
const unique = Array.unique([1, 2, 2, 3]);
const sorted = Array.sortBy(agents, (a) => a.createdAt);
```

### Ejemplo 2: Operaciones Asíncronas
```ts
import { Async, Promise } from '@/app/continuous-agent/utils/modules/async';

await Async.delay(1000);
const result = await Promise.retry(
  () => fetchData(),
  { maxAttempts: 3, delayMs: 1000 }
);
```

### Ejemplo 3: Formateo
```ts
import { Date, Formatting } from '@/app/continuous-agent/utils/modules/format';

const formatted = Date.formatDateTime(new Date());
const credits = Formatting.formatCredits(100);
```

### Ejemplo 4: Validación
```ts
import { Validation, ApiError } from '@/app/continuous-agent/utils/modules/validation';

try {
  Validation.validateName(name);
} catch (error) {
  const message = ApiError.getApiErrorMessage(error);
}
```

## 🔄 Migración desde el Index Principal

Si estás usando el index principal (`utils/index.ts`), puedes migrar gradualmente:

```ts
// Antes
import { unique, delay, formatDate } from '@/app/continuous-agent/utils';

// Después
import { unique } from '@/app/continuous-agent/utils/modules/core';
import { delay } from '@/app/continuous-agent/utils/modules/async';
import { formatDate } from '@/app/continuous-agent/utils/modules/format';
```

El index principal sigue funcionando, pero la estructura modular ofrece mejores beneficios de tree-shaking.





