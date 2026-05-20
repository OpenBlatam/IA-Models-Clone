# Continuous Agent Module

Módulo completo para la gestión de agentes continuos que funcionan 24/7 en el servidor.

## Estructura del Módulo

```
continuous-agent/
├── api/                    # Next.js API routes
│   ├── [agentId]/
│   │   ├── logs/
│   │   │   └── route.ts
│   │   └── route.ts
│   └── route.ts
├── components/            # Componentes React
│   ├── ui/                # Componentes UI reutilizables
│   │   ├── FormField.tsx
│   │   └── Modal.tsx
│   ├── agent/             # Componentes específicos de agente
│   │   ├── AgentCredits.tsx
│   │   ├── AgentStats.tsx
│   │   └── AgentGoal.tsx  # Visualización de goal/prompt
│   ├── forms/             # Componentes de formulario
│   │   ├── AgentNameField.tsx
│   │   ├── AgentDescriptionField.tsx
│   │   ├── AgentTaskTypeField.tsx
│   │   ├── AgentFrequencyField.tsx
│   │   ├── AgentParametersField.tsx
│   │   ├── AgentGoalField.tsx # Campo para goal/prompt
│   │   └── FormFooter.tsx
│   ├── AgentCard.tsx      # Tarjeta de agente individual
│   ├── AgentDashboard.tsx # Dashboard de estadísticas
│   ├── CreateAgentModal.tsx # Modal de creación
│   ├── ToggleSwitch.tsx   # Switch de activación
│   └── index.ts           # Barrel exports
├── constants/             # Constantes y configuración
│   ├── config.ts          # Configuración del módulo
│   ├── messages.ts        # Mensajes de UI
│   ├── prompt-templates.ts # Plantillas de prompts estilo Perplexity
│   └── index.ts           # Barrel exports
├── hooks/                 # Custom React hooks
│   ├── useAgentForm.ts    # Hook para formulario de agente
│   ├── useContinuousAgent.ts # Hook para agente individual
│   ├── useContinuousAgents.ts # Hook para lista de agentes
│   └── index.ts           # Barrel exports
├── services/              # Servicios y API clients
│   └── agentService.ts    # Cliente de API para agentes
├── types/                 # Definiciones de tipos TypeScript
│   └── index.ts           # Todos los tipos del módulo
├── utils/                 # Utilidades y helpers
│   ├── modules/           # Estructura modular (recomendado)
│   │   ├── core.ts        # Utilidades básicas (array, object, string, type)
│   │   ├── async.ts       # Operaciones asíncronas (async, promise, performance)
│   │   ├── format.ts      # Formateo (date, formatting, formatters)
│   │   ├── validation.ts  # Validación y manejo de errores
│   │   ├── storage.ts     # Almacenamiento local
│   │   ├── ui.ts          # Utilidades de UI
│   │   ├── index.ts       # Barrel exports modular
│   │   └── README.md      # Guía de uso modular
│   ├── apiError.ts        # Manejo de errores de API
│   ├── classNames.ts      # Utilidad para clases CSS
│   ├── dateUtils.ts       # Utilidades de fechas
│   ├── formatting.ts      # Formateo de números y créditos
│   ├── validation.ts      # Validación de formularios
│   └── index.ts           # Barrel exports (compatible)
├── page.tsx               # Página principal del módulo
└── README.md              # Este archivo
```

## Características

### Estructura Modular de Utilidades

El módulo incluye una estructura modular de utilidades (`utils/modules/`) que permite:

- **Mejor tree-shaking**: Solo importas lo que necesitas
- **Organización clara**: Utilidades agrupadas por categoría
- **Importación flexible**: Por namespace o funciones individuales
- **Compatibilidad**: El index principal sigue funcionando

Ver `utils/modules/README.md` para más detalles.

### Componentes

- **AgentCard**: Muestra información de un agente individual con controles de activación/desactivación
- **AgentDashboard**: Dashboard con estadísticas generales de todos los agentes
- **CreateAgentModal**: Modal para crear nuevos agentes con validación completa
- **AgentGoalField**: Campo especializado para gestionar prompts/objetivos estilo Perplexity
- **ToggleSwitch**: Componente reutilizable de switch con accesibilidad completa
- **FormField**: Campo de formulario reutilizable con manejo de errores
- **Modal**: Modal genérico reutilizable con manejo de teclado

### Hooks

- **useAgentForm**: Gestiona el estado y validación del formulario de creación
- **useContinuousAgent**: Hook para gestionar un agente individual con auto-refresh
- **useContinuousAgents**: Hook para gestionar la lista de agentes con CRUD completo

### Servicios

- **agentService**: Cliente de API con manejo de errores mejorado y type-safety

### Utilidades

#### Core Utilities
- **dateUtils**: Formateo de fechas usando APIs nativas e `date-fns`
- **formatting**: Formateo de números, créditos y estados
- **validation**: Validación de formularios (frecuencia, JSON, campos requeridos)
- **apiError**: Manejo centralizado de errores de API
- **classNames**: Utilidad para combinar clases Tailwind con `clsx` y `tailwind-merge`
- **debounce**: Función debounce para optimizar eventos
- **formatters**: Formateo de frecuencia, errores JSON y posiciones

#### Async Utilities
- **async**: Utilidades para operaciones asíncronas (delay, retry, timeout, waitFor)
- **promise**: Utilidades para promesas (allSettled, race, retry, map, filter, reduce)

#### Data Manipulation
- **array**: Manipulación de arrays (chunk, groupBy, unique, flatten, partition, sortBy, zip, intersection, etc.)
  - Utiliza `lodash-es` para funciones optimizadas con tree-shaking
- **object**: Manipulación de objetos (pick, omit, deepMerge, mapValues, mapKeys, invert, compact, etc.)
  - Utiliza `lodash-es` para funciones optimizadas con tree-shaking
- **string**: Manipulación de strings (truncate, capitalize, camelCase, kebabCase, slugify, escapeHtml, etc.)
  - Utiliza `lodash-es` para funciones optimizadas con tree-shaking

#### Type Utilities
- **typeGuards**: Type guards para TypeScript (isString, isNumber, isObject, isArray, isEmail, isUrl, etc.)
  - Utiliza `lodash-es` para type guards optimizados

#### Storage Utilities
- **storage**: Utilidades para localStorage y sessionStorage (getItem, setItem, removeItem, clear, etc.)

#### Math Utilities
- **math**: Operaciones matemáticas (clamp, lerp, mapRange, round, random, percentage, average, sum, min, max, median, distance, etc.)

#### URL Utilities
- **url**: Manipulación de URLs (parseUrl, buildUrl, parseQueryString, buildQueryString, getQueryParam, setQueryParam, isValidUrl, getDomain, getPath, etc.)

#### Functional Utilities
- **functional**: Programación funcional (compose, pipe, curry, memoize, once, debounceFn, throttleFn, guardArgs, defaultArgs, identity, constant, noop)

#### Crypto Utilities
- **crypto**: Utilidades criptográficas (randomString, uuid, generateId, hashString, base64Encode, base64Decode, encodeBase64Url, decodeBase64Url, randomHex, randomBytes)

#### Collection Utilities
- **collection**: Operaciones avanzadas sobre colecciones (first, last, findIndex, findLast, shuffle, range, repeat, countBy, keyBy, sample, sampleSize, fromPairs, toPairs, isEmpty, size)

## Uso

### Hooks y Componentes

```typescript
import { useContinuousAgents } from "./hooks";
import { AgentCard, CreateAgentModal } from "./components";
import type { ContinuousAgent } from "./types";

// En tu componente
const { agents, createAgent, toggleAgent } = useContinuousAgents();
```

### Utilidades

#### Estructura Modular (Recomendado)

Para mejor tree-shaking y organización, usa la estructura modular:

```typescript
// Importación por módulo (recomendado para múltiples funciones)
import { Array, Object, Async } from "./utils/modules";

// Importación selectiva (recomendado para funciones individuales)
import { unique, deepMerge, delay } from "./utils/modules/core";
import { formatDate, formatNumber } from "./utils/modules/format";
```

Ver `utils/modules/README.md` para más información.

#### Async Operations
```typescript
// Desde el index principal (compatible)
import { delay, retry, timeout, waitFor } from "./utils";

// O desde módulos (recomendado)
import { delay, retry, timeout, waitFor } from "./utils/modules/async";

// Delay
await delay(1000); // Espera 1 segundo

// Retry con backoff exponencial
const data = await retry(
  () => fetchData(),
  { maxAttempts: 3, delayMs: 1000, backoffMultiplier: 2 }
);

// Timeout
const result = await timeout(fetchData(), 5000, "Request timed out");

// Wait for condition
await waitFor(() => document.readyState === "complete");
```

#### Array Manipulation
```typescript
// Desde el index principal (compatible)
import { chunk, groupBy, unique, sortBy } from "./utils";

// O desde módulos (recomendado)
import { Array } from "./utils/modules/core";
const chunks = Array.chunk([1, 2, 3, 4, 5], 2);
const grouped = Array.groupBy(agents, (agent) => agent.status);

const chunks = chunk([1, 2, 3, 4, 5], 2); // [[1, 2], [3, 4], [5]]
const grouped = groupBy(agents, (agent) => agent.status);
const uniqueIds = unique([1, 2, 2, 3, 3]); // [1, 2, 3]
const sorted = sortBy(agents, (agent) => agent.createdAt);
```

#### Object Manipulation
```typescript
// Desde el index principal (compatible)
import { pick, omit, deepMerge, mapValues } from "./utils";

// O desde módulos (recomendado)
import { Object } from "./utils/modules/core";
const publicUser = Object.pick(user, ["id", "name", "email"]);

const user = { id: 1, name: "John", email: "john@example.com", password: "secret" };
const publicUser = pick(user, ["id", "name", "email"]);
const safeUser = omit(user, ["password"]);
const merged = deepMerge({ a: 1 }, { b: 2 }, { a: 3 }); // { a: 3, b: 2 }
```

#### String Utilities
```typescript
// Desde el index principal (compatible)
import { truncate, slugify, capitalize, camelCase } from "./utils";

// O desde módulos (recomendado)
import { String } from "./utils/modules/core";
const slug = String.slugify("Hello World!");

truncate("Long text", 10); // "Long text..."
slugify("Hello World!"); // "hello-world"
capitalize("hello"); // "Hello"
camelCase("hello world"); // "helloWorld"
```

#### Type Guards
```typescript
import { isString, isNumber, isEmail, isUrl } from "./utils";

if (isString(value)) {
  // TypeScript sabe que value es string aquí
}
if (isEmail(email)) {
  // Validar email
}
```

#### Storage
```typescript
import { getStorageItem, setStorageItem, removeStorageItem } from "./utils";

setStorageItem("user", { id: 1, name: "John" });
const user = getStorageItem<{ id: number; name: string }>("user");
removeStorageItem("user");
```

#### Math Operations
```typescript
import { clamp, round, random, average, percentage } from "./utils";

clamp(150, 0, 100); // 100
round(3.14159, 2); // 3.14
random(1, 10); // Random number between 1 and 10
average([1, 2, 3, 4, 5]); // 3
percentage(25, 100); // 25
```

#### URL Manipulation
```typescript
import { parseUrl, buildUrl, getQueryParam, setQueryParam } from "./utils";

const url = buildUrl("/api/agents", { page: 1, limit: 10 });
const page = getQueryParam(url, "page"); // "1"
const newUrl = setQueryParam(url, "page", "2");
```

#### Functional Programming
```typescript
import { compose, pipe, curry, memoize } from "./utils";

const add = (x: number) => x + 1;
const multiply = (x: number) => x * 2;
const composed = compose(multiply, add);
composed(5); // 12

const piped = pipe(add, multiply);
piped(5); // 12

const curriedAdd = curry((a: number, b: number) => a + b);
curriedAdd(1)(2); // 3
```

#### Crypto & Random
```typescript
import { uuid, generateId, randomString, hashString } from "./utils";

uuid(); // "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx"
generateId("agent"); // "agent_abc123..."
randomString(16); // Random 16-character string
const hash = await hashString("password");
```

#### Collection Operations
```typescript
import { first, last, shuffle, range, countBy, sample } from "./utils";

first([1, 2, 3]); // 1
last([1, 2, 3]); // 3
shuffle([1, 2, 3]); // [2, 1, 3] (random)
range(5); // [0, 1, 2, 3, 4]
countBy(agents, (a) => a.status); // { active: 5, inactive: 2 }
sample([1, 2, 3]); // Random item
```

## Principios de Arquitectura

1. **Separación de Responsabilidades**: Cada carpeta tiene un propósito claro
2. **Barrel Exports**: Facilita imports y mantiene la API pública limpia
3. **Type Safety**: TypeScript estricto en todo el módulo
4. **Reutilización**: Componentes y hooks reutilizables
5. **Accesibilidad**: ARIA attributes y navegación por teclado en todos los componentes
6. **DRY**: Sin duplicación de código, utilidades centralizadas

## Constantes

Las constantes están organizadas en:
- **config.ts**: Configuración del módulo (intervalos, tipos de tareas, valores por defecto)
- **messages.ts**: Mensajes de UI, éxito y error (facilita i18n)
- **prompt-templates.ts**: Plantillas de prompts estilo Perplexity para diferentes casos de uso

## Prompts y Objetivos

El módulo soporta prompts estilo Perplexity para definir objetivos de agentes:

### Campo Goal

Los agentes pueden tener un campo `goal` opcional que define su prompt de sistema. Este campo:
- Es completamente opcional
- Soporta hasta 10,000 caracteres
- Incluye plantillas predefinidas
- Se valida en tiempo real

### Plantillas Disponibles

- **Perplexity Base**: Plantilla completa con todas las reglas de formato
- **Research Assistant**: Simplificada para investigación
- **Content Generator**: Para generación de contenido
- **Data Analyst**: Para análisis de datos
- **Custom**: Plantilla vacía para personalización

Ver `PROMPT_IMPROVEMENTS.md` para más detalles sobre la implementación.

## Tipos

Todos los tipos TypeScript están centralizados en `types/index.ts` para fácil mantenimiento y reutilización.



