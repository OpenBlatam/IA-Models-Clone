# Arquitectura del Frontend

## 📁 Estructura de Directorios

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # Layout principal
│   ├── page.tsx            # Página de inicio
│   ├── providers.tsx       # Providers globales
│   ├── music/              # Páginas de Music Analyzer
│   │   ├── components/     # Componentes específicos de música
│   │   ├── hooks/          # Hooks específicos de música
│   │   └── page.tsx        # Página principal de música
│   └── robot/              # Páginas de Robot Movement
├── components/             # Componentes React reutilizables
│   ├── ui/                 # Componentes UI base (barrel export)
│   ├── music/              # Componentes específicos de música
│   ├── robot/              # Componentes específicos de robot
│   ├── error-boundary.tsx  # Error boundary global
│   └── Navigation.tsx      # Navegación principal
├── lib/                    # Utilidades y servicios
│   ├── api/                # Servicios API
│   │   ├── client.ts       # Cliente Axios configurado
│   │   ├── tracks.ts        # Endpoints de tracks
│   │   ├── comparison.ts   # Endpoints de comparación
│   │   ├── recommendations.ts # Endpoints de recomendaciones
│   │   ├── favorites.ts    # Endpoints de favoritos
│   │   ├── types.ts        # Tipos TypeScript con Zod
│   │   └── index.ts        # Exportaciones principales
│   ├── config/             # Configuración
│   │   ├── env.ts          # Variables de entorno
│   │   └── app.ts          # Configuración de la aplicación
│   ├── constants/          # Constantes
│   │   └── index.ts        # Todas las constantes
│   ├── hooks/              # Hooks compartidos
│   │   ├── use-debounce.ts
│   │   ├── use-local-storage.ts
│   │   ├── use-media-query.ts
│   │   └── index.ts
│   ├── store/              # Zustand stores
│   │   ├── music-store.ts  # Store de música
│   │   └── index.ts
│   ├── types/              # Tipos TypeScript compartidos
│   │   ├── common.ts
│   │   └── index.ts
│   ├── validations/        # Esquemas Zod
│   │   ├── music.ts
│   │   └── index.ts
│   ├── errors.ts           # Tipos de error personalizados
│   └── utils.ts            # Utilidades generales
├── middleware.ts           # Middleware de Next.js
├── __tests__/              # Tests
└── public/                 # Archivos estáticos
```

## 🏗️ Principios Arquitectónicos

### 1. Separación de Responsabilidades

- **App Router**: Páginas y layouts (Next.js 14)
- **Components**: Componentes UI reutilizables
- **Lib**: Lógica de negocio, servicios, utilidades
- **Store**: Estado global (Zustand)
- **Hooks**: Lógica reutilizable de componentes

### 2. Organización por Features

Cada feature (music, robot) tiene su propia carpeta con:
- Componentes específicos
- Hooks específicos
- Páginas relacionadas

### 3. Barrel Exports

Uso de `index.ts` para exportaciones centralizadas:
```typescript
// En lugar de:
import { useDebounce } from '@/lib/hooks/use-debounce';

// Usar:
import { useDebounce } from '@/lib/hooks';
```

## 🔧 Configuración

### Variables de Entorno

Las variables de entorno se acceden a través de `lib/config/env.ts`:
```typescript
import { env } from '@/lib/config/env';

const apiUrl = env.MUSIC_API_URL;
```

### Configuración de la Aplicación

Configuración unificada en `lib/config/app.ts`:
```typescript
import { appConfig, uiConfig, apiConfig, performanceConfig } from '@/lib/config';

// Acceso unificado
const toastDuration = appConfig.ui.toast.duration;
const apiTimeout = appConfig.api.music.timeout;

// O acceso directo (backward compatibility)
const toastPosition = uiConfig.toast.position;
```

**Estructura de configuración:**
- `appConfig`: Configuración principal unificada
  - `appConfig.ui`: Configuración de UI (toast, theme, pagination)
  - `appConfig.api`: Configuración de APIs (music, robot)
  - `appConfig.performance`: Configuración de rendimiento (debounce, cache)
- `uiConfig`: Exportación directa de configuración UI
- `apiConfig`: Exportación directa de configuración API
- `performanceConfig`: Exportación directa de configuración de rendimiento

**Tipos TypeScript:**
```typescript
import type {
  AppConfigType,
  UIConfig,
  APIConfig,
  PerformanceConfig,
} from '@/lib/config';
```

### Constantes

Todas las constantes están en `lib/constants/index.ts`:
```typescript
import { QUERY_KEYS, ROUTES, PAGINATION } from '@/lib/constants';
```

## 📦 Gestión de Estado

### Zustand Stores

Para estado global que necesita persistencia:
```typescript
import { useMusicStore } from '@/lib/store';

const { currentTrack, setCurrentTrack } = useMusicStore();
```

### React Query

Para estado del servidor y caché:
```typescript
import { useQuery } from '@tanstack/react-query';
import { QUERY_KEYS } from '@/lib/constants';

const { data } = useQuery({
  queryKey: QUERY_KEYS.MUSIC.SEARCH(query),
  queryFn: () => searchTracks(query),
});
```

## 🎣 Hooks Personalizados

### Hooks Compartidos

**Hooks Básicos:**
- `useDebounce`: Para debounce de valores
- `useLocalStorage`: Para localStorage con type safety
- `useMediaQuery`: Para responsive design
- `useFormValidation`: Para validación de formularios
- `useApiHealth`: Para monitoreo de salud de API

**Hooks de Optimización:**
- `useMemoizedValue`: Memoización con igualdad personalizada
- `useStableCallback`: Callback estable con mejor inferencia
- `useDebouncedCallback`: Callback con debounce
- `useThrottledCallback`: Callback con throttle
- `useRenderCount`: Contador de renders para debugging

**Hooks Avanzados:**
- `useReactQuery`: Wrappers para React Query con error handling
- `useSafeAction`: Para acciones seguras con manejo de errores
- `usePerformanceMonitor`: Monitoreo de rendimiento
- `useCircuitBreaker`: Circuit breaker pattern
- `useRetryAdvanced`: Retry avanzado con estrategias

### Hooks por Feature

Los hooks específicos de cada feature van en su carpeta:
- `app/music/hooks/use-music-state.ts`

## ✅ Validación

### Zod Schemas

Validación en tiempo de ejecución con Zod:
```typescript
import { searchQuerySchema } from '@/lib/validations';

const result = searchQuerySchema.parse(query);
```

### Schemas Comunes

Schemas reutilizables disponibles:
```typescript
import {
  emailSchema,
  urlSchema,
  nonEmptyStringSchema,
  positiveIntegerSchema,
  stringWithLengthRange,
  numberWithRange,
  arrayWithMinLength,
} from '@/lib/validations/common';

// Validar email
const email = emailSchema.parse(userEmail);

// Validar URL
const url = urlSchema.parse(userUrl);

// String con rango
const name = stringWithLengthRange(3, 50).parse(userName);

// Número con rango
const age = numberWithRange(0, 120).parse(userAge);

// Array con longitud mínima
const tags = arrayWithMinLength(z.string(), 1).parse(userTags);
```

## 🚨 Manejo de Errores

### Error Boundaries

Componentes envueltos en error boundaries:
```typescript
<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

### Custom Errors

Tipos de error personalizados:
- `ApiError`: Errores de API
- `NetworkError`: Errores de red
- `ValidationError`: Errores de validación

### Manejo Avanzado de Errores

Utilidades avanzadas para manejo de errores:

```typescript
import {
  handleError,
  createErrorHandler,
  withErrorHandling,
  retryWithBackoff,
  recoverFromError,
  ErrorSeverity,
} from '@/lib/utils/error-handling-advanced';

// Manejo básico de errores
handleError(error, {
  showToast: true,
  logError: true,
  context: { component: 'MyComponent' },
});

// Crear handler personalizado
const errorHandler = createErrorHandler({
  showToast: true,
  reportError: true,
});

// Wrapper para funciones async
const safeFunction = withErrorHandling(asyncFunction, {
  context: { action: 'fetchData' },
});

// Retry con backoff exponencial
const result = await retryWithBackoff(
  () => fetchData(),
  { maxRetries: 3, initialDelay: 1000 }
);

// Recuperación de errores
const recovered = await recoverFromError(
  error,
  RecoveryStrategy.RETRY,
  { retryFn: () => fetchData() }
);
```

### Hooks de Manejo de Errores

```typescript
import { useErrorHandler } from '@/lib/hooks';

const handleError = useErrorHandler({
  showToast: true,
  context: { component: 'MyComponent' },
});
```

## 🎨 Componentes

### Estructura de Componentes

```typescript
/**
 * Component description.
 * @param props - Component props
 */
export function Component({ prop1, prop2 }: ComponentProps) {
  // Hooks
  // State
  // Effects
  // Handlers
  // Render
}
```

### Dynamic Imports

Para code splitting:
```typescript
const HeavyComponent = dynamic(() => import('./HeavyComponent'), {
  ssr: false,
  loading: () => <LoadingState />,
});
```

## ♿ Accesibilidad

### Utilidades de Accesibilidad

Utilidades para mejorar la accesibilidad:

```typescript
import {
  announceToScreenReader,
  trapFocus,
  getFocusableElements,
  validateAriaAttributes,
  meetsContrastRatio,
} from '@/lib/utils/accessibility';

// Anunciar a lectores de pantalla
announceToScreenReader('Formulario enviado exitosamente');

// Atrapar foco en modal
const cleanup = trapFocus(modalElement);

// Validar atributos ARIA
const { valid, errors } = validateAriaAttributes(element);

// Verificar contraste
const accessible = meetsContrastRatio('#000000', '#ffffff');
```

### Hooks de Accesibilidad

```typescript
import {
  useAnnounce,
  useFocusTrap,
  usePrefersReducedMotion,
  useKeyboardNavigation,
} from '@/lib/hooks';

// Anunciar mensajes
const announce = useAnnounce();
announce('Operación completada');

// Atrapar foco
const containerRef = useFocusTrap({ active: isOpen });

// Detectar preferencias
const prefersReduced = usePrefersReducedMotion();
const prefersHighContrast = usePrefersHighContrast();

// Navegación por teclado
const keyboardProps = useKeyboardNavigation({
  onEnter: () => handleSubmit(),
  onEscape: () => handleClose(),
});
```

## 🔒 Seguridad

### Headers de Seguridad

Configurados en `middleware.ts` y `next.config.js`:
- X-Frame-Options
- X-Content-Type-Options
- Referrer-Policy
- X-DNS-Prefetch-Control

### Sanitización Básica

Utilidades de sanitización disponibles:
```typescript
import {
  sanitizeString,
  sanitizeSearchQuery,
  sanitizeUrl,
  escapeHtml,
} from '@/lib/utils/sanitization';
```

### Seguridad Avanzada

Utilidades avanzadas de seguridad:

```typescript
import {
  sanitizeXss,
  sanitizeUrl,
  generateSecureToken,
  hashString,
  validateCSP,
  containsDangerousContent,
  validateFileType,
  validateFileSize,
  createSecureDownload,
  ClientRateLimiter,
} from '@/lib/utils/security-advanced';

// Sanitización XSS
const safe = sanitizeXss(userInput, { stripTags: true });

// Validar y sanitizar URLs
const safeUrl = sanitizeUrl(url, ['example.com', 'trusted-domain.com']);

// Generar tokens seguros
const token = generateSecureToken(32);

// Hash de strings
const hash = await hashString('password', 'SHA-256');

// Validar CSP
const { valid, errors } = validateCSP(cspHeader);

// Detectar contenido peligroso
if (containsDangerousContent(userInput)) {
  // Handle dangerous content
}

// Validar archivos
const isValid = validateFileType(filename, mimeType, ['jpg', 'png', 'pdf']);
const isValidSize = validateFileSize(fileSize, 5, 'MB');

// Rate limiting en cliente
const limiter = new ClientRateLimiter(60000, 10); // 10 requests per minute
if (limiter.isAllowed('user-action')) {
  // Proceed
}
```

### Validación de Archivos

```typescript
// Validar tipo de archivo
const isValidType = validateFileType(
  'document.pdf',
  'application/pdf',
  ['pdf', 'doc', 'docx']
);

// Validar tamaño
const isValidSize = validateFileSize(1024 * 1024, 5, 'MB'); // 5MB max

// Descarga segura
createSecureDownload(blob, 'document.pdf', 'application/pdf');
```

## ⚡ Optimizaciones

### Next.js Config

- Image optimization (WebP, AVIF)
- Package import optimization
- Console removal en producción
- Bundle size optimization

### Code Splitting

- Dynamic imports para componentes pesados
- Lazy loading de rutas
- Optimización de imports

### Utilidades de Optimización de Componentes

El proyecto incluye utilidades avanzadas para optimización de componentes React:

```typescript
import {
  memoizeComponent,
  lazyLoadComponent,
  conditionalRender,
  clientOnly,
} from '@/lib/utils/react-optimization';

// Memoizar componente con comparación personalizada
const OptimizedButton = memoizeComponent(Button, {
  displayName: 'OptimizedButton',
  areEqual: (prev, next) => prev.id === next.id,
});

// Lazy load con fallback
const HeavyComponent = lazyLoadComponent(
  () => import('./HeavyComponent'),
  { fallback: <Loading />, preload: true }
);

// Renderizado condicional
const ConditionalComponent = conditionalRender(
  (props) => props.isVisible,
  () => <div>Hidden</div>
)(MyComponent);

// Solo en cliente (evita problemas de SSR)
const ClientOnlyComponent = clientOnly(MyComponent, <div>Loading...</div>);
```

### Hooks de Optimización de Rendimiento

Hooks especializados para optimización de rendimiento:

```typescript
import {
  useMemoizedValue,
  useStableCallback,
  useDebouncedCallback,
  useThrottledCallback,
  useRenderCount,
} from '@/lib/hooks';

// Memoización con igualdad personalizada
const memoizedValue = useMemoizedValue(
  () => expensiveCalculation(data),
  [data],
  { equalityFn: deepEqual }
);

// Callback estable
const handleClick = useStableCallback((id: string) => {
  doSomething(id);
}, [dependency]);

// Debounce
const debouncedSearch = useDebouncedCallback(
  (query: string) => search(query),
  300,
  [],
  { leading: false, trailing: true }
);

// Throttle
const throttledScroll = useThrottledCallback(
  () => handleScroll(),
  100,
  [],
  { leading: true, trailing: true }
);

// Debugging de renders
const renderCount = useRenderCount('MyComponent');
```

### React Query Optimization

Configuración optimizada de React Query en `app/providers.tsx`:

- `structuralSharing`: Habilitado para mejor rendimiento
- Cache configurado con `performanceConfig`
- Retry logic inteligente (no reintenta errores 4xx)
- Network mode optimizado
- Error handling mejorado

## 📝 Convenciones

### Nombres de Archivos

- Componentes: `PascalCase.tsx`
- Hooks: `use-kebab-case.ts`
- Utilidades: `kebab-case.ts`
- Tipos: `kebab-case.ts`

## 🔷 Tipos TypeScript

### Tipos Comunes

Tipos utilitarios disponibles:

```typescript
import type {
  Result,
  Optional,
  NonNullable,
  DeepReadonly,
  DeepPartial,
  RequireFields,
  OptionalFields,
  Awaited,
  EventHandler,
  AsyncFunction,
} from '@/lib/types';

// Result type para operaciones que pueden fallar
type UserResult = Result<User, ApiError>;

// Optional type
type MaybeUser = Optional<User>;

// Deep readonly
type ReadonlyConfig = DeepReadonly<Config>;

// Deep partial
type PartialConfig = DeepPartial<Config>;

// Require specific fields
type UserWithEmail = RequireFields<User, 'email'>;

// Optional specific fields
type PartialUser = OptionalFields<User, 'password'>;

// Extract promise type
type UserData = Awaited<Promise<User>>;

// Event handler
const handleClick: EventHandler<MouseEvent> = (e) => {};

// Async function
const fetchUser: AsyncFunction<[string], User> = async (id) => {
  return await getUser(id);
};
```

### Imports

Orden de imports:
1. React/Next.js
2. Librerías de terceros
3. Componentes internos
4. Utilidades y tipos
5. Estilos

```typescript
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Button } from '@/components/ui';
import { useDebounce } from '@/lib/hooks';
import type { Track } from '@/lib/api/types';
```

## 🧪 Testing

Estructura de tests:
```
__tests__/
├── components/
├── lib/
└── utils.test.ts
```

## 📚 Recursos

- [Next.js Documentation](https://nextjs.org/docs)
- [React Query](https://tanstack.com/query/latest)
- [Zustand](https://github.com/pmndrs/zustand)
- [Zod](https://zod.dev/)

