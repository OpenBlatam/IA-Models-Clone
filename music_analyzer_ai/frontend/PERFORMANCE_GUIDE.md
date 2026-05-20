# Guía de Optimización de Rendimiento

Esta guía describe las mejores prácticas y utilidades disponibles para optimizar el rendimiento de la aplicación.

## 📊 Utilidades de Optimización de Componentes

### Memoización de Componentes

Use `memoizeComponent` para memoizar componentes y evitar re-renders innecesarios:

```typescript
import { memoizeComponent } from '@/lib/utils/react-optimization';

// Memoización básica
const OptimizedButton = memoizeComponent(Button);

// Con comparación personalizada
const OptimizedCard = memoizeComponent(Card, {
  displayName: 'OptimizedCard',
  areEqual: (prevProps, nextProps) => {
    return prevProps.id === nextProps.id && prevProps.title === nextProps.title;
  },
});
```

**Cuándo usar:**
- Componentes que reciben props que cambian frecuentemente
- Componentes que renderizan listas grandes
- Componentes con cálculos costosos

### Lazy Loading de Componentes

Use `lazyLoadComponent` para cargar componentes pesados de forma diferida:

```typescript
import { lazyLoadComponent } from '@/lib/utils/react-optimization';

// Lazy load básico
const HeavyComponent = lazyLoadComponent(
  () => import('./HeavyComponent')
);

// Con fallback y preload
const ChartComponent = lazyLoadComponent(
  () => import('./ChartComponent'),
  {
    fallback: <ChartSkeleton />,
    preload: true, // Preload en background
  }
);
```

**Cuándo usar:**
- Componentes que no son críticos para el render inicial
- Componentes pesados (gráficos, editores, visualizadores)
- Componentes que solo se muestran bajo ciertas condiciones

### Renderizado Condicional

Use `conditionalRender` para crear HOCs que renderizan condicionalmente:

```typescript
import { conditionalRender } from '@/lib/utils/react-optimization';

const ConditionalFeature = conditionalRender(
  (props) => props.user?.hasFeature,
  () => <UpgradePrompt />
)(FeatureComponent);
```

### Componentes Solo Cliente

Use `clientOnly` para componentes que solo deben renderizarse en el cliente:

```typescript
import { clientOnly } from '@/lib/utils/react-optimization';

const ClientOnlyChart = clientOnly(
  ChartComponent,
  <div>Loading chart...</div>
);
```

**Cuándo usar:**
- Componentes que usan APIs del navegador no disponibles en SSR
- Componentes que causan problemas de hidratación
- Componentes que dependen de `window` o `document`

## 🎣 Hooks de Optimización

### useMemoizedValue

Memoización avanzada con función de igualdad personalizada:

```typescript
import { useMemoizedValue } from '@/lib/hooks';

// Con igualdad personalizada
const expensiveResult = useMemoizedValue(
  () => complexCalculation(data),
  [data],
  {
    equalityFn: (a, b) => {
      // Comparación profunda personalizada
      return JSON.stringify(a) === JSON.stringify(b);
    },
  }
);
```

### useStableCallback

Callbacks estables que solo cambian cuando las dependencias cambian:

```typescript
import { useStableCallback } from '@/lib/hooks';

const handleSubmit = useStableCallback(
  (formData: FormData) => {
    submitForm(formData);
  },
  [userId] // Solo recrea si userId cambia
);
```

### useDebouncedCallback

Callbacks con debounce para limitar la frecuencia de ejecución:

```typescript
import { useDebouncedCallback } from '@/lib/hooks';

const debouncedSearch = useDebouncedCallback(
  (query: string) => {
    performSearch(query);
  },
  300, // 300ms de delay
  [], // Dependencias
  {
    leading: false, // No ejecutar en el primer call
    trailing: true, // Ejecutar después del delay
  }
);

// Uso
<input onChange={(e) => debouncedSearch(e.target.value)} />
```

**Cuándo usar:**
- Búsquedas en tiempo real
- Validación de formularios
- Filtros que se actualizan mientras el usuario escribe

### useThrottledCallback

Callbacks con throttle para limitar la frecuencia máxima:

```typescript
import { useThrottledCallback } from '@/lib/hooks';

const throttledScroll = useThrottledCallback(
  () => {
    updateScrollPosition();
  },
  100, // Máximo cada 100ms
  [],
  {
    leading: true, // Ejecutar inmediatamente
    trailing: true, // Ejecutar al final
  }
);

// Uso
<div onScroll={throttledScroll}>...</div>
```

**Cuándo usar:**
- Eventos de scroll
- Eventos de resize
- Eventos de mouse move
- Cualquier evento que se dispara muy frecuentemente

### useRenderCount

Hook de debugging para contar renders:

```typescript
import { useRenderCount } from '@/lib/hooks';

function MyComponent() {
  const renderCount = useRenderCount('MyComponent');
  
  // En desarrollo, verás en consola:
  // [MyComponent] Render count: 1
  // [MyComponent] Render count: 2
  // ...
  
  return <div>...</div>;
}
```

## ⚡ Optimizaciones de React Query

### Configuración Optimizada

El `QueryClient` está configurado con optimizaciones en `app/providers.tsx`:

```typescript
// Ya configurado automáticamente:
{
  staleTime: 60000, // 1 minuto
  gcTime: 300000, // 5 minutos
  structuralSharing: true, // Compartir estructura de datos
  refetchOnWindowFocus: false, // No refetch al cambiar de ventana
}
```

### Mejores Prácticas

1. **Usar Query Keys consistentes:**
```typescript
import { QUERY_KEYS } from '@/lib/constants';

useQuery({
  queryKey: QUERY_KEYS.MUSIC.SEARCH(query),
  queryFn: () => searchTracks(query),
});
```

2. **Usar select para transformar datos:**
```typescript
const { data: trackNames } = useQuery({
  queryKey: QUERY_KEYS.MUSIC.TRACKS,
  queryFn: fetchTracks,
  select: (data) => data.map(track => track.name),
});
```

3. **Usar enabled para queries condicionales:**
```typescript
const { data } = useQuery({
  queryKey: QUERY_KEYS.MUSIC.TRACK(id),
  queryFn: () => fetchTrack(id),
  enabled: !!id, // Solo ejecutar si id existe
});
```

## 🎯 Optimizaciones de Next.js

### Dynamic Imports

Use dynamic imports para code splitting:

```typescript
import dynamic from 'next/dynamic';

const HeavyComponent = dynamic(() => import('./HeavyComponent'), {
  ssr: false, // No renderizar en servidor
  loading: () => <Skeleton />, // Componente de carga
});
```

### Image Optimization

Use el componente `Image` de Next.js:

```typescript
import Image from 'next/image';

<Image
  src="/image.jpg"
  alt="Description"
  width={500}
  height={300}
  priority // Para imágenes above the fold
  placeholder="blur" // Blur placeholder
/>
```

### Font Optimization

Las fuentes están optimizadas en `app/layout.tsx`:

```typescript
const inter = Inter({
  subsets: ['latin'],
  display: 'swap', // No bloquea render
  preload: true, // Precargar
});
```

## 📈 Monitoreo de Rendimiento

### usePerformanceMonitor

Monitoreo de rendimiento de componentes:

```typescript
import { usePerformanceMonitor } from '@/lib/hooks';

function MyComponent() {
  usePerformanceMonitor('MyComponent', {
    onRender: (duration) => {
      if (duration > 100) {
        console.warn('Slow render detected:', duration);
      }
    },
  });
  
  return <div>...</div>;
}
```

## 🔍 Debugging

### React Query DevTools

En desarrollo, React Query DevTools está disponible automáticamente.

### Render Count

Use `useRenderCount` para identificar componentes que se renderizan demasiado:

```typescript
const renderCount = useRenderCount('MyComponent');
// Revisa la consola para ver cuántas veces se renderiza
```

## 📝 Checklist de Optimización

- [ ] Componentes pesados memoizados con `memoizeComponent`
- [ ] Componentes no críticos con lazy loading
- [ ] Callbacks con debounce/throttle cuando sea necesario
- [ ] Queries de React Query con `select` para transformaciones
- [ ] Dynamic imports para code splitting
- [ ] Imágenes optimizadas con Next.js Image
- [ ] Render count monitoreado en desarrollo
- [ ] Structural sharing habilitado en React Query
- [ ] Cache configurado apropiadamente

## 🚀 Recursos Adicionales

- [React Performance Optimization](https://react.dev/learn/render-and-commit)
- [Next.js Performance](https://nextjs.org/docs/app/building-your-application/optimizing)
- [React Query Performance](https://tanstack.com/query/latest/docs/react/guides/performance)




