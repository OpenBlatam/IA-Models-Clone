# Mejoras del Frontend - Versión 5

## 📋 Resumen

Esta versión incluye hooks adicionales para interacciones, componentes de UI básicos mejorados, y utilidades para animaciones y clases.

## ✨ Nuevas Funcionalidades

### 1. Hooks de Interacción

#### `useDragAndDrop`
Hook para implementar drag and drop.

```typescript
const { elementRef, isDragging, handleMouseDown } = useDragAndDrop({
  onDragStart: (pos) => console.log('Drag started', pos),
  onDrag: (pos) => console.log('Dragging', pos),
  onDragEnd: (pos) => console.log('Drag ended', pos),
});

<div ref={elementRef} onMouseDown={handleMouseDown}>
  Draggable element
</div>
```

#### `useHover`
Hook para detectar cuando el mouse está sobre un elemento.

```typescript
const [ref, isHovered] = useHover<HTMLButtonElement>();

<button ref={ref} className={isHovered ? 'bg-blue-500' : 'bg-gray-500'}>
  Hover me
</button>
```

#### `useToggle`
Hook para alternar un valor booleano.

```typescript
const [isOpen, toggle, setToggle] = useToggle(false);

<button onClick={toggle}>Toggle</button>
<button onClick={() => setToggle(true)}>Open</button>
```

#### `useCountdown`
Hook para crear un contador regresivo.

```typescript
const { seconds, isRunning, start, pause, reset, restart } = useCountdown({
  initialSeconds: 60,
  onComplete: () => console.log('Countdown finished'),
  autoStart: false,
});
```

#### `useIdle`
Hook para detectar cuando el usuario está inactivo.

```typescript
const isIdle = useIdle({
  timeout: 30000, // 30 seconds
  events: ['mousedown', 'keypress'],
});

{isIdle && <IdleMessage />}
```

### 2. Componentes de UI Básicos

#### `ProgressBar`
Barra de progreso con múltiples variantes.

```tsx
<ProgressBar
  value={75}
  showLabel
  size="lg"
  color="primary"
  animated
  striped
/>
```

**Características:**
- Tamaños: sm, md, lg
- Colores: primary, success, warning, danger
- Animaciones opcionales
- Modo striped
- Accesible (ARIA)

#### `Skeleton`
Componentes de skeleton para estados de carga.

```tsx
// Skeleton básico
<Skeleton width="200px" height="20px" rounded />

// Skeleton de texto
<SkeletonText lines={3} />

// Skeleton de card
<SkeletonCard />
```

**Variantes:**
- `Skeleton` - Skeleton básico personalizable
- `SkeletonText` - Múltiples líneas de texto
- `SkeletonCard` - Card completo con skeleton

#### `Badge`
Badge para etiquetas y estados.

```tsx
<Badge variant="success" size="md" rounded>
  Active
</Badge>
```

**Variantes:**
- default, primary, success, warning, danger, info
- Tamaños: sm, md, lg
- Opción rounded

#### `Divider`
Separador horizontal o vertical.

```tsx
// Horizontal simple
<Divider />

// Con texto
<Divider text="OR" />

// Vertical
<Divider orientation="vertical" />
```

### 3. Utilidades Adicionales

#### `animations.ts`
Utilidades para animaciones predefinidas.

```typescript
import { animations, getTransitionClasses } from '@/lib/utils/animations';

// Usar animaciones predefinidas
<div className={animations.fadeIn}>Content</div>

// Obtener clases de transición
const classes = getTransitionClasses('fade', 'normal');
```

**Animaciones disponibles:**
- fadeIn, fadeOut
- slideInUp, slideInDown, slideInLeft, slideInRight
- zoomIn, zoomOut
- spin, pulse, bounce, ping

#### `classNames.ts`
Utilidades mejoradas para manejo de clases.

```typescript
import { cn, classNames, mergeClasses } from '@/lib/utils/classNames';

// cn - merge con resolución de conflictos Tailwind
const classes = cn('px-4', isActive && 'bg-blue-500', className);

// classNames - simple conditional
const classes = classNames('base-class', isActive && 'active', null);

// mergeClasses - alias de cn
const classes = mergeClasses('px-4', 'py-2');
```

## 🎯 Ejemplos de Uso

### Drag and Drop

```tsx
import { useDragAndDrop } from '@/lib/hooks';

function DraggableCard() {
  const { elementRef, isDragging, handleMouseDown } = useDragAndDrop({
    onDragEnd: (pos) => {
      console.log('Dropped at', pos);
    },
  });

  return (
    <div
      ref={elementRef}
      onMouseDown={handleMouseDown}
      className={`cursor-move ${isDragging ? 'opacity-50' : ''}`}
    >
      Drag me
    </div>
  );
}
```

### Countdown Timer

```tsx
import { useCountdown } from '@/lib/hooks';

function Timer() {
  const { seconds, isRunning, start, pause, reset } = useCountdown({
    initialSeconds: 60,
    onComplete: () => alert('Time up!'),
  });

  return (
    <div>
      <div>{Math.floor(seconds / 60)}:{(seconds % 60).toString().padStart(2, '0')}</div>
      <button onClick={start} disabled={isRunning}>Start</button>
      <button onClick={pause} disabled={!isRunning}>Pause</button>
      <button onClick={reset}>Reset</button>
    </div>
  );
}
```

### Progress Bar con Estados

```tsx
import { ProgressBar } from '@/lib/components';

function UploadProgress({ progress }: { progress: number }) {
  const getColor = () => {
    if (progress < 30) return 'danger';
    if (progress < 70) return 'warning';
    return 'success';
  };

  return (
    <ProgressBar
      value={progress}
      showLabel
      color={getColor()}
      animated
      size="lg"
    />
  );
}
```

### Skeleton Loading

```tsx
import { SkeletonCard } from '@/lib/components';

function LoadingCards() {
  return (
    <div className="grid grid-cols-3 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}
```

### Badges para Estados

```tsx
import { Badge } from '@/lib/components';

function StatusBadges() {
  return (
    <div className="flex gap-2">
      <Badge variant="success">Active</Badge>
      <Badge variant="warning">Pending</Badge>
      <Badge variant="danger">Error</Badge>
      <Badge variant="info" rounded>New</Badge>
    </div>
  );
}
```

### Detección de Inactividad

```tsx
import { useIdle } from '@/lib/hooks';

function App() {
  const isIdle = useIdle({ timeout: 300000 }); // 5 minutes

  return (
    <>
      {isIdle && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded">
            <p>You've been idle. Stay on page?</p>
            <button>Yes</button>
            <button>No</button>
          </div>
        </div>
      )}
      {/* Rest of app */}
    </>
  );
}
```

## 📦 Archivos Creados

**Hooks:**
- `lib/hooks/useDragAndDrop.ts`
- `lib/hooks/useHover.ts`
- `lib/hooks/useToggle.ts`
- `lib/hooks/useCountdown.ts`
- `lib/hooks/useIdle.ts`

**Componentes:**
- `lib/components/ProgressBar.tsx`
- `lib/components/Skeleton.tsx`
- `lib/components/Badge.tsx`
- `lib/components/Divider.tsx`

**Utilidades:**
- `lib/utils/animations.ts`
- `lib/utils/classNames.ts`

## 🎨 Características Destacadas

### ProgressBar
- ✅ Múltiples variantes de color
- ✅ Tamaños configurables
- ✅ Animaciones opcionales
- ✅ Modo striped
- ✅ Accesible

### Skeleton
- ✅ Múltiples variantes
- ✅ Animación pulse
- ✅ Fácil de personalizar
- ✅ Componentes predefinidos

### Badge
- ✅ Variantes de color
- ✅ Tamaños configurables
- ✅ Opción rounded
- ✅ Estilos consistentes

## 🚀 Beneficios

1. **Mejor UX:**
   - Estados de carga claros con Skeleton
   - Feedback visual con ProgressBar
   - Indicadores de estado con Badge

2. **Interactividad:**
   - Drag and drop fácil de implementar
   - Detección de hover
   - Contadores y timers

3. **Productividad:**
   - Hooks reutilizables
   - Componentes listos para usar
   - Utilidades de animación

4. **Consistencia:**
   - Componentes con estilos consistentes
   - Utilidades para clases
   - Animaciones predefinidas

## 📚 Documentación

- Ver `lib/hooks/index.ts` para todos los hooks
- Ver `lib/components/index.ts` para todos los componentes
- Ver `lib/utils/index.ts` para todas las utilidades

## 🔄 Resumen de Versiones

### Versión 2
- Hooks básicos (retry, debounce, throttle, etc.)
- Utilidades (errorHandler, format, validation)
- Componentes básicos (ErrorBoundary, LoadingState)

### Versión 3
- Hooks de rendimiento (performance, intersection, image loader)
- Componentes optimizados (LazyImage, AnimatedSection)
- Optimizaciones Next.js

### Versión 4
- Hooks de interacción (clickOutside, mediaQuery, clipboard)
- Componentes avanzados (Modal, Tooltip, Confetti)
- Integración en layout

### Versión 5
- Hooks de interacción avanzados (drag, hover, toggle, countdown, idle)
- Componentes de UI básicos (ProgressBar, Skeleton, Badge, Divider)
- Utilidades de animación y clases

## 📊 Estadísticas Totales

- **Total de hooks:** 23
- **Total de componentes:** 16
- **Total de utilidades:** 7 módulos
- **Archivos creados:** 40+
- **Líneas de código:** 3000+



