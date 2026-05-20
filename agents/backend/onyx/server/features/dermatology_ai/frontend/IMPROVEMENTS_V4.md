# Mejoras del Frontend - Versión 4

## 📋 Resumen

Esta versión incluye hooks adicionales, componentes avanzados de UI, y mejoras de integración en el layout principal.

## ✨ Nuevas Funcionalidades

### 1. Hooks Adicionales

#### `useClickOutside`
Detecta clics fuera de un elemento (útil para modales, dropdowns).

```typescript
const ref = useRef<HTMLDivElement>(null);
useClickOutside(ref, () => {
  setIsOpen(false);
});
```

#### `useMediaQuery`
Hook para detectar media queries y breakpoints.

```typescript
const isMobile = useMediaQuery('(max-width: 768px)');
const prefersDark = useMediaQuery('(prefers-color-scheme: dark)');
```

**Hooks predefinidos:**
- `useIsMobile()` - Detecta dispositivos móviles
- `useIsTablet()` - Detecta tablets
- `useIsDesktop()` - Detecta escritorio
- `useIsLargeScreen()` - Detecta pantallas grandes
- `usePrefersReducedMotion()` - Detecta preferencia de movimiento reducido
- `usePrefersDarkMode()` - Detecta preferencia de modo oscuro

#### `useWindowSize`
Hook para obtener el tamaño de la ventana.

```typescript
const { width, height } = useWindowSize();
```

#### `useCopyToClipboard`
Hook para copiar texto al portapapeles.

```typescript
const { copy, isCopied, error } = useCopyToClipboard();

await copy('Text to copy');
```

#### `usePrevious`
Hook para obtener el valor anterior de una variable.

```typescript
const previousValue = usePrevious(currentValue);
```

### 2. Componentes Avanzados de UI

#### `Modal`
Componente de modal completo con accesibilidad.

```tsx
<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Modal Title"
  size="lg"
  closeOnClickOutside
  closeOnEscape
>
  <p>Modal content</p>
</Modal>
```

**Características:**
- Focus trap automático
- Cierre con Escape
- Cierre al hacer clic fuera
- Animaciones suaves
- Responsive
- Accesible (ARIA)

#### `Tooltip`
Componente de tooltip con posicionamiento inteligente.

```tsx
<Tooltip content="Tooltip text" position="top">
  <button>Hover me</button>
</Tooltip>
```

**Características:**
- Posicionamiento automático (top, bottom, left, right)
- Mantiene tooltip dentro del viewport
- Delay configurable
- Soporte para hover y focus
- Animaciones suaves

#### `Confetti`
Componente de confetti para celebraciones.

```tsx
<Confetti
  active={showConfetti}
  duration={3000}
  colors={['#3b82f6', '#8b5cf6']}
  particleCount={50}
/>
```

**Características:**
- Animación de caída
- Colores personalizables
- Cantidad de partículas configurable
- Duración configurable

### 3. Utilidades Adicionales

#### `clipboard.ts`
Utilidades para el portapapeles.

```typescript
import { copyToClipboard, readFromClipboard, canUseClipboard } from '@/lib/utils/clipboard';

// Copiar
await copyToClipboard('Text');

// Leer
const text = await readFromClipboard();

// Verificar soporte
if (canUseClipboard()) {
  // Usar clipboard API
}
```

### 4. Integración en Layout

Los siguientes componentes se han integrado automáticamente en el layout principal:

- `NetworkStatus` - Muestra el estado de conexión
- `ToastNotifications` - Sistema de notificaciones mejorado
- `ErrorBoundary` - Captura de errores global

## 🎯 Ejemplos de Uso

### Modal con Formulario

```tsx
import { Modal } from '@/lib/components';
import { useState } from 'react';

function MyComponent() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button onClick={() => setIsOpen(true)}>Open Modal</button>
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Edit Profile"
        size="md"
      >
        <form>
          <input type="text" placeholder="Name" />
          <button type="submit">Save</button>
        </form>
      </Modal>
    </>
  );
}
```

### Tooltip en Botones

```tsx
import { Tooltip } from '@/lib/components';

function ActionButtons() {
  return (
    <div className="flex gap-2">
      <Tooltip content="Save changes" position="top">
        <button>💾</button>
      </Tooltip>
      <Tooltip content="Delete item" position="top">
        <button>🗑️</button>
      </Tooltip>
    </div>
  );
}
```

### Responsive con Media Queries

```tsx
import { useIsMobile, useIsDesktop } from '@/lib/hooks';

function ResponsiveComponent() {
  const isMobile = useIsMobile();
  const isDesktop = useIsDesktop();

  return (
    <div>
      {isMobile && <MobileView />}
      {isDesktop && <DesktopView />}
    </div>
  );
}
```

### Copiar al Portapapeles

```tsx
import { useCopyToClipboard } from '@/lib/hooks';
import { Check, Copy } from 'lucide-react';

function CopyButton({ text }: { text: string }) {
  const { copy, isCopied } = useCopyToClipboard();

  return (
    <button
      onClick={() => copy(text)}
      className="flex items-center gap-2"
    >
      {isCopied ? (
        <>
          <Check className="w-4 h-4" />
          Copied!
        </>
      ) : (
        <>
          <Copy className="w-4 h-4" />
          Copy
        </>
      )}
    </button>
  );
}
```

### Confetti en Éxito

```tsx
import { Confetti } from '@/lib/components';
import { useState } from 'react';

function SuccessPage() {
  const [showConfetti, setShowConfetti] = useState(true);

  return (
    <>
      <Confetti active={showConfetti} duration={3000} />
      <div>
        <h1>Success!</h1>
        <p>Your operation completed successfully</p>
      </div>
    </>
  );
}
```

### Click Outside para Dropdown

```tsx
import { useClickOutside } from '@/lib/hooks';
import { useRef, useState } from 'react';

function Dropdown() {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useClickOutside(dropdownRef, () => {
    setIsOpen(false);
  });

  return (
    <div ref={dropdownRef}>
      <button onClick={() => setIsOpen(!isOpen)}>Toggle</button>
      {isOpen && (
        <div className="dropdown-menu">
          <a href="#">Item 1</a>
          <a href="#">Item 2</a>
        </div>
      )}
    </div>
  );
}
```

## 📦 Archivos Creados

**Hooks:**
- `lib/hooks/useClickOutside.ts`
- `lib/hooks/useMediaQuery.ts`
- `lib/hooks/useWindowSize.ts`
- `lib/hooks/useCopyToClipboard.ts`
- `lib/hooks/usePrevious.ts`

**Componentes:**
- `lib/components/Modal.tsx`
- `lib/components/Tooltip.tsx`
- `lib/components/Confetti.tsx`
- `lib/components/index.ts` (exportaciones)

**Utilidades:**
- `lib/utils/clipboard.ts`

**Integración:**
- `app/layout.tsx` (actualizado con nuevos componentes)

## 🎨 Características Destacadas

### Modal
- ✅ Focus trap automático
- ✅ Cierre con Escape
- ✅ Cierre al hacer clic fuera
- ✅ Animaciones suaves
- ✅ Tamaños configurables
- ✅ Accesible (ARIA)

### Tooltip
- ✅ Posicionamiento inteligente
- ✅ Mantiene dentro del viewport
- ✅ Delay configurable
- ✅ Soporte hover y focus

### Confetti
- ✅ Animación fluida
- ✅ Colores personalizables
- ✅ Configuración flexible

## 🚀 Beneficios

1. **Mejor UX:**
   - Modales accesibles y fáciles de usar
   - Tooltips informativos
   - Feedback visual con confetti

2. **Responsive:**
   - Hooks para detectar breakpoints
   - Componentes adaptativos
   - Mejor experiencia móvil

3. **Accesibilidad:**
   - Focus management
   - ARIA labels
   - Navegación por teclado

4. **Productividad:**
   - Hooks reutilizables
   - Componentes listos para usar
   - Menos código boilerplate

## 📚 Documentación

- Ver `lib/hooks/index.ts` para todos los hooks
- Ver `lib/components/index.ts` para todos los componentes
- Ver `lib/utils/index.ts` para todas las utilidades

## 🔄 Integración Automática

Los siguientes componentes están integrados automáticamente en el layout:

- `NetworkStatus` - Aparece cuando se pierde la conexión
- `ToastNotifications` - Sistema de notificaciones global
- `ErrorBoundary` - Captura errores de React

No necesitas importarlos manualmente, ya están disponibles en toda la aplicación.



