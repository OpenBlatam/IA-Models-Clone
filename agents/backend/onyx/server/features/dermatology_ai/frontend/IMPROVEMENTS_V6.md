# Mejoras del Frontend - Versión 6

## 📋 Resumen

Esta versión incluye hooks para funcionalidades del navegador, componentes de UI mejorados, y utilidades para storage y URLs.

## ✨ Nuevas Funcionalidades

### 1. Hooks del Navegador

#### `useGeolocation`
Hook para obtener la ubicación geográfica del usuario.

```typescript
const { latitude, longitude, accuracy, error, loading } = useGeolocation({
  enableHighAccuracy: true,
  timeout: 5000,
});

if (loading) return <div>Getting location...</div>;
if (error) return <div>Error: {error.message}</div>;
return <div>Lat: {latitude}, Lng: {longitude}</div>;
```

#### `useFullscreen`
Hook para manejar modo pantalla completa.

```typescript
const [ref, isFullscreen, enterFullscreen, exitFullscreen, toggleFullscreen] = useFullscreen<HTMLDivElement>();

<div ref={ref}>
  <button onClick={toggleFullscreen}>
    {isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
  </button>
</div>
```

**Características:**
- Soporte cross-browser
- Detección automática de cambios
- Métodos para entrar, salir y alternar

#### `useVisibility`
Hook para detectar cuando la pestaña está visible o oculta.

```typescript
const isVisible = useVisibility();

useEffect(() => {
  if (!isVisible) {
    // Pause video, stop animations, etc.
  }
}, [isVisible]);
```

#### `useLockBodyScroll`
Hook para bloquear el scroll del body (útil para modales).

```typescript
useLockBodyScroll(isModalOpen);
```

#### `useFocusTrap`
Hook para atrapar el foco dentro de un elemento.

```typescript
const modalRef = useRef<HTMLDivElement>(null);
useFocusTrap(modalRef, isOpen);
```

### 2. Componentes de UI Mejorados

#### `Backdrop`
Componente de fondo con blur opcional.

```tsx
<Backdrop isOpen={isOpen} onClose={handleClose} blur>
  <ModalContent />
</Backdrop>
```

**Características:**
- Blur opcional
- Cierre al hacer clic fuera
- Portal automático

#### `Spinner`
Componente de spinner mejorado.

```tsx
<Spinner size="lg" color="primary" />
```

**Opciones:**
- Tamaños: sm, md, lg, xl
- Colores: primary, white, gray
- Accesible

#### `Alert`
Componente de alerta mejorado con variantes.

```tsx
<Alert variant="success" title="Success!" onClose={handleClose}>
  Your changes have been saved.
</Alert>
```

**Variantes:**
- default, success, warning, error, info
- Iconos automáticos
- Botón de cierre opcional

### 3. Utilidades Adicionales

#### `storage.ts`
Utilidades mejoradas para localStorage y sessionStorage.

```typescript
import { 
  getLocalStorage, 
  setLocalStorage, 
  removeLocalStorage,
  localStorageManager 
} from '@/lib/utils/storage';

// Convenience functions
const user = getLocalStorage<User>('user');
setLocalStorage('theme', 'dark');
removeLocalStorage('token');

// Or use the manager directly
localStorageManager?.set('key', value);
const keys = localStorageManager?.keys();
const size = localStorageManager?.size();
```

**Características:**
- Type-safe
- Error handling
- Métodos adicionales (has, keys, size)
- Soporte para localStorage y sessionStorage

#### `url.ts`
Utilidades para manejo de URLs y query parameters.

```typescript
import {
  getQueryParam,
  setQueryParam,
  removeQueryParam,
  getAllQueryParams,
  buildQueryString,
  parseQueryString,
  isValidUrl,
  getDomain,
  getPathname,
} from '@/lib/utils/url';

// Get query param
const id = getQueryParam('id');

// Set query param
setQueryParam('page', '2');

// Remove query param
removeQueryParam('filter');

// Get all params
const params = getAllQueryParams();

// Build query string
const query = buildQueryString({ page: 1, sort: 'name' });

// Parse query string
const parsed = parseQueryString('?page=1&sort=name');

// URL validation
if (isValidUrl(url)) {
  const domain = getDomain(url);
  const path = getPathname(url);
}
```

## 🎯 Ejemplos de Uso

### Geolocation para Análisis

```tsx
import { useGeolocation } from '@/lib/hooks';

function LocationBasedAnalysis() {
  const { latitude, longitude, loading, error } = useGeolocation();

  if (loading) return <Spinner />;
  if (error) return <Alert variant="error">Location access denied</Alert>;

  return (
    <div>
      <p>Your location: {latitude}, {longitude}</p>
      <button onClick={() => analyzeWithLocation(latitude, longitude)}>
        Analyze with location
      </button>
    </div>
  );
}
```

### Fullscreen para Imágenes

```tsx
import { useFullscreen } from '@/lib/hooks';

function ImageViewer({ imageUrl }: { imageUrl: string }) {
  const [ref, isFullscreen, toggleFullscreen] = useFullscreen<HTMLDivElement>();

  return (
    <div ref={ref} className="relative">
      <img src={imageUrl} alt="Skin analysis" />
      <button onClick={toggleFullscreen} className="absolute top-4 right-4">
        {isFullscreen ? 'Exit' : 'Fullscreen'}
      </button>
    </div>
  );
}
```

### Pausar cuando la pestaña está oculta

```tsx
import { useVisibility } from '@/lib/hooks';

function VideoPlayer() {
  const isVisible = useVisibility();
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!isVisible && isPlaying) {
      setIsPlaying(false);
      // Pause video
    }
  }, [isVisible, isPlaying]);

  return <video autoPlay={isPlaying} />;
}
```

### Modal con Backdrop

```tsx
import { Backdrop, Modal } from '@/lib/components';
import { useLockBodyScroll } from '@/lib/hooks';

function MyModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  useLockBodyScroll(isOpen);

  return (
    <Backdrop isOpen={isOpen} onClose={onClose} blur>
      <Modal isOpen={isOpen} onClose={onClose} title="Settings">
        <p>Modal content</p>
      </Modal>
    </Backdrop>
  );
}
```

### Storage con Type Safety

```tsx
import { getLocalStorage, setLocalStorage } from '@/lib/utils/storage';

interface UserPreferences {
  theme: 'light' | 'dark';
  language: string;
}

function Settings() {
  const [prefs, setPrefs] = useState<UserPreferences>(
    getLocalStorage<UserPreferences>('preferences', { theme: 'light', language: 'en' })
  );

  const updatePrefs = (newPrefs: UserPreferences) => {
    setPrefs(newPrefs);
    setLocalStorage('preferences', newPrefs);
  };

  return (
    <div>
      <select
        value={prefs.theme}
        onChange={(e) => updatePrefs({ ...prefs, theme: e.target.value as 'light' | 'dark' })}
      >
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </div>
  );
}
```

### Query Parameters

```tsx
import { getQueryParam, setQueryParam } from '@/lib/utils/url';

function FilterableList() {
  const [filter, setFilter] = useState(getQueryParam('filter') || 'all');

  const handleFilterChange = (newFilter: string) => {
    setFilter(newFilter);
    setQueryParam('filter', newFilter);
  };

  return (
    <div>
      <select value={filter} onChange={(e) => handleFilterChange(e.target.value)}>
        <option value="all">All</option>
        <option value="active">Active</option>
      </select>
    </div>
  );
}
```

## 📦 Archivos Creados

**Hooks:**
- `lib/hooks/useGeolocation.ts`
- `lib/hooks/useFullscreen.ts`
- `lib/hooks/useVisibility.ts`
- `lib/hooks/useLockBodyScroll.ts`
- `lib/hooks/useFocusTrap.ts`

**Componentes:**
- `lib/components/Backdrop.tsx`
- `lib/components/Spinner.tsx`
- `lib/components/Alert.tsx`

**Utilidades:**
- `lib/utils/storage.ts`
- `lib/utils/url.ts`

## 🎨 Características Destacadas

### Storage Manager
- ✅ Type-safe get/set
- ✅ Error handling robusto
- ✅ Métodos adicionales (has, keys, size)
- ✅ Soporte localStorage y sessionStorage

### URL Utilities
- ✅ Manejo de query parameters
- ✅ Validación de URLs
- ✅ Parsing y building
- ✅ Helpers para dominio y pathname

### Browser Hooks
- ✅ Geolocation con error handling
- ✅ Fullscreen cross-browser
- ✅ Visibility detection
- ✅ Body scroll lock
- ✅ Focus trap

## 🚀 Beneficios

1. **Funcionalidades del Navegador:**
   - Geolocation para análisis basado en ubicación
   - Fullscreen para mejor visualización
   - Visibility para optimizar recursos

2. **Mejor UX:**
   - Backdrop con blur
   - Alertas mejoradas
   - Spinner consistente

3. **Utilidades Robustas:**
   - Storage type-safe
   - URL manipulation fácil
   - Error handling completo

4. **Accesibilidad:**
   - Focus trap automático
   - Body scroll lock
   - ARIA labels

## 📚 Documentación

- Ver `lib/hooks/index.ts` para todos los hooks
- Ver `lib/components/index.ts` para todos los componentes
- Ver `lib/utils/index.ts` para todas las utilidades

## 🔄 Resumen de Versiones

### Versión 2-5
- Hooks básicos y avanzados
- Componentes de UI
- Utilidades fundamentales

### Versión 6
- Hooks del navegador (geolocation, fullscreen, visibility)
- Componentes mejorados (Backdrop, Spinner, Alert)
- Utilidades avanzadas (storage, url)

## 📊 Estadísticas Totales

- **Total de hooks:** 28
- **Total de componentes:** 19
- **Total de utilidades:** 9 módulos
- **Archivos creados:** 55+
- **Líneas de código:** 4500+



