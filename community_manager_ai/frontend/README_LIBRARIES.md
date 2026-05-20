# Guía de Instalación de Librerías

## 🚀 Instalación

Después de actualizar el `package.json`, ejecuta:

```bash
npm install
```

## 📚 Librerías Agregadas

### Validación
- **zod** + **@hookform/resolvers**: Validación TypeScript-first para formularios

### Estado y Datos
- **@tanstack/react-query**: Manejo de estado del servidor con caché inteligente
- **zustand**: Estado global ligero con persistencia

### UI y Animaciones
- **framer-motion**: Animaciones fluidas y gestos
- **sonner**: Sistema de toasts moderno
- **@radix-ui/react-***: Componentes accesibles headless

### Utilidades
- **react-dropzone**: Upload con drag & drop
- **react-datepicker**: Selector de fechas
- **react-select**: Selects avanzados
- **use-debounce**: Debounce para búsquedas
- **nanoid**: IDs únicos
- **react-number-format**: Formateo de números

### Testing
- **@testing-library/react**: Testing de componentes
- **@testing-library/jest-dom**: Matchers de Jest
- **@testing-library/user-event**: Simulación de eventos

## 🔧 Configuración Requerida

### 1. Providers (ya configurado)
El archivo `app/providers.tsx` ya está configurado con:
- QueryClientProvider
- Toaster de Sonner
- ReactQueryDevtools

### 2. Esquemas de Validación
Los esquemas Zod están en `lib/zod-schemas.ts`:
- `postSchema`
- `memeSchema`
- `templateSchema`
- `platformConnectSchema`

### 3. Store Global
El store Zustand está en `lib/store.ts`:
- `useAppStore` - Estado global de la app
- Persistencia automática
- DevTools integrado

### 4. Query Client
Configurado en `lib/query-client.ts` con:
- Caché de 5 minutos
- Retry configurado
- Refetch optimizado

## 💡 Ejemplos de Uso

### Validación con Zod
```typescript
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import { postSchema } from '@/lib/zod-schemas';

const form = useForm({
  resolver: zodResolver(postSchema),
  defaultValues: {
    content: '',
    platforms: [],
  },
});
```

### React Query
```typescript
import { useQuery, useMutation } from '@tanstack/react-query';
import { postsApi } from '@/lib/api';

// Query
const { data, isLoading, error } = useQuery({
  queryKey: ['posts'],
  queryFn: () => postsApi.getAll(),
});

// Mutation
const mutation = useMutation({
  mutationFn: postsApi.create,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['posts'] });
  },
});
```

### Zustand Store
```typescript
import { useAppStore } from '@/lib/store';

const theme = useAppStore((state) => state.theme);
const setTheme = useAppStore((state) => state.setTheme);
const toggleSidebar = useAppStore((state) => state.toggleSidebar);
```

### Sonner Toasts
```typescript
import { toast } from 'sonner';

toast.success('Operación exitosa');
toast.error('Error en la operación');
toast.info('Información importante');
toast.warning('Advertencia');
```

### Framer Motion
```typescript
import { motion } from 'framer-motion';

<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.3 }}
>
  Contenido animado
</motion.div>
```

### React Dropzone
```typescript
import { FileUpload } from '@/components/ui/FileUpload';

<FileUpload
  onFileSelect={(file) => console.log(file)}
  accept={{ 'image/*': ['.png', '.jpg'] }}
  maxSize={10 * 1024 * 1024}
/>
```

### Debounce Hook
```typescript
import { useDebounce } from '@/hooks/useDebounce';

const [search, setSearch] = useState('');
const debouncedSearch = useDebounce(search, 300);

useEffect(() => {
  // Búsqueda con debounce
  if (debouncedSearch) {
    // hacer búsqueda
  }
}, [debouncedSearch]);
```

## 🎯 Beneficios

1. **Mejor DX**: Desarrollo más rápido y eficiente
2. **Type Safety**: Validación TypeScript-first
3. **Performance**: Caché inteligente y optimizaciones
4. **UX**: Animaciones fluidas y feedback visual
5. **Accesibilidad**: Componentes Radix UI con ARIA completo
6. **Testing**: Herramientas modernas de testing

## 📝 Notas

- Todas las librerías están actualizadas a las últimas versiones estables
- Compatibles con Next.js 14 y React 18
- TypeScript support completo
- Tree-shaking habilitado para optimización de bundle



