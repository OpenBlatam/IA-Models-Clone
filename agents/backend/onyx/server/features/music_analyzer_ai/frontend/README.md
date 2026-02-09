# Blatam Academy Frontend

Frontend Next.js 14 con TypeScript para las plataformas Music Analyzer AI y Robot Movement AI.

## 🚀 Características

- **Next.js 14** con App Router y Server Components
- **TypeScript** para type safety completo
- **Tailwind CSS** para estilos modernos y responsive
- **React Query** para manejo de estado y caché optimizado
- **Zustand** para estado global ligero
- **Zod** para validación type-safe
- **Integración completa** con ambos backends
- **UI moderna y responsive** con mobile-first approach
- **Accesibilidad completa** (WCAG 2.1)
- **Performance optimizada** con code splitting y lazy loading
- **Seguridad robusta** con sanitización y validación
- **Utilidades de optimización** para componentes React (memoización, lazy loading)
- **Hooks de rendimiento** (debounce, throttle, memoización avanzada)
- **Configuración unificada** con tipos TypeScript completos
- **Manejo avanzado de errores** con recuperación y retry
- **Utilidades de accesibilidad** (WCAG 2.1 compliant)
- **Seguridad avanzada** (XSS protection, file validation, rate limiting)
- **Tipos TypeScript utilitarios** para mejor type safety

## 📦 Instalación

```bash
# Instalar dependencias
npm install

# O con yarn
yarn install

# O con pnpm
pnpm install
```

## ⚙️ Configuración

Crea un archivo `.env.local` basado en `.env.example`:

```env
# API URLs
NEXT_PUBLIC_MUSIC_API_URL=http://localhost:8010
NEXT_PUBLIC_ROBOT_API_URL=http://localhost:8010

# Feature Flags
NEXT_PUBLIC_ENABLE_VOICE_COMMANDS=false
NEXT_PUBLIC_ENABLE_OFFLINE_MODE=false
NEXT_PUBLIC_ENABLE_ANALYTICS=true
```

## 🏃 Desarrollo

```bash
# Iniciar servidor de desarrollo
npm run dev

# Build para producción
npm run build

# Iniciar servidor de producción
npm start

# Type checking
npm run type-check

# Linting
npm run lint

# Tests
npm test

# Tests con coverage
npm run test:coverage
```

## 📁 Estructura del Proyecto

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Layout principal
│   ├── providers.tsx      # Providers (React Query, etc.)
│   ├── globals.css        # Estilos globales
│   ├── music/             # Página de Music Analyzer
│   └── robot/             # Página de Robot Movement
├── components/             # Componentes React
│   ├── ui/                # Componentes UI reutilizables
│   └── music/             # Componentes específicos de música
├── lib/                    # Utilidades y lógica
│   ├── api/               # Cliente API y servicios
│   ├── config/            # Configuración
│   ├── constants/         # Constantes
│   ├── errors/            # Tipos de error
│   ├── hooks/             # Hooks personalizados
│   ├── store/             # Zustand stores
│   ├── utils/             # Utilidades
│   ├── validations/       # Zod schemas
│   └── types/             # Tipos TypeScript
└── public/                # Archivos estáticos
```

## 🎯 Características Principales

### Music Analyzer AI
- ✅ Búsqueda de canciones
- ✅ Análisis musical con IA
- ✅ Comparación de tracks
- ✅ Recomendaciones personalizadas
- ✅ Gestión de favoritos
- ✅ Historial de búsquedas
- ✅ Análisis avanzado con ML

### Robot Movement AI
- ✅ Control de robot
- ✅ Comandos de movimiento
- ✅ Visualización en tiempo real

## 🛠️ Tecnologías

- **Next.js 14** - Framework React
- **TypeScript** - Type safety
- **Tailwind CSS** - Estilos
- **React Query** - Data fetching y caché
- **Zustand** - Estado global
- **Zod** - Validación
- **Axios** - HTTP client
- **Lucide React** - Iconos
- **React Hot Toast** - Notificaciones

## ⚡ Optimizaciones Incluidas

### Utilidades de Componentes
- `memoizeComponent` - Memoización con comparación personalizada
- `lazyLoadComponent` - Lazy loading con Suspense
- `conditionalRender` - Renderizado condicional
- `clientOnly` - Componentes solo cliente

### Hooks de Rendimiento
- `useMemoizedValue` - Memoización avanzada
- `useStableCallback` - Callbacks estables
- `useDebouncedCallback` - Debounce de callbacks
- `useThrottledCallback` - Throttle de callbacks
- `useRenderCount` - Debugging de renders

### Hooks de Accesibilidad
- `useAnnounce` - Anuncios a lectores de pantalla
- `useFocusTrap` - Atrapar foco en modales
- `usePrefersReducedMotion` - Detectar preferencia de movimiento reducido
- `usePrefersHighContrast` - Detectar preferencia de alto contraste
- `useKeyboardNavigation` - Navegación por teclado

### Manejo de Errores
- `handleError` - Manejo completo de errores
- `retryWithBackoff` - Retry con backoff exponencial
- `recoverFromError` - Estrategias de recuperación
- `withErrorHandling` - Wrapper para funciones async

### Seguridad
- `sanitizeXss` - Sanitización XSS
- `validateFileType` - Validación de tipos de archivo
- `validateFileSize` - Validación de tamaño de archivo
- `ClientRateLimiter` - Rate limiting en cliente
- `hashString` - Hash seguro de strings

### Configuración Optimizada
- React Query con `structuralSharing` habilitado
- Cache configurado con `performanceConfig`
- Retry logic inteligente
- Error handling mejorado

## 📚 Documentación

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Arquitectura del proyecto
- [PERFORMANCE_GUIDE.md](./PERFORMANCE_GUIDE.md) - Guía de optimización de rendimiento
- [STATE_MANAGEMENT_GUIDE.md](./STATE_MANAGEMENT_GUIDE.md) - Guía de estado
- [VALIDATION_GUIDE.md](./VALIDATION_GUIDE.md) - Guía de validación
- [API_CONNECTION_GUIDE.md](./API_CONNECTION_GUIDE.md) - Guía de conexión API
- [COMPLETE_FINAL_SUMMARY.md](./COMPLETE_FINAL_SUMMARY.md) - Resumen completo

## 🧪 Testing

```bash
# Ejecutar tests
npm test

# Tests en modo watch
npm run test:watch

# Coverage
npm run test:coverage
```

## 🚀 Deployment

El proyecto está optimizado para deployment en:
- Vercel (recomendado)
- Netlify
- AWS Amplify
- Cualquier plataforma que soporte Next.js

## 📝 Notas

- El proyecto usa Server Components donde es posible
- Client Components solo cuando se necesita interactividad
- Todas las imágenes están optimizadas
- El código sigue las mejores prácticas de Next.js 14

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto es parte de Blatam Academy.
