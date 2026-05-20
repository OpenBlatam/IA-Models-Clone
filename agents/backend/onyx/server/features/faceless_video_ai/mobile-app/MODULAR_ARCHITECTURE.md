# Arquitectura Modular

La aplicación sigue una arquitectura modular basada en features, donde cada módulo encapsula toda la funcionalidad relacionada con un dominio específico.

## 📐 Estructura Modular

```
modules/
├── video/              # Módulo de videos
│   ├── components/     # Componentes específicos de video
│   ├── hooks/          # Hooks de video
│   ├── services/       # Servicios de video
│   ├── types/          # Tipos específicos de video
│   ├── utils/          # Utilidades de video
│   └── index.ts        # Exportaciones centralizadas
├── auth/               # Módulo de autenticación
│   ├── components/
│   ├── hooks/
│   ├── services/
│   ├── types/
│   ├── utils/
│   └── index.ts
├── template/           # Módulo de templates
│   └── ...
├── analytics/          # Módulo de analytics
│   └── ...
└── shared/             # Módulo compartido
    └── ...
```

## 🎯 Principios de Modularidad

### 1. Feature-Based Organization
Cada feature tiene su propio módulo con toda su funcionalidad:
- Componentes UI
- Lógica de negocio (hooks)
- Servicios API
- Tipos TypeScript
- Utilidades específicas

### 2. Encapsulación
Cada módulo es autocontenido:
- No depende de implementaciones internas de otros módulos
- Expone solo lo necesario a través de `index.ts`
- Mantiene su propia estructura interna

### 3. Reutilización
El módulo `shared` contiene:
- Componentes reutilizables
- Hooks comunes
- Utilidades compartidas

### 4. Separación de Concerns
- **Components**: Solo presentación
- **Hooks**: Lógica de negocio
- **Services**: Comunicación con API
- **Types**: Definiciones de tipos
- **Utils**: Funciones helper

## 📦 Módulos Disponibles

### Video Module (`modules/video/`)

**Componentes:**
- `VideoCard` - Tarjeta de video
- `VideoList` - Lista de videos
- `VideoProgress` - Barra de progreso
- `VideoStatusBadge` - Badge de estado

**Hooks:**
- `useVideoGeneration` - Generar video
- `useVideoStatus` - Estado de video
- `useVideoActions` - Acciones (share, delete, etc.)
- `useVideoVersions` - Versiones de video

**Utils:**
- `getVideoStatusColor()` - Color por estado
- `isVideoProcessing()` - Verificar si está procesando
- `canDownloadVideo()` - Verificar si se puede descargar

**Uso:**
```tsx
import { VideoCard, VideoList, useVideoGeneration } from '@/modules/video';
```

### Auth Module (`modules/auth/`)

**Componentes:**
- `LoginForm` - Formulario de login
- `RegisterForm` - Formulario de registro
- `AuthGuard` - Guard de autenticación

**Hooks:**
- `useAuth` - Hook de autenticación

**Utils:**
- `validateLoginForm()` - Validar formulario de login
- `validateRegisterForm()` - Validar formulario de registro

**Uso:**
```tsx
import { AuthGuard, validateLoginForm } from '@/modules/auth';
```

### Template Module (`modules/template/`)

**Componentes:**
- `TemplateCard` - Tarjeta de template
- `TemplateList` - Lista de templates
- `TemplateSelector` - Selector de template

**Hooks:**
- `useTemplates` - Obtener templates
- `useTemplateGeneration` - Generar desde template

**Utils:**
- `isCustomTemplate()` - Verificar si es custom
- `getTemplateConfig()` - Obtener configuración

**Uso:**
```tsx
import { TemplateCard, useTemplates } from '@/modules/template';
```

### Analytics Module (`modules/analytics/`)

**Componentes:**
- `AnalyticsCard` - Tarjeta de métrica
- `MetricsChart` - Gráfico de métricas
- `QuotaIndicator` - Indicador de cuota

**Hooks:**
- `useAnalytics` - Obtener analytics
- `useRecommendations` - Obtener recomendaciones
- `useQuota` - Obtener cuota

**Utils:**
- `getQuotaUsagePercentage()` - Porcentaje de uso
- `isQuotaExceeded()` - Verificar si excedió cuota

**Uso:**
```tsx
import { QuotaIndicator, useAnalytics } from '@/modules/analytics';
```

### Shared Module (`modules/shared/`)

**Componentes:**
- `EmptyState` - Estado vacío
- `ErrorState` - Estado de error
- `LoadingState` - Estado de carga
- `SectionHeader` - Header de sección

**Hooks:**
- `useDebounce` - Debounce hook
- `useThrottle` - Throttle hook

**Utils:**
- `createId()` - Crear ID único
- `delay()` - Delay async
- `isDefined()` - Verificar si está definido

**Uso:**
```tsx
import { EmptyState, ErrorState, SectionHeader } from '@/modules/shared';
```

## 🔄 Flujo de Datos en Módulos

```
Component (UI)
    ↓
Hook (Logic)
    ↓
Service (API)
    ↓
API Client
```

## 📝 Convenciones

### Naming
- Módulos: `kebab-case` (ej: `video-generation`)
- Componentes: `PascalCase` (ej: `VideoCard`)
- Hooks: `camelCase` con prefijo `use` (ej: `useVideoGeneration`)
- Services: `camelCase` con sufijo `Service` (ej: `videoService`)
- Types: `PascalCase` con sufijo `Props` o `Type` (ej: `VideoCardProps`)

### Estructura de Archivos
```
module-name/
├── components/
│   ├── component-name.tsx
│   └── index.ts
├── hooks/
│   ├── use-hook-name.ts
│   └── index.ts
├── services/
│   ├── service-name.ts
│   └── index.ts
├── types/
│   ├── module-types.ts
│   └── index.ts
├── utils/
│   ├── module-helpers.ts
│   └── index.ts
└── index.ts
```

### Barrel Exports
Cada módulo tiene un `index.ts` que exporta todo:

```tsx
// modules/video/index.ts
export * from './components/video-card';
export * from './hooks/use-video-generation';
export * from './services/video-service';
```

## 🎨 Uso en Componentes

### Importar desde módulo específico
```tsx
import { VideoCard, useVideoGeneration } from '@/modules/video';
```

### Importar desde índice central
```tsx
import { VideoModule, AuthModule } from '@/modules';
```

### Importar componentes compartidos
```tsx
import { EmptyState, ErrorState } from '@/modules/shared';
```

## 🔒 Reglas de Módulos

1. **No circular dependencies**: Los módulos no deben importarse entre sí
2. **Shared es común**: Todos los módulos pueden usar `shared`
3. **Exports controlados**: Solo exportar lo necesario
4. **Types compartidos**: Tipos comunes en `@/types`
5. **Services centralizados**: Servicios API en `@/services`

## 🚀 Beneficios

1. **Mantenibilidad**: Fácil encontrar y modificar código
2. **Escalabilidad**: Agregar nuevos módulos es simple
3. **Testabilidad**: Cada módulo se puede testear independientemente
4. **Reutilización**: Componentes y hooks reutilizables
5. **Colaboración**: Múltiples desarrolladores pueden trabajar en paralelo
6. **Performance**: Code splitting por módulo

## 📊 Métricas

- **Módulos**: 5 (video, auth, template, analytics, shared)
- **Componentes**: 15+
- **Hooks**: 20+
- **Services**: 10+
- **Type Safety**: 100%

---

**Versión**: 2.0.0  
**Última actualización**: 2024

