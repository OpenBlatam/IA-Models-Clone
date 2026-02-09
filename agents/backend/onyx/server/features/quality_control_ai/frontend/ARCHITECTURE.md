# Arquitectura del Frontend

## 🏗️ Estructura de Arquitectura

### Capas de la Aplicación

```
frontend/
├── config/              # Configuración centralizada
│   ├── constants.ts    # Constantes de la aplicación
│   ├── api.config.ts   # Configuración de API
│   └── app.config.ts   # Configuración de la app
├── lib/                # Librerías y utilidades
│   ├── api/            # Cliente API
│   │   └── client.ts   # Cliente HTTP configurado
│   ├── services/       # Servicios de dominio
│   │   ├── quality.service.ts
│   │   ├── defect.service.ts
│   │   └── statistics.service.ts
│   ├── validators/     # Validadores Zod
│   │   ├── camera.validator.ts
│   │   └── detection.validator.ts
│   ├── utils/          # Utilidades
│   │   ├── cn.ts       # Clase names utility
│   │   ├── formatting.ts
│   │   ├── styles.ts
│   │   └── dom.ts
│   ├── hooks/          # Hooks compartidos
│   └── store.ts        # Estado global (Zustand)
├── modules/            # Módulos de dominio
│   ├── camera/         # Módulo de cámara
│   ├── inspection/     # Módulo de inspección
│   ├── alerts/         # Módulo de alertas
│   ├── detection/      # Módulo de detección
│   ├── reports/        # Módulo de reportes
│   ├── statistics/     # Módulo de estadísticas
│   └── control/        # Módulo de control
├── components/         # Componentes compartidos
│   ├── ui/             # Componentes UI primitivos
│   └── layout/         # Componentes de layout
└── app/                # Next.js App Router
    ├── layout.tsx
    ├── page.tsx
    └── providers.tsx
```

## 📦 Capas de Arquitectura

### 1. Config Layer (Configuración)
- **constants.ts**: Constantes de la aplicación
- **api.config.ts**: Endpoints y configuración de API
- **app.config.ts**: Configuración de la aplicación

### 2. Infrastructure Layer (Infraestructura)
- **lib/api/client.ts**: Cliente HTTP con interceptores
- **lib/store.ts**: Estado global con Zustand
- **lib/utils/**: Utilidades generales

### 3. Domain Layer (Dominio)
- **lib/services/**: Servicios de lógica de negocio
  - QualityService: Cálculo de calidad
  - DefectService: Análisis de defectos
  - StatisticsService: Estadísticas

### 4. Presentation Layer (Presentación)
- **modules/**: Módulos de dominio con componentes
- **components/ui/**: Componentes UI reutilizables
- **components/layout/**: Componentes de layout

## 🔄 Flujo de Datos

```
User Action
    ↓
Component (Presentation)
    ↓
Hook (Custom Hook)
    ↓
API Service (Infrastructure)
    ↓
Backend API
    ↓
Response
    ↓
Store Update (Zustand)
    ↓
Component Re-render
```

## 🎯 Principios de Arquitectura

### 1. Separación de Responsabilidades
- **Config**: Configuración centralizada
- **Services**: Lógica de negocio
- **API**: Comunicación con backend
- **Components**: Presentación
- **Hooks**: Lógica de componentes

### 2. Modularidad
- Cada módulo es independiente
- Módulos comunican vía APIs
- Tipos compartidos por módulo

### 3. Reutilización
- Servicios reutilizables
- Componentes UI reutilizables
- Hooks compartidos

### 4. Testabilidad
- Servicios puros (fáciles de testear)
- Componentes aislados
- Mocks fáciles de crear

## 📋 Patrones Utilizados

### Repository Pattern
- APIs encapsulan acceso a datos
- Fácil cambiar implementación

### Service Pattern
- Lógica de negocio en servicios
- Servicios puros (sin side effects)

### Factory Pattern
- createApiClient() para crear clientes

### Observer Pattern
- Zustand store para estado global
- React Query para caché

## 🔧 Configuración Centralizada

### Ventajas
- ✅ Fácil cambiar configuración
- ✅ Un solo lugar para constantes
- ✅ Type-safe
- ✅ Fácil de testear

### Ejemplo
```typescript
import { API_CONFIG, QUALITY_THRESHOLDS } from '@/config';

// En lugar de valores hardcodeados
const threshold = QUALITY_THRESHOLDS.EXCELLENT;
const endpoint = API_CONFIG.ENDPOINTS.CAMERA.INFO;
```

## 🚀 Beneficios

1. **Mantenibilidad**: Código organizado y fácil de encontrar
2. **Escalabilidad**: Fácil agregar nuevos módulos
3. **Testabilidad**: Servicios y componentes aislados
4. **Reutilización**: Componentes y servicios compartidos
5. **Type Safety**: TypeScript en toda la aplicación
6. **Performance**: Optimizaciones aplicadas

