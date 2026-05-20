# Arquitectura de la Aplicación Móvil

## 📐 Visión General

La aplicación móvil está construida con **Expo** usando la **nueva arquitectura de React Native** y **TypeScript estricto**. Sigue principios de diseño moderno, escalabilidad y mantenibilidad.

## 🏗️ Estructura de Carpetas

```
mobile/
├── app/                      # Expo Router (file-based routing)
│   ├── (tabs)/              # Navegación por tabs
│   │   ├── index.tsx        # Pantalla principal
│   │   ├── generate.tsx     # Generar manual
│   │   ├── history.tsx      # Historial
│   │   └── profile.tsx      # Perfil
│   ├── manual/[id].tsx      # Detalle de manual (dynamic route)
│   └── _layout.tsx          # Layout raíz
├── src/
│   ├── components/          # Componentes reutilizables
│   │   ├── home/           # Componentes de inicio
│   │   ├── generate/       # Componentes de generación
│   │   ├── history/        # Componentes de historial
│   │   ├── manual/        # Componentes de manual
│   │   ├── navigation/    # Componentes de navegación
│   │   └── ui/            # Componentes UI base
│   ├── constants/          # Constantes (colores, categorías)
│   ├── lib/               # Utilidades y contexto
│   │   ├── context/       # Context providers
│   │   ├── hooks/         # Custom hooks
│   │   └── utils/         # Utilidades
│   ├── services/          # Servicios API
│   │   └── api/           # Cliente API y servicios
│   ├── types/             # Tipos TypeScript
│   └── utils/             # Utilidades adicionales
├── assets/                # Imágenes y recursos
└── config files          # Configuración (tsconfig, babel, etc.)
```

## 🔄 Flujo de Datos

### 1. Data Fetching

```
Component → React Query Hook → API Service → API Client → Backend
                ↓
         Cache & State Management
```

### 2. Estado Global

- **React Context** para tema y configuración global
- **React Query** para estado del servidor (caching, sync)
- **Zustand** (opcional) para estado local complejo

### 3. Navegación

```
Expo Router (file-based)
    ↓
Stack Navigation (tabs)
    ↓
Screen Components
```

## 🎨 Sistema de Diseño

### Colores

- Centralizado en `src/constants/colors.ts`
- Soporte automático para light/dark mode
- Basado en la configuración del sistema

### Componentes

- **Atomic Design**: Componentes base → Componentes compuestos → Pantallas
- **Reutilizables**: Componentes UI en `src/components/ui/`
- **Específicos**: Componentes de dominio en carpetas específicas

## 🔌 Integración con Backend

### API Client

- Cliente centralizado con Axios
- Interceptores para auth y errores
- Manejo de FormData para imágenes
- Timeout y retry logic

### Servicios

- `manualService`: Todas las operaciones relacionadas con manuales
- Tipos TypeScript que coinciden con los modelos del backend
- Validación con Zod

## 📱 Características Principales

### 1. Generación de Manuales

- **Modo Texto**: Descripción escrita
- **Modo Imagen**: Una foto del problema
- **Modo Galería**: Múltiples imágenes (hasta 5)

### 2. Historial

- Lista de manuales generados
- Búsqueda y filtrado
- Pull-to-refresh

### 3. Perfil

- Estadísticas de uso
- Configuración de tema
- Preferencias

## 🛡️ Seguridad

- **Secure Store**: Almacenamiento seguro de tokens
- **Validación**: Zod para validación de entrada
- **HTTPS**: Todas las comunicaciones API
- **Error Boundaries**: Manejo global de errores

## ⚡ Optimizaciones

### Performance

- **Lazy Loading**: Componentes cargados bajo demanda
- **Image Optimization**: expo-image para mejor rendimiento
- **Memoization**: useMemo y useCallback donde sea necesario
- **Code Splitting**: Separación de código por rutas

### Caching

- **React Query**: Cache automático de peticiones
- **Stale Time**: 5 minutos por defecto
- **Garbage Collection**: 10 minutos

## 🧪 Testing

- **Jest**: Framework de testing
- **React Native Testing Library**: Testing de componentes
- **Coverage**: Configurado para reportes de cobertura

## 📦 Build y Deploy

### Desarrollo

```bash
npm start
```

### Producción

```bash
# Build nativo
eas build --platform ios
eas build --platform android

# OTA Updates
eas update --branch production
```

## 🔄 Sincronización con Backend

Los tipos TypeScript en `src/types/api.ts` están diseñados para coincidir exactamente con los modelos Pydantic del backend:

- `ManualTextRequest` ↔ `ManualTextRequest` (backend)
- `ManualResponse` ↔ `ManualResponse` (backend)
- `Manual` ↔ `Manual` (database model)

Esto asegura type safety end-to-end.

## 🎯 Mejores Prácticas

1. **TypeScript Estricto**: Siempre tipar todo
2. **Componentes Funcionales**: No usar clases
3. **Hooks Personalizados**: Reutilizar lógica
4. **Error Handling**: Manejar errores en todos los niveles
5. **Accessibility**: Soporte completo de a11y
6. **Performance**: Optimizar renders y re-renders

## 📚 Referencias

- [Expo Documentation](https://docs.expo.dev/)
- [Expo Router](https://docs.expo.dev/router/introduction/)
- [React Query](https://tanstack.com/query/latest)
- [React Native Best Practices](https://reactnative.dev/docs/performance)




