# Arquitectura Modular

## 📁 Estructura del Proyecto

El proyecto está organizado siguiendo principios de arquitectura modular, separando responsabilidades y agrupando código relacionado por features.

```
src/
├── features/              # Módulos de features (organizados por dominio)
│   ├── home/
│   │   ├── screens/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── index.ts       # Exportaciones públicas del módulo
│   ├── camera/
│   ├── analysis/
│   ├── recommendations/
│   ├── history/
│   ├── profile/
│   └── real-time-scan/
│
├── components/            # Componentes reutilizables compartidos
│   ├── ui/               # Componentes UI básicos
│   ├── layout/           # Componentes de layout
│   ├── forms/             # Componentes de formularios
│   └── index.ts          # Exportaciones centralizadas
│
├── hooks/                # Hooks reutilizables
│   ├── data/             # Hooks de manejo de datos
│   ├── ui/               # Hooks de UI
│   ├── performance/      # Hooks de optimización
│   └── index.ts          # Exportaciones centralizadas
│
├── utils/                # Utilidades y helpers
│   ├── performance.ts    # Utilidades de performance
│   ├── validation.ts     # Validaciones
│   └── index.ts          # Exportaciones centralizadas
│
├── navigation/           # Configuración de navegación
│   ├── TabNavigator.tsx
│   ├── StackNavigator.tsx
│   └── NavigationContainer.tsx
│
├── providers/            # Context providers
│   └── AppProviders.tsx
│
├── config/               # Configuraciones
│   └── navigation.ts
│
├── types/                # Tipos TypeScript compartidos
│   └── navigation.ts
│
├── context/              # Context API
├── store/                # Redux store
└── services/             # Servicios API
```

## 🎯 Principios de Diseño

### 1. Separación por Features
Cada feature es un módulo independiente que contiene:
- **Screens**: Pantallas específicas del feature
- **Components**: Componentes específicos del feature
- **Hooks**: Hooks específicos del feature
- **Types**: Tipos TypeScript específicos
- **Utils**: Utilidades específicas
- **index.ts**: Punto de entrada público del módulo

### 2. Componentes Compartidos
Los componentes reutilizables se organizan en:
- **UI Components**: Botones, cards, inputs básicos
- **Layout Components**: Headers, containers, grids
- **Form Components**: Inputs, selects, validators
- **Data Display**: Tables, lists, charts

### 3. Hooks Modulares
Los hooks se organizan por categoría:
- **Data Management**: usePagination, useSort, useFilter
- **UI**: useModal, useToast, useConfirm
- **Performance**: useOptimizedFlatList, useMemoizedCallback
- **System**: usePermissions, useNetworkStatus

### 4. Exportaciones Centralizadas
Cada módulo tiene un `index.ts` que exporta solo lo necesario:
```typescript
// features/home/index.ts
export { default as HomeScreen } from './screens/HomeScreen';
export { HomeFeatureComponent } from './components/HomeFeatureComponent';
```

## 📦 Estructura de un Feature Module

```
features/home/
├── screens/
│   └── HomeScreen.tsx
├── components/
│   ├── QuickActions.tsx
│   └── RecentAnalyses.tsx
├── hooks/
│   └── useHomeData.ts
├── types/
│   └── home.types.ts
├── utils/
│   └── home.utils.ts
└── index.ts              # Exportaciones públicas
```

### Reglas de un Feature Module:
1. **Encapsulación**: El código dentro de un feature solo se exporta a través de `index.ts`
2. **Independencia**: Los features no deben depender directamente de otros features
3. **Compartir a través de Common**: Si algo es compartido, va en `components/`, `hooks/`, o `utils/`
4. **Tipos Compartidos**: Los tipos compartidos van en `types/`

## 🔄 Flujo de Importaciones

### ✅ Correcto:
```typescript
// Desde un feature
import { HomeScreen } from '../features/home';

// Desde componentes compartidos
import { Button, Card } from '../components';

// Desde hooks compartidos
import { usePagination, useModal } from '../hooks';

// Desde utils
import { debounce, throttle } from '../utils';
```

### ❌ Incorrecto:
```typescript
// No importar directamente desde dentro de un feature
import HomeScreen from '../features/home/screens/HomeScreen';

// No importar desde otro feature directamente
import { CameraComponent } from '../features/camera/components';
```

## 🎨 Beneficios de la Arquitectura Modular

### 1. Mantenibilidad
- Código organizado por dominio de negocio
- Fácil localizar y modificar código relacionado
- Cambios aislados por feature

### 2. Escalabilidad
- Fácil agregar nuevos features
- Estructura consistente en todo el proyecto
- Reutilización de componentes compartidos

### 3. Testabilidad
- Features pueden ser testeados independientemente
- Mocks más simples y específicos
- Tests más enfocados

### 4. Colaboración
- Equipos pueden trabajar en features diferentes sin conflictos
- Código más fácil de revisar
- Onboarding más rápido para nuevos desarrolladores

## 📝 Convenciones de Nomenclatura

### Directorios
- **Features**: `kebab-case` (ej: `real-time-scan`)
- **Componentes**: `PascalCase` (ej: `HomeScreen.tsx`)
- **Hooks**: `camelCase` con prefijo `use` (ej: `useHomeData.ts`)
- **Utils**: `camelCase` (ej: `home.utils.ts`)
- **Types**: `camelCase` con sufijo `.types` (ej: `home.types.ts`)

### Archivos
- **Componentes**: `PascalCase.tsx`
- **Hooks**: `use*.ts` o `use*.tsx`
- **Utils**: `*.utils.ts`
- **Types**: `*.types.ts` o `*.ts` en carpeta `types/`
- **Config**: `*.config.ts` o `*.ts` en carpeta `config/`

## 🚀 Migración a Estructura Modular

Para migrar código existente:

1. **Identificar Features**: Agrupar código relacionado
2. **Crear Estructura**: Crear carpetas de feature con subcarpetas
3. **Mover Código**: Mover archivos a sus nuevas ubicaciones
4. **Crear index.ts**: Exportar solo lo necesario
5. **Actualizar Imports**: Actualizar todas las referencias
6. **Verificar**: Ejecutar tests y linting

## 🔍 Ejemplo Completo

### Feature: Analysis
```typescript
// features/analysis/index.ts
export { default as AnalysisScreen } from './screens/AnalysisScreen';
export { default as ReportScreen } from './screens/ReportScreen';
export { AnalysisCard } from './components/AnalysisCard';
export { useAnalysis } from './hooks/useAnalysis';

// features/analysis/types/analysis.types.ts
export interface AnalysisResult {
  id: string;
  score: number;
  recommendations: string[];
}

// features/analysis/hooks/useAnalysis.ts
export const useAnalysis = () => {
  // Hook específico del feature
};

// Uso en otro lugar:
import { AnalysisScreen, useAnalysis } from '../features/analysis';
```

## 📚 Referencias

- [React Native Best Practices](https://reactnative.dev/docs/performance)
- [TypeScript Module Resolution](https://www.typescriptlang.org/docs/handbook/module-resolution.html)
- [Feature-Sliced Design](https://feature-sliced.design/)

