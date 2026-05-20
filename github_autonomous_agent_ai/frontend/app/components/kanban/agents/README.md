# Agents Section - Estructura Modular Ultra-Granular

Este módulo contiene la implementación altamente modular del componente `AgentsSection` para gestionar agentes continuos.

## 📁 Estructura de Archivos

```
agents/
├── config/
│   ├── constants.ts          # Constantes y configuraciones
│   └── index.ts
├── services/
│   ├── agentsService.ts      # Servicio API para agentes
│   └── index.ts
├── utils/
│   ├── calculations.ts       # Funciones de cálculo (success rate, stats)
│   ├── filters.ts            # Lógica de filtrado
│   ├── formatters.ts         # Formateo de datos (fechas, números)
│   ├── index.ts
│   └── utils.ts              # (deprecado - usar módulos específicos)
├── hooks/
│   ├── useAgents.ts          # Hook para fetch y estado de agentes
│   ├── useFilters.ts         # Hook para filtrado y búsqueda
│   ├── useAgentStats.ts      # Hook para estadísticas (memoizado)
│   ├── useAgentActions.ts    # Hook para acciones (toggle, etc.)
│   └── index.ts
├── components/
│   ├── ui/                   # Componentes UI reutilizables
│   │   ├── SearchInput.tsx
│   │   ├── StatusBadge.tsx
│   │   ├── StatusIndicator.tsx
│   │   ├── ToggleButton.tsx
│   │   ├── StatsDisplay.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ExpandButton.tsx
│   │   ├── ViewModeToggle.tsx
│   │   ├── Select.tsx
│   │   └── index.ts
│   ├── AgentHeader.tsx       # Encabezado con estadísticas
│   ├── AgentFilters.tsx      # Filtros y búsqueda
│   ├── AgentCard.tsx         # Tarjeta individual (vista cards)
│   ├── AgentTable.tsx        # Tabla de agentes (vista table)
│   └── EmptyState.tsx        # Estado vacío
├── types.ts                  # Tipos e interfaces TypeScript
├── AgentsSection.tsx         # Componente principal orquestador
├── index.ts                  # Exports públicos
└── README.md                 # Esta documentación
```

## 🏗️ Arquitectura por Capas

### 1. **Config Layer** (`config/`)
- **constants.ts**: Todas las constantes centralizadas
  - `REFRESH_INTERVAL_MS`: Intervalo de auto-refresh
  - `API_ENDPOINTS`: URLs de API
  - `FILTER_OPTIONS`: Opciones de filtrado
  - `VIEW_MODES`: Modos de vista
  - `SUCCESS_RATE_THRESHOLDS`: Umbrales de éxito

### 2. **Service Layer** (`services/`)
- **agentsService.ts**: Clase de servicio para comunicación con API
  - `fetchAll()`: Obtener todos los agentes
  - `toggleActive()`: Cambiar estado activo/inactivo
  - Manejo centralizado de errores

### 3. **Utils Layer** (`utils/`)
- **calculations.ts**: Funciones puras de cálculo
  - `calculateSuccessRate()`: Calcular tasa de éxito
  - `calculateAgentStats()`: Calcular estadísticas agregadas

- **filters.ts**: Lógica de filtrado
  - `filterAgents()`: Filtrar agentes por estado y búsqueda

- **formatters.ts**: Formateo de datos
  - `formatSuccessRate()`: Formatear porcentaje
  - `formatCredits()`: Formatear créditos
  - `formatDate()`: Formatear fechas
  - `formatAgentCount()`: Formatear contador

### 4. **Hooks Layer** (`hooks/`)
- **useAgents.ts**: Gestión de estado de agentes
  - Fetch automático
  - Auto-refresh cada 10 segundos
  - Manejo de loading y errores

- **useFilters.ts**: Gestión de filtros
  - Búsqueda por texto
  - Filtrado por estado
  - Memoización de resultados

- **useAgentStats.ts**: Cálculo de estadísticas
  - Memoizado para performance
  - Estadísticas agregadas

- **useAgentActions.ts**: Acciones sobre agentes
  - Toggle activo/inactivo
  - Callbacks de éxito

### 5. **Components Layer** (`components/`)

#### UI Components (`components/ui/`)
Componentes reutilizables y atómicos:

- **SearchInput**: Input de búsqueda con icono
- **StatusBadge**: Badge de estado (Activo/Inactivo)
- **StatusIndicator**: Indicador visual de estado
- **ToggleButton**: Botón para activar/pausar
- **StatsDisplay**: Display de estadísticas
- **LoadingSpinner**: Spinner de carga
- **ExpandButton**: Botón de expandir/colapsar
- **ViewModeToggle**: Toggle de modo de vista
- **Select**: Selector dropdown reutilizable

#### Feature Components
- **AgentHeader**: Encabezado con estadísticas y controles
- **AgentFilters**: Filtros y búsqueda
- **AgentCard**: Tarjeta individual (vista cards)
- **AgentTable**: Tabla completa (vista table)
- **EmptyState**: Estado vacío con mensajes

### 6. **Main Component** (`AgentsSection.tsx`)
Componente orquestador que:
- Combina todos los hooks
- Orquesta los componentes
- Maneja estado de UI (expanded, viewMode)

## 🔄 Flujo de Datos

```
AgentsSection
  ├─> useAgents (fetch desde API)
  ├─> useFilters (filtrado memoizado)
  ├─> useAgentStats (cálculo memoizado)
  └─> useAgentActions (acciones con callbacks)
       │
       ├─> AgentHeader (estadísticas + controles)
       ├─> AgentFilters (búsqueda + filtros)
       └─> AgentCard/AgentTable (vista de datos)
            └─> UI Components (componentes atómicos)
```

## 📦 Principios de Modularidad

### 1. **Separación de Responsabilidades**
- Cada módulo tiene una responsabilidad única
- Servicios solo manejan API
- Utils solo contienen funciones puras
- Hooks solo manejan estado y efectos

### 2. **Reutilización**
- Componentes UI son completamente reutilizables
- Hooks pueden usarse independientemente
- Utils son funciones puras sin dependencias

### 3. **Testabilidad**
- Cada módulo puede testearse aisladamente
- Funciones puras fáciles de testear
- Servicios mockeables
- Hooks testeables con React Testing Library

### 4. **Mantenibilidad**
- Código organizado por función
- Fácil encontrar y modificar código
- Cambios localizados sin efectos secundarios

### 5. **Escalabilidad**
- Fácil agregar nuevas funcionalidades
- Nuevos componentes UI sin afectar existentes
- Nuevos hooks sin modificar componentes

## 🎯 Uso

```tsx
import { AgentsSection } from "./components/kanban/agents";

<AgentsSection className="my-custom-class" />
```

## 🔧 Extensibilidad

### Agregar nuevo filtro:
1. Agregar opción en `config/constants.ts`
2. Actualizar `utils/filters.ts`
3. Actualizar `hooks/useFilters.ts`
4. Actualizar `components/AgentFilters.tsx`

### Agregar nuevo componente UI:
1. Crear en `components/ui/`
2. Exportar en `components/ui/index.ts`
3. Usar en componentes feature

### Agregar nueva acción:
1. Agregar método en `services/agentsService.ts`
2. Crear hook en `hooks/useAgentActions.ts` o nuevo hook
3. Usar en componentes

## 📊 Ventajas de esta Estructura

1. **Ultra-modular**: Cada pieza es independiente
2. **Fácil testing**: Módulos aislados
3. **Reutilizable**: Componentes UI reutilizables
4. **Mantenible**: Código organizado y claro
5. **Escalable**: Fácil agregar funcionalidades
6. **Type-safe**: TypeScript en toda la estructura
7. **Performance**: Memoización donde corresponde

## 🚀 Mejoras Futuras

- [ ] Agregar tests unitarios para cada módulo
- [ ] Agregar Storybook para componentes UI
- [ ] Agregar validación con Zod
- [ ] Agregar error boundaries
- [ ] Agregar optimistic updates
- [ ] Agregar cache con React Query
