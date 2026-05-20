# Estructura Modular del Frontend

## Organización de Carpetas

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Layout principal
│   ├── page.tsx                 # Página principal
│   └── globals.css              # Estilos globales
│
├── components/
│   ├── ui/                      # Componentes UI reutilizables
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Card.tsx
│   │   ├── Badge.tsx
│   │   ├── Modal.tsx
│   │   ├── Toast.tsx
│   │   └── index.ts            # Barrel exports
│   │
│   ├── features/                # Componentes de características
│   │   ├── ProjectGeneratorForm/
│   │   │   ├── index.tsx       # Componente principal
│   │   │   └── formFields/    # Campos del formulario modulares
│   │   │       ├── DescriptionField.tsx
│   │   │       ├── BasicInfoFields.tsx
│   │   │       ├── PriorityField.tsx
│   │   │       ├── OptionsFields.tsx
│   │   │       ├── TagsField.tsx
│   │   │       └── index.ts
│   │   ├── ProjectCard.tsx
│   │   ├── ProjectQueue.tsx
│   │   ├── ProjectList.tsx
│   │   ├── Statistics.tsx
│   │   └── index.ts
│   │
│   └── layout/                  # Componentes de layout
│       ├── Header.tsx
│       ├── Navigation.tsx
│       └── index.ts
│
├── hooks/
│   ├── api/                     # Hooks de API
│   │   ├── useDashboardData.ts
│   │   ├── useProjectGenerator.ts
│   │   ├── useGeneratorControl.ts
│   │   └── index.ts
│   │
│   ├── ui/                      # Hooks de UI
│   │   ├── useDebounce.ts
│   │   ├── useToast.ts
│   │   ├── usePagination.ts
│   │   ├── useSearch.ts
│   │   └── index.ts
│   │
│   ├── forms/                   # Hooks de formularios
│   │   ├── useProjectForm.ts
│   │   └── index.ts
│   │
│   └── useWebSocket.ts          # Hook de WebSocket
│
├── lib/
│   ├── api/                     # Cliente API
│   │   ├── index.ts
│   │   └── api.ts
│   │
│   ├── config/                  # Configuraciones
│   │   ├── theme.ts            # Configuración de tema
│   │   ├── formConfig.ts       # Configuración de formularios
│   │   └── index.ts
│   │
│   ├── constants/               # Constantes
│   │   └── index.ts
│   │
│   └── utils/                   # Utilidades
│       ├── index.ts
│       ├── validation.ts       # Funciones de validación
│       └── format.ts           # Funciones de formateo
│
└── types/                       # Tipos TypeScript
    └── index.ts
```

## Principios de Modularización

### 1. Separación de Responsabilidades
- **UI Components**: Componentes puros de presentación
- **Feature Components**: Componentes con lógica de negocio específica
- **Layout Components**: Componentes de estructura de página
- **Hooks**: Lógica reutilizable separada de componentes

### 2. Componentes Modulares
- Cada campo del formulario es un componente independiente
- Campos agrupados lógicamente (BasicInfoFields, OptionsFields)
- Fácil de mantener y testear

### 3. Configuración Centralizada
- `lib/config/theme.ts`: Configuración de tema
- `lib/config/formConfig.ts`: Configuración de formularios
- `lib/constants/`: Constantes de la aplicación

### 4. Hooks Personalizados
- `useProjectForm`: Manejo de estado y validación del formulario
- `useToast`: Sistema de notificaciones
- `usePagination`: Paginación de datos
- `useSearch`: Búsqueda y filtrado

### 5. Barrel Exports
- Cada carpeta tiene un `index.ts` para exports centralizados
- Imports más limpios: `from '@/components/ui'` en lugar de rutas largas

## Beneficios

1. **Mantenibilidad**: Código organizado y fácil de encontrar
2. **Reutilización**: Componentes y hooks reutilizables
3. **Testabilidad**: Componentes pequeños y aislados
4. **Escalabilidad**: Fácil agregar nuevas características
5. **Legibilidad**: Código más claro y fácil de entender

