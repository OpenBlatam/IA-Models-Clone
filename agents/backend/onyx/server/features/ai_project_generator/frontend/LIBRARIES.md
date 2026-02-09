# Librerías y Dependencias

## Dependencias Principales

### Framework y Core
- **next** (^14.2.5) - Framework React con SSR y optimizaciones
- **react** (^18.3.1) - Biblioteca UI
- **react-dom** (^18.3.1) - Renderizado React para DOM

### HTTP y API
- **axios** (^1.7.2) - Cliente HTTP para peticiones API
- **@tanstack/react-query** (^5.28.9) - Manejo de estado del servidor, caché y sincronización

### UI y Estilos
- **tailwindcss** (^3.4.4) - Framework CSS utility-first
- **tailwind-merge** (^2.3.0) - Merge inteligente de clases Tailwind
- **clsx** (^2.1.1) - Utilidad para construir className condicionales
- **lucide-react** (^0.400.0) - Iconos SVG optimizados
- **framer-motion** (^11.3.6) - Biblioteca de animaciones para React

### Formularios y Validación
- **react-hook-form** (^7.51.5) - Manejo de formularios performante
- **@hookform/resolvers** (^3.3.4) - Resolvers para react-hook-form
- **zod** (^3.23.8) - Validación de esquemas TypeScript-first

### Estado y Utilidades
- **zustand** (^4.5.2) - Estado global ligero y simple
- **date-fns** (^3.6.0) - Utilidades para manipulación de fechas
- **react-intersection-observer** (^9.8.1) - Hook para detectar cuando elementos entran en viewport

### Notificaciones
- **react-hot-toast** (^2.4.1) - Sistema de notificaciones toast elegante

## DevDependencies

### TypeScript
- **typescript** (^5.5.3) - Superset tipado de JavaScript
- **@types/node** (^20.14.10) - Tipos para Node.js
- **@types/react** (^18.3.3) - Tipos para React
- **@types/react-dom** (^18.3.0) - Tipos para React DOM

### Linting y Formateo
- **eslint** (^8.57.0) - Linter de JavaScript/TypeScript
- **eslint-config-next** (^14.2.5) - Configuración ESLint para Next.js
- **@typescript-eslint/eslint-plugin** (^7.13.1) - Plugin ESLint para TypeScript
- **@typescript-eslint/parser** (^7.13.1) - Parser ESLint para TypeScript
- **eslint-config-prettier** (^9.1.0) - Desactiva reglas ESLint que conflictúan con Prettier
- **eslint-plugin-react** (^7.34.2) - Plugin ESLint para React
- **eslint-plugin-react-hooks** (^4.6.2) - Plugin ESLint para React Hooks
- **prettier** (^3.3.2) - Formateador de código
- **prettier-plugin-tailwindcss** (^0.5.11) - Plugin Prettier para ordenar clases Tailwind

### Build Tools
- **postcss** (^8.4.39) - Transformador CSS
- **autoprefixer** (^10.4.19) - Agrega prefijos de vendor automáticamente

## Beneficios de las Librerías

### React Query (@tanstack/react-query)
- Caché automático de datos
- Sincronización en background
- Manejo de estados de carga y error
- Refetch inteligente
- Optimistic updates

### Zod
- Validación type-safe
- Inferencia automática de tipos TypeScript
- Mensajes de error personalizables
- Validación en runtime y compile-time

### React Hook Form
- Mejor rendimiento (menos re-renders)
- Validación integrada
- Fácil integración con Zod
- Manejo de errores robusto

### Framer Motion
- Animaciones fluidas y performantes
- API declarativa
- Gestos y transiciones
- Optimizado para React

### Zustand
- Estado global simple
- Sin boilerplate
- TypeScript-first
- Pequeño bundle size

### React Hot Toast
- Notificaciones elegantes
- Fácil de usar
- Personalizable
- Accesible

## Scripts Disponibles

- `npm run dev` - Inicia servidor de desarrollo
- `npm run build` - Construye para producción
- `npm run start` - Inicia servidor de producción
- `npm run lint` - Ejecuta ESLint
- `npm run lint:fix` - Ejecuta ESLint y corrige automáticamente
- `npm run type-check` - Verifica tipos TypeScript
- `npm run format` - Formatea código con Prettier
- `npm run format:check` - Verifica formato sin modificar

