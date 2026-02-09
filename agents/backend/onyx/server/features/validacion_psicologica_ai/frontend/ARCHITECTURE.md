# Arquitectura del Frontend - Validación Psicológica AI

## 📋 Visión General

Frontend moderno construido con Next.js 14, React, TypeScript y TailwindCSS, siguiendo las mejores prácticas de desarrollo y principios de diseño accesible.

## 🏗️ Arquitectura

### Stack Tecnológico

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Type safety completo
- **TailwindCSS**: Estilos utility-first
- **React Query**: Gestión de estado del servidor y caché
- **React Hook Form**: Manejo de formularios
- **Zod**: Validación de esquemas
- **Recharts**: Visualización de datos
- **Lucide React**: Iconos

### Principios de Diseño

1. **Type Safety**: Todos los componentes y funciones están completamente tipados
2. **DRY (Don't Repeat Yourself)**: Componentes reutilizables y hooks compartidos
3. **Accesibilidad**: ARIA labels, navegación por teclado, soporte para lectores de pantalla
4. **Early Returns**: Código más legible con retornos tempranos
5. **Naming Conventions**: Funciones de eventos con prefijo "handle"

## 📁 Estructura de Directorios

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Layout raíz
│   ├── page.tsx                 # Página principal (Dashboard)
│   ├── providers.tsx            # Providers globales (React Query, Toast)
│   ├── globals.css              # Estilos globales y variables CSS
│   ├── connections/             # Página de conexiones
│   │   └── page.tsx
│   └── validations/             # Páginas de validaciones
│       └── [id]/
│           └── page.tsx         # Página de detalle
│
├── components/
│   ├── ui/                      # Componentes UI base reutilizables
│   │   ├── Button.tsx          # Botón con variantes y estados
│   │   ├── Input.tsx            # Input con label y validación
│   │   ├── Card.tsx             # Card con subcomponentes
│   │   ├── LoadingSpinner.tsx   # Spinner de carga
│   │   └── index.ts            # Barrel export
│   │
│   └── features/                # Componentes específicos de features
│       ├── ValidationForm.tsx   # Formulario de creación
│       ├── ValidationList.tsx   # Lista de validaciones
│       ├── ProfileDisplay.tsx   # Visualización de perfil
│       ├── PersonalityChart.tsx # Gráfico de personalidad
│       └── index.ts            # Barrel export
│
├── hooks/                       # Custom React hooks
│   ├── useValidations.ts        # Hooks para validaciones
│   ├── useConnections.ts        # Hooks para conexiones
│   └── index.ts                # Barrel export
│
├── lib/
│   ├── api/                     # Cliente API
│   │   ├── client.ts           # Configuración Axios
│   │   ├── validations.ts      # Endpoints de validaciones
│   │   ├── connections.ts       # Endpoints de conexiones
│   │   ├── profiles.ts         # Endpoints de perfiles
│   │   ├── reports.ts          # Endpoints de reportes
│   │   └── index.ts            # Barrel export
│   │
│   ├── types/                   # TypeScript types
│   │   └── index.ts            # Types matching backend schemas
│   │
│   ├── config/                  # Configuración
│   │   └── env.ts              # Variables de entorno
│   │
│   └── utils/                   # Utilidades
│       └── cn.ts               # Merge de clases Tailwind
│
└── package.json                 # Dependencias
```

## 🔄 Flujo de Datos

### Data Fetching

1. **React Query** maneja todo el estado del servidor
2. Los hooks personalizados (`useValidations`, `useConnections`) encapsulan la lógica de fetching
3. Los componentes consumen los hooks directamente
4. Invalidación automática de caché después de mutaciones

### Estado Local

- Estado de formularios: React Hook Form
- Estado de UI: useState para estados simples
- Estado compartido: React Query cache

## 🎨 Sistema de Diseño

### Componentes UI Base

Todos los componentes UI siguen estos principios:

- **Accesibilidad**: ARIA labels, roles, estados de focus
- **Variantes**: Sistema de variantes usando class-variance-authority
- **Composición**: Componentes compuestos (Card, CardHeader, etc.)
- **Type Safety**: Props completamente tipados

### Estilos

- **TailwindCSS**: Utility-first CSS
- **Variables CSS**: Tema configurable mediante variables CSS
- **Dark Mode Ready**: Variables preparadas para modo oscuro
- **Responsive**: Mobile-first approach

## 🔌 Integración con Backend

### API Client

- **Axios**: Cliente HTTP configurado
- **Interceptores**: Manejo automático de autenticación y errores
- **Type Safety**: Todos los endpoints tipados con TypeScript

### Endpoints Principales

- `/psychological-validation/validations` - CRUD de validaciones
- `/psychological-validation/connect` - Gestión de conexiones
- `/psychological-validation/profile/{id}` - Perfiles psicológicos
- `/psychological-validation/report/{id}` - Reportes

## ♿ Accesibilidad

### Implementaciones

1. **ARIA Labels**: Todos los elementos interactivos tienen labels descriptivos
2. **Keyboard Navigation**: Navegación completa por teclado
3. **Focus Management**: Estados de focus visibles y lógicos
4. **Screen Readers**: Estructura semántica correcta
5. **Error Handling**: Mensajes de error accesibles

### Ejemplos

```tsx
// Botón accesible
<button
  aria-label="Desconectar Facebook"
  aria-pressed={isSelected}
  onKeyDown={handleKeyDown}
  tabIndex={0}
>
  Desconectar
</button>

// Input con error accesible
<input
  aria-invalid={error ? 'true' : 'false'}
  aria-describedby={error ? 'input-error' : undefined}
/>
{error && (
  <p id="input-error" role="alert">
    {error}
  </p>
)}
```

## 🧪 Testing (Preparado)

La estructura está preparada para testing con:

- **Jest**: Framework de testing
- **React Testing Library**: Testing de componentes
- **TypeScript**: Type checking en tests

## 📦 Build y Deployment

### Scripts Disponibles

```bash
npm run dev      # Desarrollo
npm run build    # Build de producción
npm run start    # Servidor de producción
npm run lint     # Linting
npm run type-check # Verificación de tipos
```

### Optimizaciones

- **Next.js Image**: Optimización automática de imágenes
- **Code Splitting**: Automático con Next.js
- **Tree Shaking**: Eliminación de código no usado
- **Minification**: Minificación automática en build

## 🔐 Seguridad

- **Token Storage**: localStorage para tokens (considerar httpOnly cookies en producción)
- **XSS Protection**: React escapa automáticamente
- **CSRF**: Protección mediante tokens en headers
- **HTTPS**: Requerido en producción

## 📈 Performance

### Optimizaciones Implementadas

1. **React Query Caching**: Caché inteligente de datos
2. **Code Splitting**: Lazy loading de rutas
3. **Image Optimization**: Next.js Image component
4. **Bundle Size**: Dependencias optimizadas

### Métricas a Monitorear

- First Contentful Paint (FCP)
- Largest Contentful Paint (LCP)
- Time to Interactive (TTI)
- Cumulative Layout Shift (CLS)

## 🚀 Próximas Mejoras

- [ ] PWA support
- [ ] Offline mode
- [ ] Real-time updates con WebSocket
- [ ] Exportación de reportes (PDF, Excel)
- [ ] Dashboard con métricas avanzadas
- [ ] Comparación de validaciones
- [ ] Filtros y búsqueda avanzada




