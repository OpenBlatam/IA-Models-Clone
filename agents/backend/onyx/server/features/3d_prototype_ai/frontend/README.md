# 3D Prototype AI - Frontend

Frontend completo para el sistema de generación de prototipos 3D, construido con Next.js 15, TypeScript y Turbopack.

## 🚀 Características

- ✅ **Next.js 15** con App Router
- ✅ **TypeScript** para type safety
- ✅ **Turbopack** para desarrollo rápido
- ✅ **React Query** para manejo de estado del servidor
- ✅ **Zustand** para estado global
- ✅ **Tailwind CSS** para estilos
- ✅ **Framer Motion** para animaciones
- ✅ **Recharts** para visualizaciones
- ✅ **React Hook Form** + **Zod** para formularios
- ✅ **Three.js** para visualización 3D (preparado)

## 📦 Instalación

```bash
cd frontend
npm install
```

## 🛠️ Desarrollo

```bash
npm run dev
```

El servidor de desarrollo estará disponible en `http://localhost:3000`

## 🏗️ Estructura del Proyecto

```
frontend/
├── src/
│   ├── app/              # Páginas y rutas (App Router)
│   │   ├── dashboard/    # Dashboard principal
│   │   ├── prototypes/    # Gestión de prototipos
│   │   ├── marketplace/  # Marketplace
│   │   ├── analytics/     # Analytics
│   │   └── auth/         # Autenticación
│   ├── components/       # Componentes React
│   │   ├── layout/       # Componentes de layout
│   │   └── prototypes/   # Componentes de prototipos
│   ├── services/         # Servicios API
│   ├── store/            # Estado global (Zustand)
│   ├── types/            # Tipos TypeScript
│   └── lib/              # Utilidades
├── public/               # Archivos estáticos
└── package.json
```

## 🔌 Configuración

Crea un archivo `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8030
NEXT_PUBLIC_WS_URL=ws://localhost:8030
```

## 📱 Páginas Principales

- `/` - Página de inicio
- `/dashboard` - Dashboard con estadísticas
- `/prototypes` - Lista de prototipos
- `/prototypes/create` - Crear nuevo prototipo
- `/marketplace` - Marketplace de prototipos
- `/analytics` - Analytics y reportes
- `/auth/login` - Iniciar sesión
- `/auth/register` - Registrarse

## 🎨 Componentes Principales

- `Header` - Navegación principal
- `PrototypeForm` - Formulario para crear prototipos
- `PrototypeView` - Visualización de prototipos
- `MaterialCard` - Tarjeta de material
- `CADPartCard` - Tarjeta de parte CAD
- `AssemblyStepCard` - Tarjeta de paso de ensamblaje
- `BudgetOptionCard` - Tarjeta de opción de presupuesto

## 🔐 Autenticación

El sistema incluye autenticación completa con:
- Login
- Registro
- Gestión de sesión
- Protección de rutas
- Token management

## 📊 Servicios API

Todos los servicios están en `src/services/`:
- `prototype.service.ts` - Gestión de prototipos
- `auth.service.ts` - Autenticación
- `analytics.service.ts` - Analytics
- `marketplace.service.ts` - Marketplace
- `gamification.service.ts` - Gamificación
- `notification.service.ts` - Notificaciones

## 🎯 Próximas Mejoras

- [ ] Visualización 3D con Three.js
- [ ] Chat en tiempo real
- [ ] Notificaciones push
- [ ] Exportación de PDF
- [ ] Modo oscuro
- [ ] Internacionalización (i18n)
- [ ] PWA support
- [ ] Optimización de imágenes
- [ ] Tests E2E

## 📄 Licencia

Propietaria - Blatam Academy



