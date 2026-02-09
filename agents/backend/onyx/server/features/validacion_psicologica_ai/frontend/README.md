# Validación Psicológica AI - Frontend

Frontend moderno para el sistema de Validación Psicológica AI, construido con Next.js 14, React, TypeScript y TailwindCSS.

## 🚀 Características

- **Next.js 14** con App Router
- **TypeScript** para type safety completo
- **TailwindCSS** para estilos modernos
- **React Query** para gestión de estado del servidor
- **React Hook Form** con validación Zod
- **Componentes accesibles** con ARIA labels y navegación por teclado
- **UI moderna** con componentes reutilizables

## 📁 Estructura

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Layout principal
│   ├── page.tsx           # Página de inicio (Dashboard)
│   ├── providers.tsx      # Providers globales
│   ├── globals.css        # Estilos globales
│   └── validations/       # Páginas de validaciones
│       └── [id]/          # Página de detalle
├── components/
│   ├── ui/                # Componentes UI base
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Card.tsx
│   │   └── LoadingSpinner.tsx
│   └── features/          # Componentes específicos
│       ├── ValidationForm.tsx
│       ├── ValidationList.tsx
│       ├── ProfileDisplay.tsx
│       └── PersonalityChart.tsx
├── hooks/                 # Custom hooks
│   ├── useValidations.ts
│   └── useConnections.ts
├── lib/
│   ├── api/               # Cliente API
│   │   ├── client.ts
│   │   ├── validations.ts
│   │   ├── connections.ts
│   │   ├── profiles.ts
│   │   └── reports.ts
│   ├── types/             # TypeScript types
│   │   └── index.ts
│   ├── config/            # Configuración
│   │   └── env.ts
│   └── utils/             # Utilidades
│       └── cn.ts
└── package.json
```

## 🛠️ Instalación

```bash
cd agents/backend/onyx/server/features/validacion_psicologica_ai/frontend
npm install
```

## 🚀 Desarrollo

```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## 🔧 Configuración

Configura las variables de entorno en `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## 📝 Componentes Principales

### ValidationForm
Formulario para crear nuevas validaciones con selección de plataformas de redes sociales.

### ValidationList
Lista de todas las validaciones con estado y información básica.

### ProfileDisplay
Visualización completa del perfil psicológico con gráficos y recomendaciones.

### PersonalityChart
Gráfico de barras para visualizar los rasgos de personalidad Big Five.

## 🎨 Estilos

El proyecto usa TailwindCSS con variables CSS para temas. Los colores se definen en `globals.css` y pueden ser personalizados fácilmente.

## ♿ Accesibilidad

Todos los componentes incluyen:
- ARIA labels apropiados
- Navegación por teclado
- Estados de focus visibles
- Soporte para lectores de pantalla

## 📦 Dependencias Principales

- **next**: Framework React
- **react**: Biblioteca UI
- **@tanstack/react-query**: Gestión de estado del servidor
- **react-hook-form**: Manejo de formularios
- **zod**: Validación de esquemas
- **recharts**: Gráficos
- **lucide-react**: Iconos
- **tailwindcss**: Estilos

## 🔗 API Integration

El frontend se conecta con el backend a través de los endpoints definidos en `/lib/api/`. Todos los endpoints están tipados con TypeScript para garantizar type safety.




