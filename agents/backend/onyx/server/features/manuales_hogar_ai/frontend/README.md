# Manuales Hogar AI - Frontend

Frontend moderno en Next.js 14 con TypeScript para el sistema de generación de manuales paso a paso tipo LEGO.

## 🚀 Características

- ✅ Next.js 14 con App Router y Turbo Pack
- ✅ TypeScript para type safety completo
- ✅ Tailwind CSS para estilos modernos
- ✅ Integración completa con la API del backend
- ✅ React Query para gestión de estado del servidor
- ✅ Componentes reutilizables con Radix UI
- ✅ Manejo de archivos (imágenes)
- ✅ Búsqueda avanzada (simple, semántica, avanzada)
- ✅ Sistema de calificaciones y favoritos
- ✅ Historial de manuales
- ✅ Accesibilidad completa (ARIA labels, keyboard navigation)

## 📦 Instalación

```bash
npm install
```

## 🏃 Desarrollo

```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## 🔧 Configuración

Asegúrate de que el backend esté corriendo en `http://localhost:8000`

Puedes cambiar la URL del backend creando un archivo `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📁 Estructura

```
frontend/
├── app/                    # App Router de Next.js
│   ├── layout.tsx         # Layout principal
│   ├── page.tsx           # Página principal
│   ├── history/           # Página de historial
│   ├── search/            # Página de búsqueda
│   ├── manual/[id]/       # Página de detalle de manual
│   ├── providers.tsx      # React Query Provider
│   └── globals.css        # Estilos globales
├── components/            # Componentes React
│   ├── ui/               # Componentes de UI base
│   ├── navigation.tsx     # Navegación principal
│   ├── manual-generator.tsx  # Generador de manuales
│   ├── manuals-list.tsx  # Lista de manuales
│   ├── search-panel.tsx  # Panel de búsqueda
│   ├── manual-detail.tsx # Detalle de manual
│   └── recent-manuals.tsx # Manuales recientes
├── lib/                  # Utilidades y configuraciones
│   ├── api/              # Cliente API
│   ├── hooks/            # React Query hooks
│   ├── types/            # Tipos TypeScript
│   └── utils/            # Utilidades
└── public/               # Archivos estáticos
```

## 🎨 Tecnologías

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Type safety
- **Tailwind CSS**: Estilos
- **React Query**: Gestión de estado del servidor
- **Axios**: Cliente HTTP
- **React Hook Form**: Manejo de formularios
- **Zod**: Validación de esquemas
- **Radix UI**: Componentes accesibles
- **Lucide React**: Iconos
- **React Hot Toast**: Notificaciones
- **date-fns**: Manejo de fechas

## 📝 Funcionalidades Implementadas

### Generación de Manuales
- Generación desde texto
- Generación desde imagen
- Generación desde múltiples imágenes (hasta 5)
- Generación combinada (texto + imagen)
- Selección de categoría y modelo de IA
- Opciones configurables (seguridad, herramientas, materiales)

### Búsqueda
- Búsqueda simple
- Búsqueda semántica
- Búsqueda avanzada
- Filtrado por categoría

### Historial
- Lista de todos los manuales
- Paginación
- Filtrado por categoría
- Búsqueda en historial

### Detalle de Manual
- Visualización completa del manual
- Sistema de calificaciones (1-5 estrellas)
- Comentarios
- Favoritos
- Lista de calificaciones

## 🎯 Mejores Prácticas Implementadas

- ✅ Early returns para mejor legibilidad
- ✅ Tailwind classes exclusivamente (sin CSS inline)
- ✅ Consts en lugar de functions
- ✅ Nombres descriptivos con prefijo "handle" para eventos
- ✅ Accesibilidad completa (ARIA labels, tabindex, keyboard navigation)
- ✅ TypeScript estricto
- ✅ Componentes reutilizables
- ✅ Manejo de errores robusto
- ✅ Estados de carga
- ✅ Validación de formularios con Zod

## 🚀 Build para Producción

```bash
npm run build
npm start
```

## 📄 Licencia

Propietaria - Blatam Academy

