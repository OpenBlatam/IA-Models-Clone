# Artist Manager AI - Frontend

Frontend completo para el sistema Artist Manager AI construido con Next.js 14, TypeScript, Turbo Pack y Tailwind CSS.

## Características

- ✅ Next.js 14 con Turbo Pack
- ✅ TypeScript completo
- ✅ Tailwind CSS para estilos
- ✅ React Query para gestión de estado del servidor
- ✅ Componentes UI accesibles
- ✅ Integración completa con la API backend
- ✅ Hooks personalizados para todas las funcionalidades
- ✅ Diseño responsive y moderno

## Instalación

```bash
npm install
```

## Configuración

Crear archivo `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Desarrollo

```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## Build

```bash
npm run build
npm start
```

## Estructura del Proyecto

```
frontend/
├── src/
│   ├── app/              # Páginas de Next.js
│   │   ├── dashboard/    # Dashboard principal
│   │   ├── calendar/     # Gestión de calendario
│   │   ├── routines/     # Gestión de rutinas
│   │   ├── protocols/    # Gestión de protocolos
│   │   └── wardrobe/     # Gestión de guardarropa
│   ├── components/       # Componentes React
│   │   ├── ui/           # Componentes UI base
│   │   └── layout/       # Componentes de layout
│   ├── hooks/            # React Hooks personalizados
│   ├── lib/              # Utilidades y cliente API
│   └── types/            # Tipos TypeScript
├── public/               # Archivos estáticos
└── package.json
```

## Funcionalidades Implementadas

### Dashboard
- Vista general con estadísticas
- Resumen diario generado por IA
- Eventos próximos
- Rutinas pendientes

### Calendario
- Lista de eventos
- Crear, editar y eliminar eventos
- Recomendaciones de vestimenta por evento
- Filtros por fecha y días

### Rutinas
- Gestión completa de rutinas
- Completar rutinas
- Rutinas pendientes
- Tasa de completación

### Protocolos
- Gestión de protocolos
- Verificación de cumplimiento
- Categorización y prioridades
- Auditoría de cumplimiento

### Guardarropa
- Gestión de items
- Gestión de outfits
- Marcado de items usados
- Filtros por categoría y código de vestimenta

## Tecnologías

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Tipado estático
- **Tailwind CSS**: Estilos utility-first
- **React Query**: Gestión de estado del servidor
- **Axios**: Cliente HTTP
- **Lucide React**: Iconos
- **Radix UI**: Componentes accesibles

## Licencia

Propietaria - Blatam Academy

