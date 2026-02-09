# Dermatology AI Frontend

Frontend moderno en Next.js con TypeScript para el sistema de análisis de piel y recomendaciones de skincare.

## 🚀 Características

- ✅ Next.js 14 con App Router
- ✅ TypeScript para type safety
- ✅ Tailwind CSS para estilos modernos
- ✅ Integración completa con la API del backend
- ✅ Componentes reutilizables
- ✅ Manejo de archivos (imágenes/videos)
- ✅ Visualizaciones con gráficos
- ✅ Dashboard interactivo
- ✅ Historial de análisis
- ✅ Sistema de recomendaciones

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

Asegúrate de que el backend esté corriendo en `http://localhost:8006`

Puedes cambiar la URL del backend en `lib/api/config.ts`

## 📁 Estructura

```
frontend/
├── app/                    # App Router de Next.js
│   ├── layout.tsx         # Layout principal
│   ├── page.tsx           # Página principal
│   └── ...
├── components/            # Componentes React
│   ├── ui/               # Componentes de UI base
│   ├── analysis/         # Componentes de análisis
│   └── ...
├── lib/                  # Utilidades y configuraciones
│   ├── api/              # Cliente API
│   └── types/            # Tipos TypeScript
└── public/               # Archivos estáticos
```

## 🎨 Tecnologías

- **Next.js 14**: Framework React
- **TypeScript**: Type safety
- **Tailwind CSS**: Estilos
- **Axios**: Cliente HTTP
- **Recharts**: Gráficos
- **React Hot Toast**: Notificaciones
- **Lucide React**: Iconos

