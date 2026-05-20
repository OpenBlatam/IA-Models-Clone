# Guía de Inicio Rápido - Frontend Dermatology AI

## 🚀 Instalación

1. **Navega al directorio del frontend:**
```bash
cd frontend
```

2. **Instala las dependencias:**
```bash
npm install
```

## ⚙️ Configuración

1. **Copia el archivo de ejemplo de variables de entorno:**
```bash
cp .env.example .env.local
```

2. **Edita `.env.local` y configura la URL de tu backend:**
```env
NEXT_PUBLIC_API_URL=http://localhost:8006
```

## 🏃 Ejecutar en Desarrollo

```bash
npm run dev
```

La aplicación estará disponible en `http://localhost:3000`

## 📋 Prerrequisitos

- Node.js 18+ instalado
- Backend corriendo en `http://localhost:8006` (o la URL que configuraste)

## 🎯 Características Implementadas

### Páginas Principales

- **Página Principal (`/`)**: Análisis de imágenes y obtención de recomendaciones
- **Dashboard (`/dashboard`)**: Vista general con estadísticas y gráficos
- **Historial (`/history`)**: Lista de análisis anteriores
- **Productos (`/products`)**: Búsqueda de productos de skincare
- **Configuración (`/settings`)**: Ajustes de la aplicación

### Componentes

- **ImageUpload**: Componente para subir imágenes con drag & drop
- **AnalysisResults**: Visualización de resultados de análisis
- **RecommendationsDisplay**: Mostrar recomendaciones de skincare
- **Button, Card**: Componentes de UI reutilizables

### Funcionalidades

- ✅ Análisis de imágenes
- ✅ Obtención de recomendaciones
- ✅ Visualización de métricas de calidad
- ✅ Historial de análisis
- ✅ Búsqueda de productos
- ✅ Dashboard con estadísticas
- ✅ Gráficos interactivos

## 🔧 Estructura del Proyecto

```
frontend/
├── app/                    # Páginas Next.js (App Router)
│   ├── page.tsx           # Página principal
│   ├── dashboard/         # Dashboard
│   ├── history/           # Historial
│   ├── products/          # Productos
│   └── settings/          # Configuración
├── components/            # Componentes React
│   ├── ui/               # Componentes base
│   ├── analysis/         # Componentes de análisis
│   └── recommendations/  # Componentes de recomendaciones
├── lib/                  # Utilidades
│   ├── api/              # Cliente API
│   ├── types/            # Tipos TypeScript
│   └── utils/            # Utilidades
└── public/               # Archivos estáticos
```

## 🎨 Tecnologías

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Type safety
- **Tailwind CSS**: Estilos modernos
- **Axios**: Cliente HTTP
- **Recharts**: Gráficos y visualizaciones
- **React Hot Toast**: Notificaciones
- **Lucide React**: Iconos

## 📝 Notas

- El frontend está completamente integrado con todos los endpoints del backend
- Todas las llamadas API están tipadas con TypeScript
- El diseño es responsive y funciona en móviles y desktop
- Los componentes son reutilizables y modulares

## 🐛 Solución de Problemas

### Error de conexión con el backend

1. Verifica que el backend esté corriendo en el puerto correcto
2. Revisa la configuración en `.env.local`
3. Verifica CORS en el backend

### Errores de TypeScript

```bash
npm run type-check
```

### Errores de linting

```bash
npm run lint
```

## 🚀 Build para Producción

```bash
npm run build
npm start
```

## 📚 Documentación Adicional

- Ver `README.md` para más información
- Consulta la documentación del backend para entender los endpoints disponibles

