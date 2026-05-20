# Community Manager AI - Frontend

Frontend completo para el sistema de gestión automatizada de redes sociales con IA.

## 🚀 Características

- **Dashboard**: Vista general con estadísticas y gráficos
- **Posts**: Gestión completa de publicaciones (crear, editar, programar, publicar)
- **Memes**: Biblioteca de memes con búsqueda y categorización
- **Calendario**: Vista semanal de publicaciones programadas
- **Plataformas**: Conexión y gestión de redes sociales
- **Analytics**: Métricas y análisis de rendimiento
- **Plantillas**: Sistema de plantillas de contenido reutilizables

## 🛠️ Tecnologías

- **Next.js 14**: Framework React con App Router
- **TypeScript**: Tipado estático
- **TailwindCSS**: Estilos utilitarios
- **React Hook Form**: Manejo de formularios
- **Recharts**: Gráficos y visualizaciones
- **Axios**: Cliente HTTP
- **Lucide React**: Iconos
- **date-fns**: Manejo de fechas

## 📦 Instalación

```bash
# Instalar dependencias
npm install

# Ejecutar en desarrollo
npm run dev

# Construir para producción
npm run build

# Iniciar en producción
npm start
```

## ⚙️ Configuración

Crea un archivo `.env.local` en la raíz del proyecto:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📁 Estructura del Proyecto

```
frontend/
├── app/                    # Páginas y rutas (App Router)
│   ├── dashboard/          # Dashboard principal
│   ├── posts/              # Gestión de posts
│   ├── memes/              # Gestión de memes
│   ├── calendar/           # Calendario
│   ├── platforms/          # Conexiones de plataformas
│   ├── analytics/          # Analytics y métricas
│   ├── templates/          # Plantillas
│   └── settings/           # Configuración
├── components/             # Componentes reutilizables
│   ├── layout/             # Layout, Sidebar, Header
│   └── ui/                 # Componentes UI (Button, Card, Modal)
├── lib/                    # Utilidades y API client
│   ├── api.ts              # Cliente API
│   └── utils.ts            # Funciones auxiliares
├── types/                  # Tipos TypeScript
└── public/                 # Archivos estáticos
```

## 🎨 Componentes Principales

### Layout
- `Sidebar`: Navegación lateral
- `Header`: Barra superior con búsqueda y notificaciones
- `Layout`: Contenedor principal

### UI
- `Button`: Botón con variantes
- `Card`: Tarjeta contenedora
- `Modal`: Modal reutilizable

## 🔌 API Client

El cliente API está en `lib/api.ts` y proporciona métodos para todas las operaciones:

- `postsApi`: CRUD de posts
- `memesApi`: Gestión de memes
- `calendarApi`: Eventos del calendario
- `platformsApi`: Conexiones de plataformas
- `analyticsApi`: Métricas y analytics
- `dashboardApi`: Datos del dashboard
- `templatesApi`: Gestión de plantillas

## 🎯 Páginas

### Dashboard
Vista general con:
- Estadísticas principales
- Gráficos de engagement
- Tendencias por plataforma

### Posts
- Lista de posts con filtros
- Crear/editar posts
- Programar publicaciones
- Publicar inmediatamente
- Eliminar posts

### Memes
- Galería de memes
- Búsqueda y filtros
- Subir nuevos memes
- Categorización

### Calendario
- Vista semanal
- Navegación entre semanas
- Eventos programados

### Plataformas
- Lista de plataformas soportadas
- Conectar/desconectar
- Estado de conexión

### Analytics
- Métricas por plataforma
- Tendencias de engagement
- Mejores posts
- Gráficos interactivos

### Plantillas
- Crear/editar plantillas
- Variables dinámicas
- Categorización

## 🎨 Estilos

El proyecto usa TailwindCSS con una configuración personalizada. Los colores principales están definidos en `tailwind.config.ts`.

## 📱 Responsive

Todas las páginas son completamente responsive y se adaptan a diferentes tamaños de pantalla.

## ♿ Accesibilidad

- Navegación por teclado
- Etiquetas ARIA
- Contraste adecuado
- Focus visible

## 🚀 Próximas Mejoras

- [ ] Autenticación y autorización
- [ ] Notificaciones en tiempo real
- [ ] Exportación de reportes
- [ ] Modo oscuro
- [ ] Internacionalización
- [ ] Tests unitarios y E2E

## 📝 Licencia

Propietaria - Blatam Academy



