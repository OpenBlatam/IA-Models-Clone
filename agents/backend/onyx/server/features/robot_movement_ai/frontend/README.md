# Robot Movement AI - Frontend

Frontend Next.js TypeScript para la plataforma Robot Movement AI. Interfaz moderna y completa para controlar robots mediante chat y comandos directos.

## 🚀 Características

- **Control de Robot**: Interfaz intuitiva para mover el robot a posiciones específicas
- **Chat en Tiempo Real**: Comunicación con el robot mediante WebSocket y REST API
- **Dashboard Completo**: Visualización de estado, métricas y estadísticas
- **Monitoreo en Tiempo Real**: Gráficos y métricas del sistema
- **Diseño Moderno**: UI moderna con Tailwind CSS y tema oscuro

## 📦 Instalación

### Prerrequisitos

- Node.js 18+ 
- npm o yarn

### Pasos

1. Instalar dependencias:
```bash
npm install
# o
yarn install
```

2. Configurar variables de entorno:
```bash
cp .env.example .env.local
```

Editar `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8010
```

3. Ejecutar en desarrollo:
```bash
npm run dev
# o
yarn dev
```

4. Abrir en el navegador:
```
http://localhost:3000
```

## 🏗️ Estructura del Proyecto

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Layout principal
│   ├── page.tsx           # Página principal
│   └── globals.css        # Estilos globales
├── components/             # Componentes React
│   ├── Dashboard.tsx      # Dashboard principal
│   ├── ChatPanel.tsx      # Panel de chat
│   ├── RobotControl.tsx   # Control del robot
│   ├── StatusPanel.tsx    # Panel de estado
│   └── MetricsPanel.tsx   # Panel de métricas
├── lib/                    # Utilidades y lógica
│   ├── api/               # Cliente API
│   │   ├── client.ts      # Cliente REST
│   │   ├── websocket.ts   # Cliente WebSocket
│   │   └── types.ts       # Tipos TypeScript
│   └── store/             # Estado global (Zustand)
│       └── robotStore.ts  # Store del robot
└── public/                 # Archivos estáticos
```

## 🎯 Funcionalidades

### Control del Robot

- Mover a posición absoluta (X, Y, Z)
- Detener movimiento
- Ir a posición home
- Posiciones predefinidas

### Chat

- Comandos de texto natural
- WebSocket para tiempo real
- Historial de conversación
- Ejemplos de comandos:
  - `move to (0.5, 0.3, 0.2)`
  - `go home`
  - `stop`
  - `status`

### Estado y Monitoreo

- Estado del robot en tiempo real
- Posición actual
- Ángulos de articulación
- Estado de salud del sistema
- Métricas de CPU y memoria
- Gráficos de rendimiento

## 🔌 API Endpoints Utilizados

El frontend se conecta a todos los endpoints del backend:

- `/api/v1/move/to` - Mover robot
- `/api/v1/chat` - Chat REST
- `/api/v1/status` - Estado del robot
- `/api/v1/stop` - Detener robot
- `/api/v1/statistics` - Estadísticas
- `/api/v1/metrics` - Métricas
- `/api/v1/resources` - Recursos del sistema
- `/api/v1/monitoring` - Monitoreo
- `/ws/chat` - WebSocket para chat

## 🛠️ Tecnologías

- **Next.js 14** - Framework React
- **TypeScript** - Tipado estático
- **Tailwind CSS** - Estilos
- **Zustand** - Gestión de estado
- **Axios** - Cliente HTTP
- **Recharts** - Gráficos
- **Lucide React** - Iconos

## 📝 Scripts

```bash
# Desarrollo
npm run dev

# Producción
npm run build
npm start

# Linting
npm run lint

# Type checking
npm run type-check
```

## 🌐 Variables de Entorno

- `NEXT_PUBLIC_API_URL` - URL del backend API (default: http://localhost:8010)

## 🎨 Personalización

El tema puede personalizarse en `tailwind.config.js`. Los colores principales están definidos en la configuración de Tailwind.

## 📚 Documentación

Para más información sobre el backend, consulta el README principal del proyecto.

## 🤝 Contribuir

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

## 📄 Licencia

Copyright (c) 2025 Blatam Academy

