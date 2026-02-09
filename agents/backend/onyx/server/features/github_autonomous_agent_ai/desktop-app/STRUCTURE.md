# Estructura del Proyecto Desktop

## 📁 Organización de Carpetas

```
desktop-app/
├── src/
│   ├── main/                    # Proceso principal de Electron (Node.js)
│   │   ├── main.ts             # Punto de entrada, crea ventanas
│   │   └── preload.ts          # Script de preload (puente seguro)
│   │
│   ├── renderer/                # Proceso de renderizado (React/TypeScript)
│   │   ├── components/         # Componentes React reutilizables
│   │   │   └── Toaster.tsx    # Componente de notificaciones
│   │   ├── pages/              # Páginas principales de la app
│   │   │   ├── MainPage.tsx
│   │   │   ├── AgentControlPage.tsx
│   │   │   ├── KanbanPage.tsx
│   │   │   └── ContinuousAgentPage.tsx
│   │   ├── styles/             # Estilos globales
│   │   │   └── globals.css
│   │   ├── utils/              # Utilidades
│   │   │   └── cn.ts           # Utilidad para clases CSS
│   │   ├── App.tsx             # Componente raíz
│   │   ├── main.tsx            # Punto de entrada del renderer
│   │   └── index.html          # HTML base
│   │
│   └── shared/                  # Código compartido
│       └── config.ts           # Configuración compartida
│
├── build/                       # Recursos de build
│   ├── icon.ico                # Icono Windows (requerido)
│   └── icon.icns               # Icono macOS (requerido)
│
├── scripts/                     # Scripts de automatización
│   ├── build-windows.bat       # Build para Windows
│   └── build-mac.sh            # Build para macOS
│
├── dist/                        # Archivos compilados (generado)
│   ├── main/                   # Código compilado del proceso principal
│   └── renderer/               # Código compilado del renderer
│
├── release/                     # Instaladores generados (generado)
│
├── package.json                 # Configuración del proyecto
├── tsconfig.json               # Configuración TypeScript general
├── tsconfig.electron.json      # Configuración TypeScript para Electron
├── vite.config.ts              # Configuración de Vite
├── tailwind.config.js          # Configuración de Tailwind CSS
└── postcss.config.js           # Configuración de PostCSS
```

## 🔄 Flujo de Compilación

### Desarrollo
1. `npm run dev` inicia:
   - TypeScript compiler en modo watch (`tsc -w`) para `src/main/`
   - Vite dev server para `src/renderer/`
   - Electron carga desde `dist/main/main.js` y `http://localhost:3000`

### Producción
1. `npm run build` ejecuta:
   - `npm run build:electron` → Compila `src/main/` → `dist/main/`
   - `npm run build:renderer` → Compila `src/renderer/` → `dist/renderer/`

2. `npm run build:win` o `npm run build:mac`:
   - Ejecuta `npm run build` primero
   - Usa electron-builder para crear instaladores
   - Genera archivos en `release/`

## 🔐 Seguridad

### Context Isolation
- El proceso `main` (Node.js) y `renderer` (React) están aislados
- Comunicación solo a través de IPC (Inter-Process Communication)
- `preload.ts` actúa como puente seguro

### Preload Script
- Se ejecuta antes de que el renderer cargue
- Expone APIs seguras a través de `contextBridge`
- No tiene acceso directo a Node.js APIs

## 📦 Dependencias Clave

### Runtime
- **electron**: Framework de aplicaciones desktop
- **react/react-dom**: UI framework
- **zustand**: State management
- **@tanstack/react-query**: Data fetching

### Build Tools
- **typescript**: Compilador TypeScript
- **vite**: Build tool para el renderer
- **electron-builder**: Empaquetador de aplicaciones
- **tailwindcss**: Framework CSS

## 🎯 Próximos Pasos

Para completar la migración del frontend Next.js:

1. **Copiar componentes** desde `frontend/app/components/` a `desktop-app/src/renderer/components/`
2. **Copiar hooks** desde `frontend/app/hooks/` a `desktop-app/src/renderer/hooks/`
3. **Copiar libs** desde `frontend/app/lib/` a `desktop-app/src/renderer/lib/`
4. **Adaptar API routes**: Las rutas de Next.js API deben convertirse en llamadas directas al backend
5. **Configurar API client**: Adaptar el cliente API para funcionar en Electron

## 🔗 Integración con Backend

La aplicación se conecta al backend Python en:
- **API**: `http://localhost:8030` (configurable)
- **WebSocket**: `ws://localhost:8030/ws` (configurable)

Configura estas URLs en `src/shared/config.ts` o a través de variables de entorno.


