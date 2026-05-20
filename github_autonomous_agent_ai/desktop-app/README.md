# GitHub Autonomous Agent AI - Desktop Application

Aplicación de escritorio construida con Electron y TypeScript, similar a la arquitectura de VS Code.

## 🚀 Características

- ✅ Aplicación de escritorio multiplataforma (Windows y macOS)
- ✅ Construida con Electron y TypeScript
- ✅ Interfaz moderna con React y Tailwind CSS
- ✅ Auto-actualización integrada
- ✅ Arquitectura similar a VS Code

## 📋 Requisitos Previos

- Node.js 18+ y npm
- Para Windows: Visual Studio Build Tools (para compilar módulos nativos)
- Para macOS: Xcode Command Line Tools

## 🛠️ Instalación

1. Navega al directorio de la aplicación:
```bash
cd desktop-app
```

2. Instala las dependencias:
```bash
npm install
```

## 🏃 Desarrollo

Para ejecutar la aplicación en modo desarrollo:

```bash
npm run dev
```

Esto iniciará:
- El proceso principal de Electron (main)
- El servidor de desarrollo de Vite para el renderer

## 📦 Construcción

### Windows

**Opción 1: Usando el script batch**
```bash
scripts\build-windows.bat
```

**Opción 2: Manualmente**
```bash
npm run build:win
```

El instalador se generará en la carpeta `release/`:
- `GitHub Autonomous Agent AI-{version}-x64.exe` - Instalador NSIS
- `GitHub Autonomous Agent AI-{version}-x64.exe` - Versión portable

### macOS

**Opción 1: Usando el script shell**
```bash
chmod +x scripts/build-mac.sh
./scripts/build-mac.sh
```

**Opción 2: Manualmente**
```bash
npm run build:mac
```

El instalador se generará en la carpeta `release/`:
- `GitHub Autonomous Agent AI-{version}-x64.dmg` - Imagen DMG
- `GitHub Autonomous Agent AI-{version}-x64.zip` - Archivo ZIP

### Ambas Plataformas

```bash
npm run build:all
```

## 📁 Estructura del Proyecto

```
desktop-app/
├── src/
│   ├── main/              # Proceso principal de Electron
│   │   ├── main.ts        # Punto de entrada principal
│   │   └── preload.ts     # Script de preload para seguridad
│   ├── renderer/          # Proceso de renderizado (React)
│   │   ├── components/    # Componentes React
│   │   ├── pages/         # Páginas de la aplicación
│   │   ├── styles/        # Estilos globales
│   │   ├── utils/         # Utilidades
│   │   ├── App.tsx        # Componente principal
│   │   └── main.tsx       # Punto de entrada del renderer
│   └── shared/            # Código compartido (opcional)
├── build/                 # Recursos de build (iconos, etc.)
├── dist/                  # Archivos compilados
├── release/               # Instaladores generados
└── scripts/               # Scripts de build
```

## 🔧 Configuración

### TypeScript

- `tsconfig.json` - Configuración general
- `tsconfig.electron.json` - Configuración para el proceso principal

### Electron Builder

La configuración de electron-builder está en `package.json` bajo la sección `build`. Puedes personalizar:
- Iconos de la aplicación
- Configuración del instalador
- Metadatos de la aplicación

## 🎨 Desarrollo de la UI

La aplicación usa:
- **React 19** para la UI
- **Tailwind CSS** para estilos
- **Framer Motion** para animaciones
- **Zustand** para manejo de estado
- **React Query** para gestión de datos

## 🔐 Seguridad

- Context Isolation habilitado
- Node Integration deshabilitado en el renderer
- Preload script para comunicación segura entre procesos

## 📝 Scripts Disponibles

- `npm run dev` - Desarrollo con hot reload
- `npm run build` - Construir la aplicación
- `npm run build:win` - Construir para Windows
- `npm run build:mac` - Construir para macOS
- `npm run build:all` - Construir para ambas plataformas
- `npm run start` - Ejecutar la aplicación compilada
- `npm run lint` - Ejecutar linter
- `npm run typecheck` - Verificar tipos TypeScript

## 🐛 Solución de Problemas

### Error al compilar en Windows

Si encuentras errores relacionados con módulos nativos:
1. Instala Visual Studio Build Tools
2. Ejecuta: `npm install --global windows-build-tools`

### Error al compilar en macOS

Si encuentras errores relacionados con módulos nativos:
1. Instala Xcode Command Line Tools: `xcode-select --install`

### La aplicación no se abre

1. Verifica que todas las dependencias estén instaladas: `npm install`
2. Verifica que el build se completó correctamente: `npm run build`
3. Revisa los logs en la consola de desarrollo

## 📄 Licencia

MIT

## 👥 Contribución

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.


