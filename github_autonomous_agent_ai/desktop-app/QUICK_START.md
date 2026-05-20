# Quick Start Guide - Desktop App

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias

```bash
cd desktop-app
npm install
```

### 2. Desarrollo

Ejecuta la aplicación en modo desarrollo:

```bash
npm run dev
```

La aplicación se abrirá automáticamente y se recargará cuando hagas cambios.

### 3. Construcción para Producción

#### Windows

```bash
# Opción 1: Script automático
scripts\build-windows.bat

# Opción 2: Manual
npm run build:win
```

El instalador estará en: `release/GitHub Autonomous Agent AI-{version}-x64.exe`

#### macOS

```bash
# Opción 1: Script automático
chmod +x scripts/build-mac.sh
./scripts/build-mac.sh

# Opción 2: Manual
npm run build:mac
```

El instalador estará en: `release/GitHub Autonomous Agent AI-{version}-x64.dmg`

## 📝 Notas Importantes

1. **Primera vez**: La primera compilación puede tardar varios minutos mientras se descargan las dependencias de Electron.

2. **Iconos**: Asegúrate de tener iconos en la carpeta `build/`:
   - `build/icon.ico` para Windows
   - `build/icon.icns` para macOS

3. **Firmado de código**: Para distribuir la aplicación, considera firmar el código:
   - Windows: Certificado de código
   - macOS: Certificado de desarrollador de Apple

## 🔧 Solución Rápida de Problemas

**Error: "Cannot find module"**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Error al compilar módulos nativos (Windows)**
```bash
npm install --global windows-build-tools
```

**Error al compilar módulos nativos (macOS)**
```bash
xcode-select --install
```

## 📚 Más Información

Consulta el [README.md](./README.md) para información detallada.


