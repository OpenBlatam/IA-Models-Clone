# Android Setup Guide

Este proyecto incluye scripts automatizados para facilitar el desarrollo en Android.

## 🚀 Configuración Inicial Rápida

**Primera vez configurando Android? Ejecuta:**

```bash
npm run setup-android
```

Este script te ayudará a:
- ✅ Verificar si Android Studio está instalado
- ✅ Instalar componentes del SDK faltantes (ADB, Emulator, System Images)
- ✅ Crear un Android Virtual Device (AVD) automáticamente
- ✅ Configurar variables de entorno

## Solución al Error "No Android connected device found"

Si ves el error `CommandError: No Android connected device found` al presionar 'a' en la terminal de Expo, sigue estos pasos:

### Opción 1: Usar el script automatizado (Recomendado)

```bash
npm run android
```

Este script:
- Detecta automáticamente el SDK de Android
- Busca dispositivos Android conectados
- Si no hay dispositivos, intenta iniciar un emulador automáticamente
- Espera a que el emulador esté listo antes de continuar

### Opción 2: Preparar Android manualmente antes de presionar 'a'

Si ya tienes Expo corriendo y quieres preparar Android antes de presionar 'a':

```bash
npm run prepare-android
```

Este script:
- Verifica si hay dispositivos Android disponibles
- Lista emuladores disponibles
- Intenta iniciar un emulador si no hay dispositivos
- Te indica cuándo está listo para presionar 'a'

### Opción 3: Iniciar emulador manualmente

1. Abre Android Studio
2. Ve a **Tools > Device Manager**
3. Crea un Android Virtual Device (AVD) si no tienes uno
4. Inicia el emulador desde Android Studio
5. Espera a que el emulador termine de iniciar
6. Presiona 'a' en la terminal de Expo

## Configuración del SDK de Android

El script busca automáticamente el SDK de Android en estas ubicaciones:

- `%ANDROID_HOME%` (si está configurado)
- `%ANDROID_SDK_ROOT%` (si está configurado)
- `%LOCALAPPDATA%\Android\Sdk` (ubicación por defecto en Windows)
- `%USERPROFILE%\AppData\Local\Android\Sdk`

### Configurar variables de entorno (Opcional pero recomendado)

1. Abre las variables de entorno del sistema
2. Crea una nueva variable de usuario:
   - Nombre: `ANDROID_HOME`
   - Valor: `C:\Users\TuUsuario\AppData\Local\Android\Sdk` (ajusta la ruta según tu instalación)
3. Agrega a `PATH`:
   - `%ANDROID_HOME%\platform-tools`
   - `%ANDROID_HOME%\emulator`

## Solución de Problemas

### El script no encuentra el SDK de Android

1. Verifica que Android Studio esté instalado
2. Verifica que el SDK esté instalado (Android Studio > SDK Manager)
3. Configura la variable de entorno `ANDROID_HOME` manualmente

### El emulador no inicia automáticamente

1. Asegúrate de tener al menos un AVD creado en Android Studio
2. Verifica que el emulador esté en la ruta correcta: `%ANDROID_HOME%\emulator\emulator.exe`
3. Intenta iniciar el emulador manualmente desde Android Studio primero

### El dispositivo no se detecta después de iniciar el emulador

1. Espera 30-60 segundos para que el emulador termine de iniciar completamente
2. Verifica que ADB detecte el dispositivo: `adb devices`
3. Si el dispositivo aparece como "offline", reinicia ADB: `adb kill-server && adb start-server`

## Scripts Disponibles

- `npm run setup-android` - **NUEVO**: Configuración automática del entorno Android
  - Verifica e instala componentes necesarios
  - Crea AVDs automáticamente
  - Configura variables de entorno
  
- `npm run android` - Inicia Expo con soporte Android inteligente
  - Detecta dispositivos automáticamente
  - Inicia emuladores si no hay dispositivos
  - Espera hasta que el dispositivo esté listo
  
- `npm run prepare-android` - Prepara el entorno Android sin iniciar Expo
- `npm start` - Inicia Expo normalmente (presiona 'a' después)

## Instalación Automática de Componentes

El script `setup-android` puede instalar automáticamente:

1. **Android SDK Platform-Tools** (ADB)
   - Necesario para comunicarse con dispositivos Android
   
2. **Android Emulator**
   - Herramienta para ejecutar AVDs
   
3. **System Images** (Android 13 - API 33)
   - Imágenes del sistema Android para emuladores
   - Descarga ~1GB (puede tardar varios minutos)
   
4. **Android Virtual Devices (AVDs)**
   - Puede crear un AVD básico automáticamente

## Notas

- El script `setup-android` requiere que Android Studio esté instalado primero
- Si Android Studio no está instalado, el script te dará instrucciones
- El script espera hasta 60 segundos para que un emulador inicie
- Si el emulador tarda más, inícialo manualmente desde Android Studio
- Los scripts funcionan mejor cuando `ANDROID_HOME` está configurado
- La primera instalación de system images puede tardar varios minutos

