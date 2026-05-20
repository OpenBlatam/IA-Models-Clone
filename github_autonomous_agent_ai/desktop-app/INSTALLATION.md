# Guía de Instalación - Aplicación Desktop

## 📥 Descarga e Instalación

### Windows

1. **Descargar el instalador**
   - Descarga `GitHub Autonomous Agent AI-{version}-x64.exe` desde la carpeta `release/`

2. **Ejecutar el instalador**
   - Doble clic en el archivo `.exe`
   - Sigue las instrucciones del asistente de instalación
   - Elige la ubicación de instalación (opcional)

3. **Iniciar la aplicación**
   - Busca "GitHub Autonomous Agent AI" en el menú de inicio
   - O ejecuta desde el acceso directo del escritorio

### macOS

1. **Descargar el instalador**
   - Descarga `GitHub Autonomous Agent AI-{version}-x64.dmg` desde la carpeta `release/`

2. **Montar la imagen DMG**
   - Doble clic en el archivo `.dmg`
   - Arrastra la aplicación a la carpeta "Applications"

3. **Primera ejecución**
   - macOS puede mostrar una advertencia de seguridad
   - Ve a "Preferencias del Sistema" > "Seguridad y Privacidad"
   - Haz clic en "Abrir de todos modos"

4. **Iniciar la aplicación**
   - Abre "Applications" y busca "GitHub Autonomous Agent AI"
   - O usa Spotlight (Cmd + Space) y escribe el nombre

## 🔧 Requisitos del Sistema

### Windows
- Windows 10 o superior (64-bit)
- 4 GB RAM mínimo
- 200 MB de espacio en disco

### macOS
- macOS 10.15 (Catalina) o superior
- 4 GB RAM mínimo
- 200 MB de espacio en disco
- Procesador Intel o Apple Silicon (M1/M2)

## 🚀 Primera Configuración

1. **Conectar con el Backend**
   - La aplicación intentará conectarse automáticamente a `http://localhost:8030`
   - Si el backend está en otra ubicación, configura la URL en la aplicación

2. **Autenticación de GitHub**
   - Ve a la sección de configuración
   - Configura tu token de GitHub OAuth
   - Sigue las instrucciones en la aplicación

## 🔄 Actualizaciones

La aplicación incluye auto-actualización:
- Se verifican actualizaciones automáticamente
- Se te notificará cuando haya una nueva versión disponible
- Puedes actualizar desde la aplicación

## 🐛 Solución de Problemas

### La aplicación no se inicia

1. **Verifica los requisitos del sistema**
2. **Reinstala la aplicación**
3. **Revisa los logs** (si están disponibles)

### Error de conexión al backend

1. **Verifica que el backend esté ejecutándose**
2. **Verifica la URL del backend en la configuración**
3. **Verifica el firewall** (puede estar bloqueando la conexión)

### Problemas en macOS

Si macOS no permite ejecutar la aplicación:
1. Abre "Preferencias del Sistema"
2. Ve a "Seguridad y Privacidad"
3. Haz clic en "Abrir de todos modos" junto al mensaje de la aplicación

## 📞 Soporte

Para más ayuda, consulta:
- [README.md](./README.md) - Documentación completa
- [QUICK_START.md](./QUICK_START.md) - Guía rápida de desarrollo


