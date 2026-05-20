# Inicio Rápido - App Móvil Dermatology AI

## 🚀 Configuración Rápida

### 1. Instalar Dependencias

```bash
cd mobile_app
npm install
```

### 2. Configurar Backend

Edita `src/config/api.js` y configura la URL de tu backend:

```javascript
export const API_BASE_URL = 'http://TU_IP:8006';
```

**Nota importante para dispositivos físicos:**
- No uses `localhost` o `127.0.0.1`
- Usa la IP local de tu computadora (ej: `192.168.1.100`)
- Asegúrate de que el dispositivo móvil y la computadora estén en la misma red WiFi

### 3. Iniciar la App

```bash
npm start
```

Luego:
- Presiona `i` para iOS Simulator
- Presiona `a` para Android Emulator
- Escanea el QR con Expo Go en tu dispositivo físico

## 📱 Uso Básico

1. **Tomar Foto**: Ve a la pestaña "Camera" y toma una foto
2. **Análisis Automático**: La app analizará la imagen automáticamente
3. **Ver Resultados**: Revisa las puntuaciones y condiciones detectadas
4. **Recomendaciones**: Obtén recomendaciones personalizadas de productos
5. **Historial**: Revisa tus análisis anteriores en la pestaña "History"

## 🔧 Solución de Problemas

### Error de conexión
- Verifica que el backend esté corriendo
- Verifica la IP en `src/config/api.js`
- Asegúrate de que CORS esté habilitado en el backend

### Permisos de cámara
- La app solicitará permisos automáticamente
- Si se deniegan, ve a Configuración del dispositivo

### Problemas con Expo
```bash
expo start -c  # Limpia la caché
```

## 📝 Notas

- La primera vez que ejecutes la app, puede tardar en cargar
- Los análisis pueden tardar 5-10 segundos dependiendo de la imagen
- Asegúrate de tener buena iluminación al tomar fotos

