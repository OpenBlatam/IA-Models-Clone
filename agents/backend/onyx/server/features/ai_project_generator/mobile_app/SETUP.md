# Guía de Configuración - AI Project Generator Mobile App

## 📋 Requisitos Previos

### Para iOS
- macOS (para desarrollo nativo)
- Xcode 14 o superior
- CocoaPods instalado: `sudo gem install cocoapods`
- Node.js 18+ y npm

### Para Android
- Android Studio instalado
- Android SDK configurado
- Node.js 18+ y npm
- Java JDK 11 o superior

### Para ambos
- Expo CLI: `npm install -g expo-cli`
- Cuenta de Expo (opcional, para Expo Go)

## 🚀 Instalación Inicial

1. **Instalar dependencias:**
```bash
cd mobile_app
npm install
```

2. **Configurar la URL de la API:**
   
   Edita `app.config.js` o crea un archivo `.env`:
```bash
cp .env.example .env
# Edita .env y configura API_URL
```

3. **Para iOS (solo en macOS):**
```bash
cd ios
pod install
cd ..
```

## 🏃 Ejecutar la App

### Desarrollo con Expo Go

```bash
npm start
```

Luego escanea el QR code con:
- **iOS**: Cámara nativa o Expo Go app
- **Android**: Expo Go app

### Desarrollo Nativo

#### iOS
```bash
npm run ios
```

#### Android
```bash
npm run android
```

Asegúrate de tener un emulador o dispositivo conectado.

## 🔧 Configuración Adicional

### Cambiar URL de la API

1. Edita `app.config.js`:
```javascript
extra: {
  apiUrl: 'http://tu-servidor:8020',
}
```

2. O usa variables de entorno:
```bash
export API_URL=http://tu-servidor:8020
npm start
```

### Configurar para Producción

1. Actualiza `app.json` con información de producción
2. Configura los bundle identifiers únicos
3. Genera los certificados de firma (iOS) y keystore (Android)

## 📱 Build para Producción

### iOS
```bash
eas build --platform ios
```

### Android
```bash
eas build --platform android
```

Nota: Requiere cuenta de Expo y EAS CLI instalado.

## 🐛 Solución de Problemas

### Error: "Unable to resolve module"
```bash
rm -rf node_modules
npm install
```

### Error en iOS: "Pod install failed"
```bash
cd ios
pod deintegrate
pod install
cd ..
```

### Error de conexión a la API
- Verifica que el servidor esté corriendo
- Verifica la URL en `app.config.js`
- Para Android, usa `10.0.2.2` en lugar de `localhost`
- Para iOS, usa la IP local de tu máquina

### Error: "Metro bundler failed"
```bash
npm start -- --reset-cache
```

## 📚 Recursos

- [Documentación de Expo](https://docs.expo.dev/)
- [React Navigation](https://reactnavigation.org/)
- [React Native Docs](https://reactnative.dev/)

