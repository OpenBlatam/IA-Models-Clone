# Faceless Video AI - Mobile App

Aplicación móvil React Native con Expo para generar videos sin rostro usando IA, completamente integrada con el backend de Faceless Video AI.

## 🚀 Características

- ✅ **Autenticación completa**: Login, registro, JWT tokens
- ✅ **Generación de videos**: Crear videos desde scripts con IA
- ✅ **Seguimiento en tiempo real**: Monitoreo del progreso de generación
- ✅ **Templates**: Usar y crear plantillas personalizadas
- ✅ **Batch processing**: Generación masiva de videos
- ✅ **Analytics**: Estadísticas y métricas de uso
- ✅ **Búsqueda**: Buscar videos por diferentes criterios
- ✅ **Descarga de videos**: Descargar y compartir videos generados
- ✅ **Dark mode**: Soporte para modo oscuro
- ✅ **TypeScript**: Tipado estricto en toda la aplicación
- ✅ **React Query**: Caché y sincronización de datos
- ✅ **Zustand**: State management ligero
- ✅ **Validación con Zod**: Validación de formularios

## 📋 Requisitos

- Node.js 18+
- npm o yarn
- Expo CLI (`npm install -g expo-cli`)
- iOS Simulator (para desarrollo iOS) o Android Emulator (para desarrollo Android)

## 🛠️ Instalación

1. **Instalar dependencias**:
```bash
cd mobile-app
npm install
```

2. **Configurar variables de entorno**:
Crea un archivo `.env` en la raíz de `mobile-app`:
```env
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
```

3. **Iniciar el servidor de desarrollo**:
```bash
npm start
```

4. **Ejecutar en dispositivo/emulador**:
- iOS: Presiona `i` en la terminal o escanea el QR con la app Expo Go
- Android: Presiona `a` en la terminal o escanea el QR con la app Expo Go

## 📁 Estructura del Proyecto

```
mobile-app/
├── app/                    # Pantallas con Expo Router
│   ├── (auth)/             # Pantallas de autenticación
│   ├── (tabs)/             # Pantallas principales con tabs
│   └── _layout.tsx         # Layout raíz
├── components/             # Componentes reutilizables
│   └── ui/                 # Componentes UI base
├── hooks/                  # Custom hooks
├── services/               # Servicios API
├── store/                  # Zustand stores
├── types/                  # Tipos TypeScript
├── utils/                  # Utilidades
└── assets/                 # Imágenes y recursos
```

## 🔧 Configuración

### API Base URL

La URL base de la API se configura en `utils/config.ts` o mediante la variable de entorno `EXPO_PUBLIC_API_BASE_URL`.

### Autenticación

La app usa JWT tokens almacenados de forma segura con `expo-secure-store`. Los tokens se refrescan automáticamente cuando es necesario.

## 📱 Navegación

La app usa Expo Router para navegación basada en archivos:

- `/(auth)/login` - Pantalla de login
- `/(auth)/register` - Pantalla de registro
- `/(tabs)/` - Pantallas principales con tabs
- `/video-generation` - Generar nuevo video
- `/video-detail` - Detalles de un video
- `/analytics` - Analytics y estadísticas
- `/search` - Búsqueda de videos

## 🎨 Componentes UI

### Button
```tsx
<Button
  title="Click me"
  onPress={() => {}}
  variant="primary"
  size="large"
/>
```

### Input
```tsx
<Input
  label="Email"
  value={email}
  onChangeText={setEmail}
  placeholder="Enter email"
  error={errors.email}
/>
```

### Loading
```tsx
<Loading message="Loading..." />
```

## 🔌 Servicios API

Todos los servicios API están en `services/`:

- `auth-service.ts` - Autenticación
- `video-service.ts` - Operaciones de video
- `template-service.ts` - Templates
- `analytics-service.ts` - Analytics
- `search-service.ts` - Búsqueda
- `music-service.ts` - Biblioteca de música

## 🎣 Hooks Personalizados

### useGenerateVideo
```tsx
const generateVideo = useGenerateVideo();
await generateVideo.mutateAsync(request);
```

### useVideoStatus
```tsx
const { data: video } = useVideoStatus(videoId, enabled);
```

### useTemplates
```tsx
const { data: templates } = useTemplates();
```

## 📦 State Management

### Auth Store
```tsx
const { user, isAuthenticated, login, logout } = useAuthStore();
```

### App Store
```tsx
const { theme, setTheme, language, setLanguage } = useAppStore();
```

## 🧪 Testing

```bash
# Ejecutar tests
npm test

# Ejecutar tests en modo watch
npm test -- --watch
```

## 🚢 Build y Deploy

### Desarrollo
```bash
npm start
```

### Build para producción
```bash
# iOS
eas build --platform ios

# Android
eas build --platform android
```

### Publicar
```bash
eas submit --platform ios
eas submit --platform android
```

## 🔒 Seguridad

- Tokens JWT almacenados de forma segura con `expo-secure-store`
- Validación de inputs con Zod
- Sanitización de datos antes de enviar a la API
- HTTPS obligatorio en producción

## 🌐 Internacionalización

La app soporta múltiples idiomas usando `expo-localization`. Los textos se pueden internacionalizar fácilmente.

## 📊 Analytics

La app incluye integración con analytics para rastrear:
- Generación de videos
- Uso de templates
- Errores y crashes
- Rendimiento

## 🐛 Debugging

### React Native Debugger
```bash
npm install -g react-native-debugger
```

### Flipper
La app es compatible con Flipper para debugging avanzado.

## 📝 Notas

- La app requiere que el backend esté corriendo en la URL configurada
- Los videos se descargan y almacenan localmente
- El progreso de generación se actualiza automáticamente cada 2 segundos
- Los errores se capturan y reportan automáticamente

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Propietaria - Blatam Academy

## 📞 Soporte

Para soporte, contacta al equipo de Blatam Academy.

---

**Versión**: 1.0.0  
**Autor**: Blatam Academy


