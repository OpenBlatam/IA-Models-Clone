# Dermatology AI - Mobile App

Aplicación móvil React Native para análisis de piel con IA, integrada con el backend de Dermatology AI.

## 🚀 Características

- 📸 **Captura de Fotos y Videos**: Toma fotos o graba videos para análisis
- 🔍 **Escaneo en Tiempo Real**: Análisis continuo de la piel en tiempo real
- 📊 **Análisis Detallado**: Puntuaciones de calidad, textura, hidratación, etc.
- 💡 **Recomendaciones Personalizadas**: Productos y rutinas recomendadas
- 📈 **Historial y Progreso**: Seguimiento de tus análisis a lo largo del tiempo
- 📄 **Reportes Completos**: Exporta reportes en PDF, HTML o JSON

## 📋 Requisitos Previos

- Node.js (v14 o superior)
- npm o yarn
- Expo CLI (`npm install -g expo-cli`)
- Expo Go app en tu dispositivo móvil (iOS/Android)

## 🛠️ Instalación

1. Navega al directorio de la app:
```bash
cd mobile_app
```

2. Instala las dependencias:
```bash
npm install
```

3. Configura la URL del backend en `src/config/api.js`:
```javascript
export const API_BASE_URL = 'http://tu-servidor:8006';
```

Para desarrollo local, puedes usar:
```javascript
export const API_BASE_URL = 'http://localhost:8006'; // iOS Simulator
// o
export const API_BASE_URL = 'http://10.0.2.2:8006'; // Android Emulator
// o
export const API_BASE_URL = 'http://TU_IP_LOCAL:8006'; // Dispositivo físico
```

## 🏃 Ejecución

### Desarrollo

```bash
npm start
```

Esto abrirá Expo DevTools. Luego:
- Presiona `i` para iOS Simulator
- Presiona `a` para Android Emulator
- Escanea el QR code con Expo Go en tu dispositivo físico

### Android

```bash
npm run android
```

### iOS

```bash
npm run ios
```

## ✨ Mejoras Recientes

- ✅ **TypeScript completo**: Todas las pantallas convertidas a TypeScript
- ✅ **Hooks personalizados**: `useCamera` y `useAnalysis` para mejor organización
- ✅ **Componentes mejorados**: LoadingSpinner, ErrorView, y más
- ✅ **UI/UX mejorada**: Animaciones, mejor feedback visual
- ✅ **Búsqueda en historial**: Filtros y búsqueda en tiempo real
- ✅ **Mejor manejo de errores**: Componentes de error reutilizables

Ver [IMPROVEMENTS.md](./IMPROVEMENTS.md) para más detalles.

## 📱 Estructura del Proyecto

```
mobile_app/
├── App.tsx                # Componente principal y navegación (TypeScript)
├── tsconfig.json           # Configuración TypeScript
├── src/
│   ├── hooks/             # Hooks personalizados
│   │   ├── useCamera.ts
│   │   └── useAnalysis.ts
│   ├── screens/           # Pantallas (TypeScript)
│   │   ├── HomeScreen.tsx
│   │   ├── CameraScreen.tsx
│   │   ├── AnalysisScreen.tsx
│   │   ├── HistoryScreen.tsx
│   │   └── ... (más pantallas)
│   ├── components/        # Componentes reutilizables
│   │   ├── ScoreCard.tsx
│   │   ├── RadarChart.tsx
│   │   ├── LoadingSpinner.tsx
│   │   └── ErrorView.tsx
│   ├── services/          # Servicios API
│   │   └── apiService.ts
│   ├── store/            # Redux store
│   │   ├── store.ts
│   │   └── reducers/
│   ├── types/            # Definiciones de tipos
│   │   └── index.ts
│   ├── config/           # Configuración
│   │   └── api.ts
│   └── utils/            # Utilidades
│       ├── helpers.ts
│       └── constants.ts
└── package.json
```

## 🔌 Integración con Backend

La app se conecta automáticamente con el backend de Dermatology AI. Asegúrate de que:

1. El backend esté corriendo en el puerto configurado (por defecto 8006)
2. La URL del backend esté correctamente configurada en `src/config/api.js`
3. El backend tenga CORS habilitado para permitir peticiones desde la app móvil

## 📸 Funcionalidades Principales

### Análisis de Imagen
- Captura fotos desde la cámara
- Selecciona imágenes de la galería
- Análisis automático con puntuaciones detalladas

### Análisis de Video
- Graba videos cortos
- Análisis frame por frame
- Detección de cambios temporales

### Escaneo en Tiempo Real
- Análisis continuo cada 3 segundos
- Visualización de puntuaciones en tiempo real
- Análisis completo al finalizar

### Recomendaciones
- Productos recomendados basados en el análisis
- Rutinas personalizadas
- Consejos y advertencias

### Historial
- Ver todos los análisis previos
- Comparar análisis
- Seguimiento de progreso

## 🎨 Personalización

Puedes personalizar los colores y estilos editando los archivos de estilos en cada componente. Los colores principales están definidos en:

- Color primario: `#6366f1` (índigo)
- Color secundario: `#8b5cf6` (púrpura)
- Color de éxito: `#10b981` (verde)
- Color de advertencia: `#f59e0b` (ámbar)
- Color de error: `#ef4444` (rojo)

## 🐛 Solución de Problemas

### Error de conexión con el backend
- Verifica que el backend esté corriendo
- Asegúrate de usar la IP correcta (no localhost en dispositivos físicos)
- Verifica la configuración de CORS en el backend

### Permisos de cámara
- La app solicitará permisos automáticamente
- Si se deniegan, ve a Configuración del dispositivo y otorga permisos manualmente

### Problemas con Expo
```bash
# Limpia la caché
expo start -c

# Reinstala dependencias
rm -rf node_modules
npm install
```

## 📝 Notas

- Esta app está diseñada para trabajar con el backend de Dermatology AI
- Asegúrate de tener una conexión estable a internet para el análisis
- Los análisis pueden tardar unos segundos dependiendo de la calidad de la imagen

## 📄 Licencia

Este proyecto es parte del sistema Dermatology AI.

