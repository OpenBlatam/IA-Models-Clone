# Manuales Hogar AI - Mobile App

Aplicación móvil React Native con Expo para el sistema de generación de manuales paso a paso tipo LEGO para oficios del hogar.

## 🚀 Características

- ✅ **Nueva Arquitectura de Expo** - Utilizando la nueva arquitectura de React Native
- ✅ **TypeScript Estricto** - Type safety completo
- ✅ **Expo Router** - Navegación basada en archivos
- ✅ **React Query** - Data fetching y caching
- ✅ **Dark Mode** - Soporte automático para tema claro/oscuro
- ✅ **Safe Areas** - Manejo correcto de notches y barras de estado
- ✅ **Imágenes Optimizadas** - Uso de expo-image para mejor rendimiento
- ✅ **Validación con Zod** - Validación de formularios
- ✅ **Error Boundaries** - Manejo global de errores
- ✅ **Accesibilidad** - Soporte completo de a11y

## 📋 Requisitos

- Node.js 18+
- Expo CLI
- iOS Simulator (para desarrollo iOS) o Android Emulator (para desarrollo Android)

## 🛠️ Instalación

```bash
# Instalar dependencias
npm install

# O con yarn
yarn install
```

## 🏃 Desarrollo

```bash
# Iniciar el servidor de desarrollo
npm start

# Ejecutar en iOS
npm run ios

# Ejecutar en Android
npm run android

# Ejecutar en Web
npm run web
```

## 📱 Estructura del Proyecto

```
mobile/
├── app/                    # Rutas de Expo Router
│   ├── (tabs)/            # Navegación por tabs
│   │   ├── index.tsx      # Pantalla principal
│   │   ├── generate.tsx  # Generar manual
│   │   ├── history.tsx   # Historial
│   │   └── profile.tsx   # Perfil
│   ├── manual/[id].tsx   # Detalle de manual
│   └── _layout.tsx       # Layout raíz
├── src/
│   ├── components/       # Componentes reutilizables
│   │   ├── home/         # Componentes de inicio
│   │   ├── generate/     # Componentes de generación
│   │   ├── history/      # Componentes de historial
│   │   ├── manual/       # Componentes de manual
│   │   ├── navigation/  # Componentes de navegación
│   │   └── ui/          # Componentes UI base
│   ├── constants/        # Constantes
│   ├── lib/             # Utilidades y contexto
│   ├── services/        # Servicios API
│   └── types/           # Tipos TypeScript
├── assets/              # Imágenes y recursos
└── package.json
```

## 🔌 Configuración de API

La URL base de la API se configura mediante variables de entorno:

```bash
# .env
EXPO_PUBLIC_API_URL=http://localhost:8000
```

O en `app.json`:

```json
{
  "expo": {
    "extra": {
      "apiUrl": "https://api.tudominio.com"
    }
  }
}
```

## 🎨 Temas y Colores

La app soporta temas claro y oscuro automáticamente basado en la configuración del sistema. Los colores están centralizados en `src/constants/colors.ts`.

## 📦 Dependencias Principales

- **expo-router**: Navegación basada en archivos
- **@tanstack/react-query**: Data fetching y caching
- **react-native-safe-area-context**: Manejo de safe areas
- **expo-image**: Optimización de imágenes
- **expo-image-picker**: Selección de imágenes
- **zod**: Validación de esquemas
- **zustand**: Estado global (opcional)

## 🧪 Testing

```bash
# Ejecutar tests
npm test

# Tests en modo watch
npm run test:watch
```

## 📱 Build y Deploy

```bash
# Build para producción
eas build --platform ios
eas build --platform android

# Publicar actualización OTA
eas update --branch production
```

## 🔒 Seguridad

- Almacenamiento seguro con `expo-secure-store`
- Validación de entrada con Zod
- Sanitización de datos
- HTTPS para todas las comunicaciones API

## 🌐 Internacionalización

La app está preparada para i18n usando `expo-localization`. Actualmente soporta español.

## 📄 Licencia

Propietaria - Blatam Academy




