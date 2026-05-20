# AI Project Generator - Mobile App

Aplicación móvil React Native con Expo para iOS y Android que integra todos los endpoints de la API del AI Project Generator.

## 🚀 Características

- ✅ Generación de proyectos de IA
- ✅ Listado y gestión de proyectos
- ✅ Visualización de estadísticas y métricas
- ✅ Estado de la cola de proyectos en tiempo real
- ✅ Exportación de proyectos (ZIP/TAR)
- ✅ Validación de proyectos
- ✅ Navegación intuitiva con tabs
- ✅ Diseño moderno y responsive

## 📦 Instalación

```bash
# Instalar dependencias
npm install

# O con yarn
yarn install
```

## 🏃 Uso

### Desarrollo

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

### Configuración de la API

Por defecto, la app se conecta a `http://localhost:8020`. Para cambiar la URL de la API:

1. Edita `app.json` y agrega:
```json
{
  "expo": {
    "extra": {
      "apiUrl": "http://tu-api-url:8020"
    }
  }
}
```

2. O configura una variable de entorno en desarrollo.

## 📱 Pantallas

### Home
- Dashboard con estadísticas generales
- Estado de la cola de proyectos
- Accesos rápidos a funciones principales

### Proyectos
- Lista de todos los proyectos
- Filtros por estado y autor
- Navegación a detalles del proyecto

### Generar
- Formulario completo para crear nuevos proyectos
- Opciones configurables (tests, docker, docs, etc.)
- Validación de campos

### Detalle del Proyecto
- Información completa del proyecto
- Exportación (ZIP/TAR)
- Eliminación de proyectos

## 🔧 Estructura del Proyecto

```
mobile_app/
├── src/
│   ├── components/      # Componentes reutilizables
│   ├── config/          # Configuración (API, etc.)
│   ├── navigation/      # Navegación
│   ├── screens/         # Pantallas principales
│   ├── services/        # Servicios de API
│   └── types/           # Tipos TypeScript
├── App.tsx              # Componente principal
├── app.json             # Configuración de Expo
└── package.json         # Dependencias
```

## 📚 API Endpoints Integrados

La app integra todos los endpoints principales:

- ✅ Generación: `/api/v1/generate`, `/api/v1/generate/batch`
- ✅ Proyectos: `/api/v1/projects`, `/api/v1/projects/{id}`
- ✅ Estado: `/api/v1/status`, `/api/v1/stats`, `/api/v1/queue`
- ✅ Exportación: `/api/v1/export/zip`, `/api/v1/export/tar`
- ✅ Validación: `/api/v1/validate`
- ✅ Health: `/health`, `/health/detailed`

## 🛠️ Tecnologías

- **React Native** - Framework móvil
- **Expo** - Herramientas y servicios
- **TypeScript** - Tipado estático
- **React Navigation** - Navegación
- **Axios** - Cliente HTTP
- **React Query** - Gestión de estado del servidor

## 📝 Notas

- La app está lista para iOS y Android
- Requiere que el servidor de la API esté corriendo
- Soporta autenticación con tokens JWT (almacenados en AsyncStorage)
- Manejo de errores completo con mensajes descriptivos

## 🐛 Troubleshooting

### Error de conexión a la API

Asegúrate de que:
1. El servidor de la API esté corriendo en el puerto correcto
2. La URL de la API esté correctamente configurada
3. No haya problemas de firewall o red

### Problemas con iOS

```bash
cd ios
pod install
cd ..
npm run ios
```

### Problemas con Android

Asegúrate de tener Android Studio instalado y un emulador o dispositivo conectado.

