# Guía de Configuración - Addiction Recovery AI Mobile

## 📋 Requisitos Previos

- Node.js 18+ instalado
- npm o yarn
- Expo CLI instalado globalmente: `npm install -g expo-cli`
- Para desarrollo iOS: Xcode (solo macOS)
- Para desarrollo Android: Android Studio

## 🚀 Instalación

1. **Instalar dependencias:**
```bash
cd addiction_recovery_ai_mobile
npm install
```

2. **Configurar variables de entorno:**
   - Copia `.env.example` a `.env` (si existe)
   - O configura en `app.json` en la sección `extra`:
```json
"extra": {
  "apiBaseUrl": "http://localhost:8018",
  "apiPrefix": "/recovery"
}
```

3. **Para desarrollo local con API:**
   - Asegúrate de que la API esté corriendo en `http://localhost:8018`
   - O cambia la URL en la configuración

## 🏃 Ejecutar la App

### Desarrollo
```bash
npm start
```

### Android
```bash
npm run android
```

### iOS
```bash
npm run ios
```

### Web
```bash
npm run web
```

## 📱 Estructura del Proyecto

```
addiction_recovery_ai_mobile/
├── src/
│   ├── components/      # Componentes UI reutilizables
│   ├── screens/         # Pantallas de la aplicación
│   ├── services/        # Servicios API
│   ├── types/           # Tipos TypeScript
│   ├── hooks/          # Custom hooks
│   ├── store/           # Estado global (Zustand)
│   ├── navigation/      # Configuración de navegación
│   ├── config/          # Configuración
│   └── utils/           # Utilidades
├── App.tsx              # Punto de entrada
├── package.json
├── tsconfig.json
└── app.json
```

## 🔧 Características Implementadas

### ✅ Autenticación
- Registro de usuarios
- Inicio de sesión
- Gestión de tokens

### ✅ Evaluación
- Formulario de evaluación de adicciones
- Resultados y recomendaciones

### ✅ Progreso
- Registro de entradas diarias
- Estadísticas y métricas
- Seguimiento de rachas

### ✅ Dashboard
- Vista general del progreso
- Logros y recordatorios
- Nivel de riesgo

## 🔌 Integración con API

La app está completamente integrada con todos los endpoints de la API:

- `/recovery/auth/*` - Autenticación
- `/recovery/assess` - Evaluaciones
- `/recovery/progress/*` - Progreso
- `/recovery/dashboard/*` - Dashboard
- Y muchos más...

## 📝 Notas

- Los tokens se almacenan de forma segura usando `expo-secure-store`
- El estado se gestiona con Zustand y React Query
- La navegación usa React Navigation

## 🐛 Solución de Problemas

### Error de conexión a la API
- Verifica que la API esté corriendo
- Revisa la URL en `app.json` o `.env`
- Asegúrate de que el puerto sea correcto (8018 por defecto)

### Errores de TypeScript
```bash
npm run type-check
```

### Limpiar caché
```bash
expo start -c
```

## 📚 Próximos Pasos

- Agregar más pantallas (Chatbot, Notificaciones, etc.)
- Implementar gráficos de progreso
- Agregar notificaciones push
- Mejorar UI/UX

