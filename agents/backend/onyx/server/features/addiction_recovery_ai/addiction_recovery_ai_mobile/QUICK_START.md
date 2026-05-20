# 🚀 Quick Start - Addiction Recovery AI Mobile

## Instalación Rápida

```bash
# 1. Navegar al directorio
cd addiction_recovery_ai_mobile

# 2. Instalar dependencias
npm install

# 3. Configurar API (editar app.json si es necesario)
# La URL por defecto es: http://localhost:8018

# 4. Iniciar la app
npm start
```

## Configuración de la API

La app se conecta automáticamente a la API. Por defecto usa:
- **URL Base**: `http://localhost:8018`
- **Prefijo**: `/recovery`

Para cambiar la configuración, edita `app.json`:

```json
{
  "expo": {
    "extra": {
      "apiBaseUrl": "http://tu-api.com",
      "apiPrefix": "/recovery"
    }
  }
}
```

## Uso Básico

### 1. Registro/Login
- Abre la app
- Si no tienes cuenta, regístrate con un ID de usuario
- Si ya tienes cuenta, inicia sesión

### 2. Dashboard
- Ve la vista general de tu progreso
- Días sobrio, rachas, y nivel de riesgo

### 3. Registrar Progreso
- Ve a la pestaña "Progreso"
- Toca "Registrar Entrada"
- Completa el formulario diario

### 4. Evaluación
- Ve a la pestaña "Evaluación"
- Completa el formulario
- Obtén recomendaciones personalizadas

## Estructura de Carpetas

```
src/
├── components/    # Botones, Inputs, Cards, etc.
├── screens/       # Login, Dashboard, Progress, Assessment
├── services/      # api.ts - Cliente API completo
├── types/         # Tipos TypeScript
├── hooks/         # useApi.ts - Hooks de React Query
├── store/         # Zustand stores (auth, progress)
├── navigation/    # AppNavigator
├── config/        # Configuración API
└── utils/         # Constantes y utilidades
```

## Características Principales

✅ **Autenticación completa**
✅ **Evaluación de adicciones**
✅ **Seguimiento de progreso diario**
✅ **Dashboard con métricas**
✅ **Estadísticas detalladas**
✅ **Integración completa con API**

## Comandos Útiles

```bash
# Desarrollo
npm start

# Android
npm run android

# iOS
npm run ios

# Web
npm run web

# Verificar tipos
npm run type-check

# Linter
npm run lint
```

## Solución de Problemas

### Error de conexión
- Verifica que la API esté corriendo
- Revisa la URL en `app.json`
- Asegúrate de que el puerto sea correcto

### Errores de TypeScript
```bash
npm run type-check
```

### Limpiar caché
```bash
expo start -c
```

## Próximos Pasos

1. ✅ La app está lista para usar
2. Personaliza los colores en `src/utils/constants.ts`
3. Agrega más pantallas según necesites
4. Configura notificaciones push
5. Agrega gráficos de progreso

## Documentación Adicional

- `README.md` - Información general
- `SETUP.md` - Guía de configuración detallada
- `FEATURES.md` - Lista completa de características

