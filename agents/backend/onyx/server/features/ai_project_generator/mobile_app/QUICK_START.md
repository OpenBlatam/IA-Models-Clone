# 🚀 Quick Start - AI Project Generator Mobile

## Instalación Rápida

```bash
# 1. Instalar dependencias
npm install

# 2. Configurar API (opcional, por defecto usa localhost:8020)
# Edita app.config.js si necesitas cambiar la URL

# 3. Iniciar la app
npm start
```

## Uso Básico

1. **Ver Dashboard**: Pestaña "Inicio" muestra estadísticas y estado de la cola
2. **Ver Proyectos**: Pestaña "Proyectos" lista todos los proyectos
3. **Generar Proyecto**: Pestaña "Generar" para crear nuevos proyectos
4. **Detalles**: Toca cualquier proyecto para ver detalles y exportar

## Configuración de Red

### Para Android Emulador
- Usa `10.0.2.2` en lugar de `localhost`
- O usa la IP de tu máquina (ej: `192.168.1.100:8020`)

### Para iOS Simulador
- Usa `localhost` o la IP de tu máquina
- Asegúrate de que el servidor permita conexiones externas

### Para Dispositivos Físicos
- Usa la IP de tu máquina en la red local
- Ejemplo: `http://192.168.1.100:8020`

## Comandos Útiles

```bash
# Limpiar cache
npm start -- --reset-cache

# iOS específico
cd ios && pod install && cd .. && npm run ios

# Android específico
npm run android

# Ver logs
npx react-native log-android  # Android
npx react-native log-ios      # iOS
```

## Estructura de Archivos

```
mobile_app/
├── App.tsx                 # Punto de entrada
├── src/
│   ├── components/         # Componentes reutilizables
│   ├── screens/            # Pantallas principales
│   ├── navigation/         # Configuración de navegación
│   ├── services/           # Servicios de API
│   ├── types/              # Tipos TypeScript
│   ├── config/             # Configuración
│   ├── hooks/              # Custom hooks
│   └── utils/              # Utilidades
└── assets/                 # Imágenes y recursos
```

## Próximos Pasos

1. Personaliza los colores en los estilos
2. Agrega más funcionalidades según necesites
3. Configura autenticación si es necesario
4. Prepara para producción con EAS Build

¡Listo para usar! 🎉

