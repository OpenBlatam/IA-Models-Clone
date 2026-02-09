# Quick Start Guide - Manuales Hogar AI Mobile

Guía rápida para comenzar con la aplicación móvil.

## 🚀 Inicio Rápido

### 1. Instalación

```bash
cd mobile
npm install
```

### 2. Configuración

Crea un archivo `.env` basado en `.env.example`:

```bash
cp .env.example .env
```

Edita `.env` y configura la URL de tu API:

```env
EXPO_PUBLIC_API_URL=http://localhost:8000
```

### 3. Iniciar Desarrollo

```bash
# Iniciar servidor de desarrollo
npm start

# O ejecutar directamente en iOS/Android
npm run ios
npm run android
```

## 📱 Estructura de Navegación

La app tiene 4 pestañas principales:

1. **Inicio** (`/(tabs)/index`) - Pantalla principal con acciones rápidas
2. **Generar** (`/(tabs)/generate`) - Formulario para generar manuales
3. **Historial** (`/(tabs)/history`) - Lista de manuales generados
4. **Perfil** (`/(tabs)/profile`) - Configuración y estadísticas

## 🔌 Conectar con Backend

Asegúrate de que el backend esté corriendo en la URL configurada. Por defecto, la app espera el backend en `http://localhost:8000`.

Para desarrollo en dispositivo físico, usa la IP de tu máquina:

```env
EXPO_PUBLIC_API_URL=http://192.168.1.X:8000
```

## 🎨 Personalización

### Colores

Edita `src/constants/colors.ts` para cambiar los colores de la app.

### Categorías

Edita `src/constants/categories.ts` para agregar o modificar categorías.

## 🐛 Troubleshooting

### Error: "Network request failed"

- Verifica que el backend esté corriendo
- Verifica la URL en `.env`
- Si usas dispositivo físico, asegúrate de usar la IP correcta

### Error: "Module not found"

```bash
# Limpia cache y reinstala
rm -rf node_modules
npm install
npx expo start -c
```

### Error: "Cannot connect to Metro"

```bash
# Reinicia Metro bundler
npx expo start --clear
```

## 📚 Próximos Pasos

- Revisa el [README.md](./README.md) para más detalles
- Explora los componentes en `src/components/`
- Revisa los servicios API en `src/services/`




