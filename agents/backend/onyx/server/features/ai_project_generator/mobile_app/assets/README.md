# Assets - Imágenes y Recursos

## Archivos Requeridos

Para que la app funcione correctamente, necesitas crear los siguientes archivos en esta carpeta:

### Iconos
- `icon.png` - Icono de la app (1024x1024px)
- `adaptive-icon.png` - Icono adaptativo para Android (1024x1024px)
- `favicon.png` - Favicon para web (48x48px)

### Splash Screen
- `splash.png` - Imagen de splash screen (1242x2436px recomendado)

## Generar Assets

### Opción 1: Usar Expo Asset Generator
```bash
npx expo-asset-generator
```

### Opción 2: Crear Manualmente

1. **icon.png**: 
   - Tamaño: 1024x1024px
   - Formato: PNG
   - Sin transparencia
   - Fondo sólido

2. **adaptive-icon.png**:
   - Tamaño: 1024x1024px
   - Formato: PNG
   - El área central (512x512px) será visible
   - Mantén contenido importante en el centro

3. **splash.png**:
   - Tamaño: 1242x2436px (o proporcional)
   - Formato: PNG
   - Fondo: #ffffff (blanco)
   - Logo centrado

4. **favicon.png**:
   - Tamaño: 48x48px
   - Formato: PNG

## Placeholder Temporal

Si no tienes los assets aún, puedes usar imágenes temporales:

```bash
# Crear iconos temporales simples
# Usa cualquier herramienta de diseño o generador online
```

## Herramientas Recomendadas

- [Expo Asset Generator](https://github.com/expo/expo-asset-generator)
- [App Icon Generator](https://www.appicon.co/)
- [Figma](https://www.figma.com/) para diseño
- [Canva](https://www.canva.com/) para diseño rápido

## Nota

La app funcionará sin estos assets, pero mostrará errores en la consola. Es recomendable agregarlos antes de hacer build para producción.

