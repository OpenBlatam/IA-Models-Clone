# 🎨 Frontend de Validación - Dermatology AI

Frontend simple y rápido para validar la idea del proyecto Dermatology AI.

## 🚀 Inicio Rápido

### Opción 1: Abrir Directamente (Más Simple)

1. Asegúrate que el backend esté corriendo:
```bash
cd agents/backend/onyx/server/features/dermatology_ai
python main.py
```

2. Abre `index.html` en tu navegador:
   - Doble clic en `index.html`
   - O arrastra el archivo a tu navegador

**Nota**: Algunos navegadores pueden bloquear las peticiones CORS. Si esto pasa, usa la Opción 2.

### Opción 2: Servir con un Servidor Local (Recomendado)

#### Con Python:
```bash
cd validation/frontend
python -m http.server 8080
```

#### Con Node.js:
```bash
cd validation/frontend
npx serve .
```

#### Con PHP:
```bash
cd validation/frontend
php -S localhost:8080
```

Luego abre: `http://localhost:8080`

## 📋 Requisitos

- Backend corriendo en `http://localhost:8006`
- Navegador moderno (Chrome, Firefox, Safari, Edge)
- Conexión a internet (solo para cargar fuentes, si es necesario)

## 🎯 Características

- ✅ Subida de imágenes (click o drag & drop)
- ✅ Preview de imagen antes de analizar
- ✅ Análisis de piel con métricas detalladas
- ✅ Visualización de resultados
- ✅ Recomendaciones básicas
- ✅ Diseño responsive (funciona en móvil)

## 🔧 Configuración

### Cambiar URL del Backend

Si tu backend está en otro puerto o URL, edita `app.js`:

```javascript
const API_BASE_URL = 'http://localhost:8006'; // Cambia esto
```

### Cambiar Link de Feedback

Edita `index.html` y busca:

```html
<button class="btn btn-secondary" onclick="window.open('https://forms.gle/YOUR_FORM_ID', '_blank')">
```

Reemplaza `YOUR_FORM_ID` con el ID de tu formulario de Google Forms.

## 🐛 Solución de Problemas

### Error: "No se puede conectar al backend"

1. Verifica que el backend esté corriendo:
```bash
curl http://localhost:8006/health
```

2. Verifica que la URL en `app.js` sea correcta

3. Si usas `file://` para abrir el HTML, usa un servidor local (Opción 2)

### Error: CORS (Cross-Origin Resource Sharing)

Si ves errores de CORS en la consola:
1. Asegúrate de usar un servidor local (no `file://`)
2. Verifica que el backend permita CORS (debería estar configurado)

### El análisis tarda mucho

- Normal: puede tardar 5-15 segundos
- Si tarda más de 30 segundos, revisa los logs del backend
- Prueba con una imagen más pequeña (< 5MB)

### Los resultados no se muestran

1. Abre la consola del navegador (F12)
2. Revisa si hay errores
3. Verifica que el backend responda correctamente:
```bash
curl -X POST http://localhost:8006/dermatology/analyze-image \
  -F "file=@ruta/a/imagen.jpg" \
  -F "enhance=true"
```

## 📱 Uso en Móvil

El frontend es responsive y funciona en móviles. Para probarlo:

1. Encuentra la IP de tu computadora:
   - Windows: `ipconfig`
   - Mac/Linux: `ifconfig`

2. En el móvil, abre: `http://TU_IP:8080`

3. Asegúrate que el móvil esté en la misma red WiFi

## 🎨 Personalización

### Cambiar Colores

Edita `styles.css` y busca las variables de color:

```css
/* Cambia estos colores */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Cambiar Textos

Edita `index.html` directamente. Todos los textos están en español y son fáciles de encontrar.

## 📊 Integración con Analytics

Para medir el uso, puedes agregar Google Analytics:

1. Agrega esto antes de `</head>` en `index.html`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

2. Reemplaza `GA_MEASUREMENT_ID` con tu ID de Google Analytics

## 🔗 Próximos Pasos

1. **Usa el frontend** para validar con usuarios reales
2. **Recopila feedback** usando el botón de feedback
3. **Mide métricas** básicas (cuántos usan, completan, etc.)
4. **Lee la estrategia**: [../STRATEGY.md](../STRATEGY.md)

---

**¿Listo para validar?** → Abre `index.html` y comienza! 🚀






