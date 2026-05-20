# 🌐 Interfaz HTML - Character Clothing Changer AI

## 📋 Descripción

Interfaz web completa y lista para usar para el Character Clothing Changer AI. Permite cambiar la ropa de personajes de forma visual e intuitiva.

## 🚀 Uso Rápido

### 1. Iniciar el Servidor

```bash
cd character_clothing_changer_ai
python run_server.py
```

### 2. Abrir la Interfaz

Abre tu navegador y visita:
```
http://localhost:8002
```

La interfaz HTML se cargará automáticamente.

## ✨ Características

- ✅ **Subida de Imágenes**: Arrastra y suelta o selecciona imágenes fácilmente
- ✅ **Preview en Tiempo Real**: Ve tu imagen antes de procesarla
- ✅ **Opciones Avanzadas**: Control total sobre parámetros de generación
- ✅ **Indicador de Estado**: Monitorea el estado del servidor y modelo
- ✅ **Resultados Visuales**: Ve las imágenes generadas directamente
- ✅ **Descarga de Tensors**: Descarga los tensors generados para ComfyUI
- ✅ **Diseño Responsive**: Funciona en desktop y móvil

## 📸 Cómo Usar

### Paso 1: Seleccionar Imagen
1. Haz clic en el área de carga de archivos
2. Selecciona una imagen de personaje (PNG, JPG, JPEG)
3. La imagen se mostrará como preview

### Paso 2: Describir la Ropa
1. Escribe una descripción de la nueva ropa en el campo de texto
   - Ejemplo: "a red elegant dress"
   - Ejemplo: "a blue casual t-shirt"
   - Ejemplo: "a black leather jacket"

### Paso 3: (Opcional) Configurar Opciones
1. Haz clic en "Opciones Avanzadas" para más control:
   - **Prompt Completo**: Personaliza el prompt completo
   - **Prompt Negativo**: Excluye elementos no deseados
   - **Pasos de Inferencia**: Controla la calidad (más pasos = mejor calidad pero más lento)
   - **Guidance Scale**: Controla qué tan cerca sigue el prompt
   - **Fuerza de Inpainting**: Controla qué tanto cambia la imagen
   - **Guardar Tensor**: Decide si guardar el tensor para ComfyUI

### Paso 4: Procesar
1. Haz clic en "🚀 Cambiar Ropa"
2. Espera mientras se procesa (puede tomar varios minutos)
3. El resultado se mostrará automáticamente

### Paso 5: Descargar Resultados
- La imagen generada se muestra directamente
- Si se guardó un tensor, puedes descargarlo con el botón "📥 Descargar Tensor"

## 🎨 Ejemplos de Uso

### Ejemplo 1: Vestido Elegante
- **Imagen**: Personaje con ropa casual
- **Descripción**: "a red elegant dress with white lace details"
- **Resultado**: Personaje con vestido rojo elegante

### Ejemplo 2: Ropa Casual
- **Imagen**: Personaje formal
- **Descripción**: "a blue casual t-shirt and jeans"
- **Resultado**: Personaje con ropa casual

### Ejemplo 3: Estilo Futurista
- **Imagen**: Personaje normal
- **Descripción**: "a black cyberpunk jacket with neon accents"
- **Resultado**: Personaje con estilo futurista

## ⚙️ Configuración Avanzada

### Parámetros Recomendados

**Alta Calidad (Lento)**:
- Pasos: 50-100
- Guidance Scale: 7.5-10
- Strength: 0.8-0.9

**Calidad Media (Balanceado)**:
- Pasos: 30-50
- Guidance Scale: 7.5
- Strength: 0.7-0.8

**Rápido (Baja Calidad)**:
- Pasos: 20-30
- Guidance Scale: 5-7
- Strength: 0.6-0.7

## 🔧 Solución de Problemas

### El servidor no responde
- Verifica que el servidor esté corriendo en `http://localhost:8002`
- Revisa la consola del servidor para errores
- Asegúrate de que el puerto 8002 no esté en uso

### El modelo no está inicializado
- Espera unos minutos - el modelo puede tardar en cargar
- Haz clic en "Inicializar Modelo" si está disponible
- Verifica que tengas suficiente memoria RAM

### Error al procesar imagen
- Verifica que la imagen sea válida (PNG, JPG, JPEG)
- Asegúrate de que la imagen no sea muy grande (máx 10MB recomendado)
- Revisa la descripción de la ropa - debe ser clara y específica

### La imagen no se muestra
- Verifica la consola del navegador (F12) para errores
- Asegúrate de que CORS esté habilitado en el servidor
- Intenta recargar la página

## 📝 Notas

- El procesamiento puede tomar varios minutos dependiendo de:
  - Tamaño de la imagen
  - Número de pasos de inferencia
  - Hardware disponible (CPU vs GPU)
- Los tensors se guardan en `./comfyui_tensors/` por defecto
- Las imágenes temporales se guardan en el directorio temporal del sistema

## 🌐 Acceso Remoto

Si quieres acceder desde otro dispositivo en la misma red:

1. Asegúrate de que el servidor esté configurado para escuchar en `0.0.0.0`
2. Encuentra la IP de tu máquina:
   - Windows: `ipconfig`
   - Linux/Mac: `ifconfig` o `ip addr`
3. Accede desde otro dispositivo usando: `http://TU_IP:8002`

## 📚 Recursos Adicionales

- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/api/v1/health
- **Model Info**: http://localhost:8002/api/v1/model/info

## 🎉 ¡Listo!

Ya tienes todo listo para cambiar la ropa de tus personajes con IA. ¡Disfruta creando!


