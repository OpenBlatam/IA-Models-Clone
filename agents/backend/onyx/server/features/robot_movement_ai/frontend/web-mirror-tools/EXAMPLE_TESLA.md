# 🚗 Ejemplo: Mirroring de tesla.com

**IMPORTANTE**: Este es un ejemplo educativo. **DEBES tener permiso explícito de Tesla antes de ejecutar esto en producción.**

## ⚠️ Advertencias Legales

1. **Tesla.com puede tener restricciones en robots.txt**
2. **Los términos de servicio de Tesla deben revisarse**
3. **El contenido tiene copyright de Tesla**
4. **Solo para uso legítimo (archivado personal, desarrollo, con permiso)**

## 📋 Paso a Paso

### 1. Validación Legal (OBLIGATORIO)

```bash
python scripts/validate_legal.py --url https://www.tesla.com
```

**Interpreta el resultado:**
- Si dice "NO PROCEDER" → **DETENTE AQUÍ**
- Si dice "permite crawling" → Continúa con precaución
- Si hay "delay alto" → Usa rate limiting estricto

### 2. Simulación (Dry Run)

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --dry-run \
  --max-depth 2 \
  --verbose
```

Esto te mostrará qué se descargaría sin descargar nada.

### 3. Mirroring Selectivo (Recomendado)

En lugar de descargar todo el sitio, descarga solo secciones específicas:

```bash
# Solo sección de modelos
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/models" \
  --max-depth 3 \
  --rate-limit 2 \
  --output ./output/tesla-models \
  --respect-robots \
  --convert-links
```

```bash
# Solo sección de energía
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/energy" \
  --max-depth 3 \
  --rate-limit 2 \
  --output ./output/tesla-energy \
  --respect-robots
```

### 4. Mirroring Completo (Solo si tienes permiso)

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --output ./output/tesla-complete \
  --rate-limit 2 \
  --max-depth 5 \
  --respect-robots \
  --convert-links \
  --exclude "/api,/admin,/checkout,/cart"
```

### 5. Con Playwright (Para contenido JS dinámico)

Si el sitio usa mucho JavaScript:

```bash
python scripts/mirror_playwright.py \
  --url https://www.tesla.com \
  --output ./output/tesla-playwright \
  --wait-time 5 \
  --max-depth 3 \
  --max-pages 30 \
  --screenshot
```

## 🎯 Casos de Uso Específicos

### Caso 1: Extraer solo elementos de diseño

Si solo quieres CSS, imágenes y estructura HTML para inspiración:

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/models/model-s" \
  --max-depth 1 \
  --rate-limit 3 \
  --output ./output/tesla-design-elements
```

Luego puedes:
- Extraer CSS de `output/tesla-design-elements/`
- Ver estructura HTML
- Analizar imágenes y assets

### Caso 2: Archivar una página específica

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com/models \
  --max-depth 0 \
  --rate-limit 1 \
  --output ./output/tesla-models-page
```

### Caso 3: Con wget (más rápido, menos control)

```bash
bash config/wget_mirror.sh https://www.tesla.com/models ./output/tesla-wget
```

## 📊 Análisis Post-Descarga

Después de descargar, puedes analizar:

1. **Estructura HTML**: Ver cómo están organizados los componentes
2. **CSS**: Extraer tokens de diseño, colores, tipografía
3. **JavaScript**: Ver qué librerías y frameworks usan
4. **Imágenes**: Ver formatos, tamaños, optimizaciones
5. **Assets**: Ver estructura de archivos estáticos

## 🔍 Extracción de Elementos de Diseño

Si tu objetivo es extraer elementos de diseño de Tesla para tu proyecto:

### 1. Descarga selectiva
```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/models" \
  --max-depth 2 \
  --output ./output/tesla-design
```

### 2. Analiza los archivos descargados:
- `output/tesla-design/www.tesla.com/models/*.html` - Estructura
- `output/tesla-design/www.tesla.com/*.css` - Estilos
- `output/tesla-design/www.tesla.com/*.js` - Comportamiento
- `output/tesla-design/www.tesla.com/images/` - Assets visuales

### 3. Extrae tokens de diseño:
- Colores del CSS
- Tipografía
- Espaciado
- Sombras y bordes
- Animaciones

## ⚙️ Configuración Recomendada para Tesla.com

Basado en que es un sitio grande y probablemente tenga medidas anti-bot:

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --rate-limit 3 \
  --max-workers 1 \
  --max-depth 4 \
  --respect-robots \
  --user-agent "DevinWebMirror/1.0 (Educational Use)" \
  --output ./output/tesla-mirror
```

**Parámetros clave:**
- `rate-limit 3`: Espera 3 segundos entre requests (muy conservador)
- `max-workers 1`: Solo una conexión a la vez
- `max-depth 4`: Limita profundidad para no descargar todo
- `respect-robots`: SIEMPRE activo

## 🚨 Limitaciones Conocidas

1. **Cloudflare/Anti-bot**: Tesla.com puede usar Cloudflare que bloquea bots
2. **JavaScript pesado**: Mucho contenido se genera con JS, necesitas Playwright
3. **Tamaño**: El sitio es muy grande, puede tomar horas
4. **Rate limiting**: El servidor puede bloquear si haces demasiados requests

## 📝 Notas Importantes

- **NO redistribuyas** el contenido descargado públicamente
- **NO uses** para crear productos comerciales sin licencia
- **RESPETA** el copyright de Tesla
- **USA** solo para aprendizaje y desarrollo personal (con permiso)

## 🔗 Recursos Relacionados

Si estás trabajando en el proyecto frontend que ya tiene elementos de Tesla:
- Revisa `TESLA_ANIMATIONS_V31.md`
- Revisa `TESLA_EXACT_DESIGN_TOKENS.md`
- Revisa `TESLA_ULTRA_EXACT_VALUES.md`

Estos archivos ya contienen análisis de diseño de Tesla que puedes usar como referencia.

---

**Recuerda: Este es un ejemplo educativo. Siempre obtén permiso antes de hacer mirroring de sitios web.**



