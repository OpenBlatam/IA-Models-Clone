# 🪞 Web Mirror Specialist - Herramienta de Mirroring Web

**Devin Web Mirror Specialist v1.0** - Solución completa para hacer mirroring de sitios web de forma legal y ética.

## ⚠️ ADVERTENCIAS LEGALES IMPORTANTES

**ANTES DE USAR ESTA HERRAMIENTA:**

1. ✅ **DEBES tener permiso explícito del propietario del sitio web**
2. ✅ **DEBES revisar y respetar robots.txt**
3. ✅ **DEBES revisar los términos de servicio del sitio**
4. ✅ **NO uses para redistribuir contenido con copyright sin permiso**
5. ✅ **NO recolectes datos personales sin consentimiento**

## 📋 Checklist Pre-Ejecución

Antes de ejecutar cualquier script, confirma:

- [ ] Tengo permiso del propietario del sitio web
- [ ] He revisado robots.txt y no hay restricciones
- [ ] He revisado los términos de servicio
- [ ] El uso es legítimo (archivado, pruebas offline, migración)
- [ ] No voy a redistribuir públicamente sin permiso

## 🚀 Inicio Rápido

### 1. Validación Legal (OBLIGATORIO)

```bash
# Verificar robots.txt y términos de servicio
python scripts/validate_legal.py --url https://www.tesla.com
```

### 2. Modo Simulación (Dry Run)

```bash
# Ver qué se descargaría sin descargar nada
python scripts/mirror_site.py --url https://www.tesla.com --dry-run
```

### 3. Mirroring Real

```bash
# Ejecutar mirroring completo
python scripts/mirror_site.py --url https://www.tesla.com --output ./output/tesla-mirror
```

## 🛠️ Herramientas Disponibles

### Método 1: Python Scraper (Recomendado para control fino)

- ✅ Control total sobre qué descargar
- ✅ Validación de robots.txt integrada
- ✅ Rate limiting configurable
- ✅ Logging detallado
- ✅ Soporte para autenticación

**Uso:**
```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --output ./output/tesla-mirror \
  --rate-limit 1 \
  --max-workers 2 \
  --respect-robots
```

### Método 2: Wget (Rápido y robusto)

- ✅ Muy rápido para sitios grandes
- ✅ Respeta robots.txt por defecto
- ✅ Convierte enlaces para uso offline
- ✅ Descarga recursos (CSS, JS, imágenes)

**Uso:**
```bash
# Ver comandos en config/wget_mirror.sh
bash config/wget_mirror.sh https://www.tesla.com
```

### Método 3: HTTrack (GUI y potente)

- ✅ Interfaz gráfica disponible
- ✅ Reintentos automáticos
- ✅ Filtros avanzados
- ✅ Bueno para sitios muy grandes

**Uso:**
```bash
# Ver comandos en config/httrack_mirror.sh
bash config/httrack_mirror.sh https://www.tesla.com
```

### Método 4: Playwright (Para contenido JS-heavy)

- ✅ Ejecuta JavaScript
- ✅ Captura contenido renderizado
- ✅ Soporte para SPAs (Single Page Applications)
- ✅ Más lento pero más completo

**Uso:**
```bash
python scripts/mirror_playwright.py \
  --url https://www.tesla.com \
  --output ./output/tesla-mirror
```

## 📁 Estructura de Directorios

```
web-mirror-tools/
├── README.md                 # Esta documentación
├── scripts/
│   ├── validate_legal.py     # Validación de robots.txt y TOS
│   ├── mirror_site.py        # Scraper Python principal
│   ├── mirror_playwright.py  # Scraper con Playwright (JS)
│   └── post_process.py       # Post-procesamiento de archivos
├── config/
│   ├── wget_mirror.sh        # Scripts de configuración wget
│   ├── httrack_mirror.sh     # Scripts de configuración httrack
│   └── requirements.txt      # Dependencias Python
├── output/                   # Sitios descargados (gitignored)
└── logs/                     # Logs de ejecución (gitignored)
```

## ⚙️ Configuración Avanzada

### Parámetros del Scraper Python

```bash
python scripts/mirror_site.py \
  --url URL_OBJETIVO \
  --output DIRECTORIO_SALIDA \
  --scope site-only          # site-only | site+subdomains | custom
  --include "ruta1,ruta2"    # Rutas específicas a incluir
  --exclude "ruta1,ruta2"    # Rutas a excluir
  --rate-limit 1             # Requests por segundo
  --max-workers 2            # Conexiones concurrentes
  --max-depth 5              # Profundidad máxima de crawling
  --respect-robots           # Respetar robots.txt
  --convert-links            # Convertir enlaces para offline
  --user-agent "Custom/1.0"  # User agent personalizado
  --cookies "cookie1=val1"   # Cookies si se requiere auth
  --dry-run                  # Solo simular, no descargar
  --verbose                  # Logging detallado
```

### Autenticación

Si el sitio requiere autenticación:

```bash
# Método 1: Cookies
python scripts/mirror_site.py \
  --url https://example.com \
  --cookies "session=abc123; auth=xyz789"

# Método 2: Basic Auth
python scripts/mirror_site.py \
  --url https://example.com \
  --auth-user usuario \
  --auth-pass contraseña
```

## 📊 Logs y Reportes

Todos los scripts generan logs detallados en `logs/`:

- `mirror_YYYYMMDD_HHMMSS.log` - Log completo de ejecución
- `errors_YYYYMMDD_HHMMSS.log` - Solo errores
- `summary_YYYYMMDD_HHMMSS.json` - Resumen en JSON

El resumen incluye:
- Total de páginas descargadas
- Recursos bloqueados por robots.txt
- Errores HTTP (403, 404, etc.)
- Tiempo total de ejecución
- Tamaño total descargado

## 🔒 Seguridad y Privacidad

- ✅ No se almacenan contraseñas en archivos
- ✅ Los logs no incluyen datos sensibles
- ✅ Se respetan headers de no-index
- ✅ Se excluyen rutas comunes de datos personales (/api/user, etc.)

## 🐳 Docker (Opcional)

```bash
# Construir imagen
docker build -t web-mirror .

# Ejecutar mirroring
docker run -v $(pwd)/output:/app/output web-mirror \
  --url https://www.tesla.com
```

## 📝 Ejemplos de Uso

### Ejemplo 1: Mirroring básico de tesla.com

```bash
# 1. Validar legalmente
python scripts/validate_legal.py --url https://www.tesla.com

# 2. Simular
python scripts/mirror_site.py --url https://www.tesla.com --dry-run

# 3. Ejecutar (si tienes permiso)
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --output ./output/tesla-mirror \
  --rate-limit 1 \
  --respect-robots \
  --convert-links
```

### Ejemplo 2: Solo sección específica

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/models,/energy" \
  --max-depth 3
```

### Ejemplo 3: Con Playwright (JS-heavy)

```bash
python scripts/mirror_playwright.py \
  --url https://www.tesla.com \
  --output ./output/tesla-mirror \
  --wait-time 3 \
  --screenshot
```

## 🚨 Limitaciones Conocidas

1. **Contenido dinámico**: Sitios con mucho JavaScript pueden requerir Playwright
2. **CAPTCHA**: No se puede eludir CAPTCHA automáticamente
3. **Paywalls**: No se puede acceder a contenido detrás de paywalls sin credenciales
4. **Rate limiting**: Servidores pueden bloquear si se hace demasiado rápido
5. **Tamaño**: Sitios muy grandes pueden tomar mucho tiempo y espacio

## 📞 Soporte

Si encuentras problemas:
1. Revisa los logs en `logs/`
2. Verifica que tienes permiso del propietario
3. Confirma que robots.txt permite crawling
4. Revisa que no hay medidas anti-bot activas

## ⚖️ Disclaimer Legal

Esta herramienta es para uso legítimo únicamente. El usuario es responsable de:
- Obtener permisos necesarios
- Respetar robots.txt y términos de servicio
- Cumplir con leyes de copyright y privacidad
- No usar para actividades ilegales

**El autor de esta herramienta no se hace responsable del uso indebido.**



