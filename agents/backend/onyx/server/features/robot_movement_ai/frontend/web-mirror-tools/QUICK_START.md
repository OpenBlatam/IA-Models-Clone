# 🚀 Inicio Rápido - Web Mirror Specialist

Guía rápida para empezar a usar las herramientas de mirroring.

## 📦 Instalación

### 1. Instalar dependencias Python

```bash
cd web-mirror-tools
pip install -r config/requirements.txt
```

### 2. (Opcional) Instalar Playwright para contenido JS

```bash
pip install playwright
playwright install chromium
```

### 3. (Opcional) Instalar wget (si no está instalado)

**Windows:**
- Descargar desde: https://eternallybored.org/misc/wget/
- O usar: `choco install wget`

**macOS:**
```bash
brew install wget
```

**Linux:**
```bash
sudo apt-get install wget
```

## ✅ Paso 1: Validación Legal (OBLIGATORIO)

**ANTES de hacer cualquier mirroring, valida que tienes permiso:**

```bash
python scripts/validate_legal.py --url https://www.tesla.com
```

**Interpretación del resultado:**
- ✅ `robots.txt permite crawling` → Puedes proceder (si tienes permiso)
- ⚠️ `robots.txt no permite acceso` → **NO PROCEDER**
- ⚠️ `delay alto` → Procede con rate limiting estricto

## 🧪 Paso 2: Modo Simulación (Dry Run)

**Ver qué se descargaría sin descargar nada:**

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --dry-run \
  --verbose
```

## 📥 Paso 3: Mirroring Real

### Opción A: Python Scraper (Recomendado)

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --output ./output/tesla-mirror \
  --rate-limit 1 \
  --max-depth 5 \
  --respect-robots \
  --convert-links
```

### Opción B: Wget (Rápido)

```bash
bash config/wget_mirror.sh https://www.tesla.com ./output/tesla-wget
```

### Opción C: Playwright (Para JS-heavy)

```bash
python scripts/mirror_playwright.py \
  --url https://www.tesla.com \
  --output ./output/tesla-playwright \
  --wait-time 3 \
  --screenshot
```

## 📊 Ver Resultados

Los archivos se guardan en `output/` y los logs en `logs/`.

Cada ejecución genera un reporte JSON con:
- Total de archivos descargados
- Errores encontrados
- Recursos bloqueados por robots.txt
- Tamaño total descargado

## 🎯 Ejemplos Comunes

### Solo una sección específica

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --include "/models,/energy" \
  --max-depth 3
```

### Con autenticación (cookies)

```bash
python scripts/mirror_site.py \
  --url https://example.com \
  --cookies "session=abc123; auth=xyz789" \
  --rate-limit 2
```

### Con rate limiting muy estricto

```bash
python scripts/mirror_site.py \
  --url https://www.tesla.com \
  --rate-limit 3 \
  --max-workers 1
```

## ⚠️ Recordatorios Importantes

1. **SIEMPRE valida legalmente primero** con `validate_legal.py`
2. **SIEMPRE confirma que tienes permiso** del propietario
3. **USA rate limiting apropiado** para no sobrecargar servidores
4. **REVISA el checklist legal** en `LEGAL_CHECKLIST.md`

## 🆘 Solución de Problemas

### Error: "robots.txt no permite acceso"
→ **NO PROCEDER**. Contacta al propietario del sitio.

### Error: "Timeout" o "Connection refused"
→ El servidor puede estar bloqueando. Reduce rate limit o espera.

### Error: "Playwright no está instalado"
→ Ejecuta: `pip install playwright && playwright install chromium`

### Error: "wget: command not found"
→ Instala wget según tu sistema operativo (ver arriba).

## 📚 Documentación Completa

- `README.md` - Documentación completa
- `LEGAL_CHECKLIST.md` - Checklist legal detallado
- `scripts/` - Scripts con `--help` para ver opciones

## 🎓 Próximos Pasos

1. Lee `LEGAL_CHECKLIST.md` completamente
2. Ejecuta validación legal para tu URL objetivo
3. Haz una simulación (dry-run)
4. Si todo está bien, ejecuta mirroring real
5. Revisa los logs y reportes generados

---

**Recuerda: El uso responsable y legal es tu responsabilidad.**



