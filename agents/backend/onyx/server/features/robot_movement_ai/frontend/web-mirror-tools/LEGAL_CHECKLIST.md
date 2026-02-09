# ✅ Checklist Legal y Ético - Web Mirroring

**IMPORTANTE**: Completa este checklist ANTES de ejecutar cualquier herramienta de mirroring.

## 📋 Checklist Pre-Ejecución

### 1. Permisos y Autorización

- [ ] **Tengo permiso explícito del propietario del sitio web**
  - [ ] Permiso por escrito (email, contrato, etc.)
  - [ ] O el sitio tiene política pública que permite archiving/mirroring
  - [ ] O es mi propio sitio web

- [ ] **He revisado los términos de servicio del sitio**
  - [ ] No prohíben explícitamente scraping/mirroring
  - [ ] No hay restricciones de uso de contenido
  - [ ] Entiendo las limitaciones de uso

### 2. robots.txt

- [ ] **He verificado robots.txt**
  - [ ] Ejecuté: `python scripts/validate_legal.py --url <URL>`
  - [ ] robots.txt permite crawling para mi user agent
  - [ ] No hay `Disallow: /` que bloquee todo
  - [ ] Respeta cualquier `Crawl-delay` especificado

### 3. Propósito Legítimo

- [ ] **El propósito del mirroring es legítimo:**
  - [ ] Archivado personal/backup
  - [ ] Pruebas offline/desarrollo
  - [ ] Migración de contenido (con permiso)
  - [ ] Investigación académica (con aprobación ética si aplica)
  - [ ] **NO es para:**
    - [ ] Redistribución comercial sin permiso
    - [ ] Competencia desleal
    - [ ] Violación de copyright
    - [ ] Acceso no autorizado a contenido privado

### 4. Contenido y Datos

- [ ] **No recolecto datos personales sin consentimiento**
  - [ ] No descargo secciones de usuarios/perfiles
  - [ ] No accedo a APIs de datos personales
  - [ ] Respeto políticas de privacidad

- [ ] **Respeto derechos de autor**
  - [ ] No redistribuiré contenido con copyright públicamente sin permiso
  - [ ] Entiendo que el mirroring es para uso personal/legítimo
  - [ ] No usaré el contenido para crear productos comerciales sin licencia

### 5. Configuración Técnica

- [ ] **He configurado rate limiting apropiado**
  - [ ] No sobrecargaré el servidor
  - [ ] Respeto `Crawl-delay` de robots.txt
  - [ ] Uso user agent identificable

- [ ] **No intento eludir medidas de protección**
  - [ ] No bypass de CAPTCHA
  - [ ] No acceso sin credenciales a áreas privadas
  - [ ] No elusión de paywalls sin suscripción válida

### 6. Almacenamiento y Seguridad

- [ ] **Manejo seguro de credenciales**
  - [ ] No guardo contraseñas en archivos de código
  - [ ] Uso variables de entorno para datos sensibles
  - [ ] No comparto credenciales públicamente

- [ ] **Almacenamiento apropiado**
  - [ ] El contenido descargado se almacena de forma segura
  - [ ] No comparto contenido descargado públicamente sin permiso
  - [ ] Respeto restricciones de almacenamiento de datos

## 🚨 Señales de Alerta - NO PROCEDER SI:

- ❌ robots.txt tiene `Disallow: /` para tu user agent
- ❌ Términos de servicio prohíben explícitamente scraping
- ❌ El sitio requiere autenticación y no tienes credenciales válidas
- ❌ El sitio tiene CAPTCHA o medidas anti-bot activas
- ❌ No tienes permiso del propietario
- ❌ El propósito no es legítimo

## 📝 Registro de Decisión

**Fecha**: _________________

**URL objetivo**: _________________

**Propósito**: _________________

**Permiso obtenido**: [ ] Sí [ ] No

**robots.txt verificado**: [ ] Sí [ ] No - Resultado: _________________

**Términos de servicio revisados**: [ ] Sí [ ] No

**Firma/Confirmación**: _________________

---

## ⚖️ Disclaimer

Este checklist es una guía, pero **no constituye asesoramiento legal**. Si tienes dudas sobre la legalidad de tu uso, consulta con un abogado especializado en propiedad intelectual y leyes de internet.

**El usuario es el único responsable del uso de estas herramientas y debe cumplir con todas las leyes aplicables.**

## 📞 Recursos

- [robots.txt Specification](https://www.robotstxt.org/)
- [Web Scraping Legal Guide](https://www.eff.org/issues/coders/scraping-legal-guide)
- [GDPR Compliance](https://gdpr.eu/) (si aplica en tu jurisdicción)



