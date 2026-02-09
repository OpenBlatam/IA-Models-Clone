#!/bin/bash
# httrack_mirror.sh - Script de configuración para mirroring con HTTrack
#
# Uso: bash httrack_mirror.sh <URL> [OUTPUT_DIR]
#
# Ejemplo: bash httrack_mirror.sh https://www.tesla.com ./output/tesla
#
# Nota: Requiere HTTrack instalado
#       Ubuntu/Debian: sudo apt-get install httrack
#       macOS: brew install httrack
#       Windows: Descargar desde https://www.httrack.com/

set -e

URL="${1:-}"
OUTPUT_DIR="${2:-./output/httrack-mirror}"

if [ -z "$URL" ]; then
    echo "❌ Error: Debes proporcionar una URL"
    echo "Uso: bash httrack_mirror.sh <URL> [OUTPUT_DIR]"
    exit 1
fi

# Validar URL
if [[ ! "$URL" =~ ^https?:// ]]; then
    echo "❌ Error: URL debe comenzar con http:// o https://"
    exit 1
fi

# Verificar si httrack está instalado
if ! command -v httrack &> /dev/null; then
    echo "❌ Error: HTTrack no está instalado"
    echo ""
    echo "Instalación:"
    echo "  Ubuntu/Debian: sudo apt-get install httrack"
    echo "  macOS: brew install httrack"
    echo "  Windows: https://www.httrack.com/"
    exit 1
fi

echo "🚀 Iniciando mirroring con HTTrack"
echo "🌐 URL: $URL"
echo "📁 Directorio de salida: $OUTPUT_DIR"
echo ""

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

# Extraer dominio para filtros
DOMAIN=$(echo "$URL" | sed -E 's|^https?://([^/]+).*|\1|')

echo "📥 Descargando sitio..."
httrack "$URL" \
    -O "$OUTPUT_DIR" \
    "+*${DOMAIN}/*" \
    -v \
    --sockets=2 \
    --bandwidth=100000 \
    --disable-security-limits \
    --user-agent "DevinWebMirror/1.0" \
    --robots=1 \
    --stay-on-same-domain \
    --can-go-down \
    --can-go-up \
    --near \
    --timeout=30 \
    --retries=3

echo ""
echo "✅ Mirroring completado"
echo "📁 Archivos guardados en: $OUTPUT_DIR"



