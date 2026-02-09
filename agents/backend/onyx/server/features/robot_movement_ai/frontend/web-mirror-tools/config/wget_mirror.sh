#!/bin/bash
# wget_mirror.sh - Script de configuración para mirroring con wget
#
# Uso: bash wget_mirror.sh <URL> [OUTPUT_DIR]
#
# Ejemplo: bash wget_mirror.sh https://www.tesla.com ./output/tesla

set -e

URL="${1:-}"
OUTPUT_DIR="${2:-./output/wget-mirror}"

if [ -z "$URL" ]; then
    echo "❌ Error: Debes proporcionar una URL"
    echo "Uso: bash wget_mirror.sh <URL> [OUTPUT_DIR]"
    exit 1
fi

# Validar URL
if [[ ! "$URL" =~ ^https?:// ]]; then
    echo "❌ Error: URL debe comenzar con http:// o https://"
    exit 1
fi

echo "🚀 Iniciando mirroring con wget"
echo "🌐 URL: $URL"
echo "📁 Directorio de salida: $OUTPUT_DIR"
echo ""

# Crear directorio de salida
mkdir -p "$OUTPUT_DIR"

# Modo simulación (spider) - Comentar para ejecutar realmente
# echo "⚠️  MODO SIMULACIÓN - No se descargará nada"
# wget --mirror \
#      --convert-links \
#      --adjust-extension \
#      --page-requisites \
#      --no-parent \
#      --reject-regex '\?.*' \
#      --wait=1 \
#      --limit-rate=100k \
#      --user-agent="DevinWebMirror/1.0" \
#      --spider \
#      "$URL"

# Modo real - Descomentar para ejecutar
echo "📥 Descargando sitio..."
wget --mirror \
     --convert-links \
     --adjust-extension \
     --page-requisites \
     --no-parent \
     --reject-regex '\?.*' \
     --wait=1 \
     --limit-rate=100k \
     --user-agent="DevinWebMirror/1.0" \
     --directory-prefix="$OUTPUT_DIR" \
     --no-clobber \
     --timestamping \
     --no-verbose \
     --progress=bar \
     -e robots=on \
     "$URL"

echo ""
echo "✅ Mirroring completado"
echo "📁 Archivos guardados en: $OUTPUT_DIR"



