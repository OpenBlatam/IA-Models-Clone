#!/bin/bash

echo "============================================================"
echo "  Configuración de Token de HuggingFace"
echo "============================================================"
echo ""
echo "Este script te ayudará a configurar el token necesario"
echo "para acceder al modelo Flux2."
echo ""
echo "IMPORTANTE: Necesitas:"
echo "  1. Una cuenta en HuggingFace (gratis)"
echo "  2. Un token de acceso (Read permission)"
echo "  3. Aceptar los términos del modelo"
echo ""
echo "============================================================"
echo ""

read -p "Pega tu token de HuggingFace aquí: " TOKEN

if [ -z "$TOKEN" ]; then
    echo ""
    echo "Error: No se proporcionó token."
    exit 1
fi

echo ""
echo "Configurando token..."

# Para esta sesión
export HUGGINGFACE_TOKEN="$TOKEN"

# Agregar a .bashrc o .zshrc si el usuario quiere
read -p "¿Deseas guardar el token permanentemente? (s/n): " SAVE_PERMANENT

if [ "$SAVE_PERMANENT" = "s" ] || [ "$SAVE_PERMANENT" = "S" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        echo "" >> "$HOME/.bashrc"
        echo "# HuggingFace Token for Character Clothing Changer AI" >> "$HOME/.bashrc"
        echo "export HUGGINGFACE_TOKEN=\"$TOKEN\"" >> "$HOME/.bashrc"
        echo ""
        echo "✅ Token agregado a ~/.bashrc"
        echo "   Ejecuta: source ~/.bashrc"
    elif [ -f "$HOME/.zshrc" ]; then
        echo "" >> "$HOME/.zshrc"
        echo "# HuggingFace Token for Character Clothing Changer AI" >> "$HOME/.zshrc"
        echo "export HUGGINGFACE_TOKEN=\"$TOKEN\"" >> "$HOME/.zshrc"
        echo ""
        echo "✅ Token agregado a ~/.zshrc"
        echo "   Ejecuta: source ~/.zshrc"
    else
        echo ""
        echo "⚠️  No se encontró .bashrc ni .zshrc"
        echo "   El token está configurado solo para esta sesión."
    fi
fi

echo ""
echo "============================================================"
echo "  ✅ Token configurado exitosamente!"
echo "============================================================"
echo ""
echo "Ahora puedes ejecutar:"
echo "  python run_server.py"
echo ""
echo "============================================================"
echo ""


