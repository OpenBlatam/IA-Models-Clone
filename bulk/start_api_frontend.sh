#!/bin/bash

echo "========================================"
echo "BUL API - Frontend Ready"
echo "========================================"
echo ""

# Detectar Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "ERROR: Python no encontrado"
    echo "Por favor instale Python desde https://python.org"
    exit 1
fi

echo "Python encontrado: $PYTHON_CMD"
echo ""
echo "Iniciando BUL API para Frontend..."
echo ""
echo "API disponible en: http://localhost:8000"
echo "Documentacion: http://localhost:8000/api/docs"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo ""

$PYTHON_CMD api_frontend_ready.py --host 0.0.0.0 --port 8000
































