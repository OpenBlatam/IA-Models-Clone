#!/bin/bash

echo "========================================"
echo "Ejecutando Todas las Pruebas - API BUL"
echo "========================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python no encontrado"
    exit 1
fi

PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python)

echo "[1/3] Verificando servidor..."
if ! curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
    echo "ERROR: Servidor no está corriendo"
    echo "Por favor inicia: python api_frontend_ready.py"
    exit 1
fi

echo "[2/3] Ejecutando pruebas básicas..."
$PYTHON_CMD test_api_responses.py
BASIC_EXIT=$?

echo ""
echo "[3/4] Ejecutando pruebas avanzadas..."
$PYTHON_CMD test_api_advanced.py
ADVANCED_EXIT=$?

echo ""
echo "[4/4] Ejecutando pruebas de seguridad..."
$PYTHON_CMD test_security.py
SECURITY_EXIT=$?

echo ""
echo "========================================"
echo "Pruebas completadas"
echo "Revisa los archivos de resultados:"
echo "- test_results.json"
echo "- test_results.csv"
echo "- test_dashboard.html"
echo "========================================"

# Abrir dashboard si está disponible
if command -v xdg-open &> /dev/null; then
    xdg-open test_dashboard.html 2>/dev/null &
elif command -v open &> /dev/null; then
    open test_dashboard.html 2>/dev/null &
fi

if [ $BASIC_EXIT -ne 0 ] || [ $ADVANCED_EXIT -ne 0 ] || [ $SECURITY_EXIT -ne 0 ]; then
    exit 1
fi

