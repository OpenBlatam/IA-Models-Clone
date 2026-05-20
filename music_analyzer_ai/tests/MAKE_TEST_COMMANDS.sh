#!/bin/bash

# Script de utilidades para ejecutar tests
# Uso: ./MAKE_TEST_COMMANDS.sh [comando]

set -e

case "$1" in
    "all")
        echo "🚀 Ejecutando todos los tests..."
        pytest -v
        ;;
    
    "coverage")
        echo "📊 Ejecutando tests con cobertura..."
        pytest --cov=music_analyzer_ai --cov-report=html --cov-report=term
        echo "✅ Reporte HTML generado en htmlcov/index.html"
        ;;
    
    "fast")
        echo "⚡ Ejecutando tests rápidos (sin slow)..."
        pytest -v -m "not slow"
        ;;
    
    "unit")
        echo "🔬 Ejecutando tests unitarios..."
        pytest -v -m unit
        ;;
    
    "integration")
        echo "🔗 Ejecutando tests de integración..."
        pytest -v -m integration
        ;;
    
    "api")
        echo "🌐 Ejecutando tests de API..."
        pytest -v -m api tests/test_api.py
        ;;
    
    "parallel")
        echo "⚡ Ejecutando tests en paralelo..."
        pytest -n auto -v
        ;;
    
    "watch")
        echo "👀 Modo watch (requiere pytest-watch)..."
        ptw --runner "pytest -v"
        ;;
    
    "failed")
        echo "🔍 Ejecutando solo tests que fallaron..."
        pytest --lf -v
        ;;
    
    "clean")
        echo "🧹 Limpiando caché y archivos temporales..."
        pytest --cache-clear
        find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
        find . -type d -name "htmlcov" -exec rm -r {} + 2>/dev/null || true
        find . -type d -name ".coverage" -exec rm -r {} + 2>/dev/null || true
        echo "✅ Limpieza completada"
        ;;
    
    "help"|*)
        echo "📚 Comandos disponibles:"
        echo ""
        echo "  ./MAKE_TEST_COMMANDS.sh all          - Ejecutar todos los tests"
        echo "  ./MAKE_TEST_COMMANDS.sh coverage      - Tests con cobertura"
        echo "  ./MAKE_TEST_COMMANDS.sh fast          - Tests rápidos (sin slow)"
        echo "  ./MAKE_TEST_COMMANDS.sh unit          - Solo tests unitarios"
        echo "  ./MAKE_TEST_COMMANDS.sh integration   - Solo tests de integración"
        echo "  ./MAKE_TEST_COMMANDS.sh api           - Solo tests de API"
        echo "  ./MAKE_TEST_COMMANDS.sh parallel      - Tests en paralelo"
        echo "  ./MAKE_TEST_COMMANDS.sh watch         - Modo watch (auto-ejecutar)"
        echo "  ./MAKE_TEST_COMMANDS.sh failed        - Solo tests fallidos"
        echo "  ./MAKE_TEST_COMMANDS.sh clean         - Limpiar caché"
        echo "  ./MAKE_TEST_COMMANDS.sh help          - Mostrar esta ayuda"
        ;;
esac

