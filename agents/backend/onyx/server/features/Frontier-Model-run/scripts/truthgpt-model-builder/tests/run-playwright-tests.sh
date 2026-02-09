#!/bin/bash
# Script para ejecutar tests de Playwright

echo "🧪 Ejecutando Tests de Playwright"
echo "=================================="
echo ""

# Verificar que el servidor Next.js esté corriendo
echo "⚠️  Asegúrate de que el servidor Next.js esté corriendo:"
echo "   npm run dev"
echo ""
read -p "Presiona Enter cuando el servidor esté listo..."

# Ejecutar tests
echo ""
echo "🚀 Ejecutando tests..."
npm run test:e2e

echo ""
echo "✅ Tests completados!"











