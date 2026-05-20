#!/bin/bash

# Test Coverage Script
# Genera reporte de cobertura y abre en navegador

echo "🧪 Running tests with coverage..."
npm run test:coverage

echo "📊 Coverage report generated!"
echo "Opening coverage report in browser..."

# Abrir reporte HTML (ajustar según OS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open coverage/lcov-report/index.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open coverage/lcov-report/index.html
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start coverage/lcov-report/index.html
fi

echo "✅ Done!"

