"""
Ejemplo Simple - Uso básico del AI Project Generator
====================================================

Ejemplo mínimo para empezar rápidamente.
"""

from core.easy_setup import quick_start
import uvicorn

# Crear aplicación (una línea)
app = quick_start()

if __name__ == "__main__":
    # Ejecutar servidor
    uvicorn.run(app, host="0.0.0.0", port=8020)















