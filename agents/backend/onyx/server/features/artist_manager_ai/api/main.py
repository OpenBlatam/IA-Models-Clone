"""
Main Application Entry Point
=============================

Punto de entrada principal de la aplicación.
"""

from .app_factory import create_app

# Crear aplicación
app = create_app(
    title="Artist Manager AI",
    version="1.0.0",
    enable_auth=True,
    enable_rate_limit=True,
    enable_cors=True
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




