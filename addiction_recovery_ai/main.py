"""
Servidor principal para Addiction Recovery AI
Refactored to use modular app factory pattern
"""

from core.app_factory import create_app
from config.app_config import get_config

app = create_app()

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(app, host=config.host, port=config.port)

