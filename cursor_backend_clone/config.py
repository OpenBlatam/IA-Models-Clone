"""
Configuration for Cursor Agent 24/7
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Directorio base
BASE_DIR = Path(__file__).parent

# Configuración del agente
AGENT_CONFIG = {
    "check_interval": float(os.getenv("AGENT_CHECK_INTERVAL", "1.0")),
    "max_concurrent_tasks": int(os.getenv("AGENT_MAX_CONCURRENT_TASKS", "5")),
    "task_timeout": float(os.getenv("AGENT_TASK_TIMEOUT", "300.0")),
    "auto_restart": os.getenv("AGENT_AUTO_RESTART", "true").lower() == "true",
    "persistent_storage": os.getenv("AGENT_PERSISTENT_STORAGE", "true").lower() == "true",
    "storage_path": os.getenv("AGENT_STORAGE_PATH", str(BASE_DIR / "data" / "agent_state.json")),
}

# Configuración de la API
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8024")),
    "cors_origins": os.getenv("API_CORS_ORIGINS", "*").split(","),
}

# Configuración de Cursor API (cuando esté disponible)
CURSOR_API_CONFIG = {
    "api_key": os.getenv("CURSOR_API_KEY"),
    "api_url": os.getenv("CURSOR_API_URL", "https://api.cursor.sh"),
    "webhook_url": os.getenv("CURSOR_WEBHOOK_URL"),
}


