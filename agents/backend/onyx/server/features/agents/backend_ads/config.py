# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    APP_NAME: str = "Ads IA API"
    LOG_LEVEL: str = "INFO"

    # DeepSeek Configuration
    DEEPSEEK_API_KEY: str = "sk-c5d0a8db81d14950ae13ec8434c6bfad"
    DEEPSEEK_API_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"

    # LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: Optional[str] = None # Será None si no está en .env
    DEFAULT_SYSTEM_PROMPT: str = "Eres un asistente de IA útil y creativo."
    LLM_TEMPERATURE: float = 0.7
    LLM_REQUEST_TIMEOUT_SECONDS: int = 120
    LLM_MAX_RETRIES: int = 2 # Para ChatOllama

    # Scraper Configuration
    USER_AGENT_SCRAPER: str = "Mozilla/5.0 (Python Config Scraper) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    SCRAPER_TIMEOUT_SECONDS: int = 20
    SCRAPER_MAX_CHARS: int = 15000
    SCRAPER_CACHE_SIZE: int = 100
    SCRAPER_CACHE_TTL_SECONDS: int = 300 # 5 minutos

# Instancia global de la configuración para ser importada en otros módulos
settings = Settings()

# Validar configuración esencial al cargar
if not settings.DEEPSEEK_API_KEY:
    print(f"ALERTA CRÍTICA: La variable de entorno DEEPSEEK_API_KEY no está configurada en el archivo .env o en el entorno.")
    print("La aplicación podría no funcionar correctamente sin una API key de DeepSeek.")

if not settings.LLM_MODEL_NAME:
    # Usar logger aquí puede ser problemático si el logger no está configurado aún.
    # Print es más seguro para errores de configuración críticos al inicio.
    print(f"ALERTA CRÍTICA: La variable de entorno LLM_MODEL_NAME no está configurada en el archivo .env o en el entorno.")
    print("La aplicación podría no funcionar correctamente sin un modelo LLM especificado.")
    # Podrías optar por lanzar un SystemExit aquí si es un error irrecuperable:
    # import sys
    # sys.exit("Error: LLM_MODEL_NAME no configurado. La aplicación no puede iniciar.")