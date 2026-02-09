"""
Settings and configuration management
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # OpenRouter settings
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_enabled: bool = os.getenv("OPENROUTER_ENABLED", "true").lower() == "true"
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4")
    openrouter_temperature: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))
    openrouter_max_tokens: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "2000"))
    
    # TruthGPT settings
    truthgpt_enabled: bool = os.getenv("TRUTHGPT_ENABLED", "true").lower() == "true"
    truthgpt_endpoint: Optional[str] = os.getenv("TRUTHGPT_ENDPOINT")
    truthgpt_timeout: float = float(os.getenv("TRUTHGPT_TIMEOUT", "120.0"))
    
    # ComfyUI settings
    comfyui_api_url: str = os.getenv("COMFYUI_API_URL", "http://localhost:8188")
    comfyui_workflow_path: str = os.getenv(
        "COMFYUI_WORKFLOW_PATH",
        "workflows/flux_fill_clothing_changer.json"
    )
    
    # Image processing settings
    max_image_size: int = int(os.getenv("MAX_IMAGE_SIZE", "10485760"))  # 10MB
    allowed_image_types: list = ["image/png", "image/jpeg", "image/jpg"]
    
    # Output settings
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    save_tensors: bool = os.getenv("SAVE_TENSORS", "true").lower() == "true"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings instance (singleton)"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

