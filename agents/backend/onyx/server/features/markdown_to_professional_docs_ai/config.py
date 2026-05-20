"""Configuration for Markdown to Professional Documents AI"""
from pydantic_settings import BaseSettings
from typing import List, Optional


# Default configuration values
DEFAULT_PORT = 8035
DEFAULT_CORS_ORIGINS = ["*"]
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_TEMP_DIR = "./temp"
DEFAULT_RATE_LIMIT_REQUESTS = 100
DEFAULT_RATE_LIMIT_WINDOW = 60  # seconds


class Settings(BaseSettings):
    app_name: str = "Markdown to Professional Documents AI"
    app_version: str = "1.8.0"
    port: int = DEFAULT_PORT
    cors_origins: List[str] = DEFAULT_CORS_ORIGINS
    max_file_size: int = DEFAULT_MAX_FILE_SIZE
    output_dir: str = DEFAULT_OUTPUT_DIR
    temp_dir: str = DEFAULT_TEMP_DIR
    
    # Rate limiting
    rate_limit_requests: int = DEFAULT_RATE_LIMIT_REQUESTS
    rate_limit_window: int = DEFAULT_RATE_LIMIT_WINDOW
    
    # AI/LLM settings (optional, for enhanced conversions)
    openrouter_api_key: Optional[str] = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o"
    
    # Chart/Diagram settings
    chart_theme: str = "plotly_white"
    diagram_format: str = "png"  # png, svg, pdf
    
    # PDF settings
    pdf_engine: str = "weasyprint"  # weasyprint, reportlab
    pdf_page_size: str = "A4"
    
    # Excel settings
    excel_engine: str = "openpyxl"  # openpyxl, xlsxwriter
    
    # Image processing
    max_image_size_mb: int = 5
    max_image_dimension: int = 2000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
