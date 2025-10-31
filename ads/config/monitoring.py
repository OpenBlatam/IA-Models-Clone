from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
BUFFER_SIZE = 1024

from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Monitoring configuration for the ads module.
"""

class SentrySettings(BaseModel):
    """Sentry settings."""
    dsn: str = Field(default="")
    environment: str = Field(default="development")
    traces_sample_rate: float = Field(default=1.0)
    enable_performance_monitoring: bool = Field(default=True)
    enable_error_monitoring: bool = Field(default=True)

class PrometheusSettings(BaseModel):
    """Prometheus settings."""
    enable_metrics: bool = Field(default=True)
    metrics_prefix: str = Field(default="ads_")
    scrape_interval: int = Field(default=15)
    scrape_timeout: int = Field(default=10)
    evaluation_interval: int = Field(default=15)

class LoggingSettings(BaseModel):
    """Logging settings."""
    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    date_format: str = Field(default="%Y-%m-%d %H:%M:%S")
    log_file: str = Field(default="./logs/ads.log")
    max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
    backup_count: int = Field(default=5)

class MonitoringSettings(BaseSettings):
    """Monitoring settings."""
    sentry: SentrySettings = Field(default_factory=SentrySettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Health check settings
    health_check_interval: int = Field(default=30)
    health_check_timeout: int = Field(default=5)
    enable_health_checks: bool = Field(default=True)
    
    # Alerting settings
    enable_alerts: bool = Field(default=True)
    alert_threshold: float = Field(default=0.9)
    alert_cooldown: int = Field(default=300)  # 5 minutes
    
    @dataclass
class Config:
        env_prefix = "MONITORING_"
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global monitoring settings instance
monitoring_settings = MonitoringSettings() 