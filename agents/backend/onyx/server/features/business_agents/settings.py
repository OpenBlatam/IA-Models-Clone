from pydantic import BaseSettings, Field
from typing import List


class AppSettings(BaseSettings):
    app_name: str = Field("Ultimate Quantum AI ML NLP Benchmark", env="APP_NAME")
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"], env="ALLOWED_ORIGINS")
    rps_limit: int = Field(100, env="RPS_LIMIT")
    max_body_bytes: int = Field(5 * 1024 * 1024, env="MAX_BODY_BYTES")
    request_timeout_seconds: float = Field(30.0, env="REQUEST_TIMEOUT_SECONDS")
    enable_otel: bool = Field(False, env="ENABLE_OTEL")
    api_key: str | None = Field(None, env="API_KEY")
    redis_url: str | None = Field(None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(10, env="CACHE_TTL_SECONDS")
    http_timeout_seconds: float = Field(5.0, env="HTTP_TIMEOUT_SECONDS")
    http_retries: int = Field(3, env="HTTP_RETRIES")
    cb_fail_threshold: int = Field(5, env="CB_FAIL_THRESHOLD")
    cb_recovery_seconds: int = Field(30, env="CB_RECOVERY_SECONDS")
    enforce_auth: bool = Field(False, env="ENFORCE_AUTH")
    jwt_secret: str | None = Field(None, env="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_audience: str | None = Field(None, env="JWT_AUDIENCE")
    features: List[str] = Field(default_factory=list, env="FEATURES")
    use_distributed_rate_limit: bool = Field(False, env="USE_DISTRIBUTED_RATE_LIMIT")
    enable_request_logging: bool = Field(False, env="ENABLE_REQUEST_LOGGING")
    log_request_body: bool = Field(False, env="LOG_REQUEST_BODY")
    log_request_headers: bool = Field(False, env="LOG_REQUEST_HEADERS")
    database_url: str | None = Field(None, env="DATABASE_URL")
    enable_profiling: bool = Field(False, env="ENABLE_PROFILING")
    slow_request_threshold: float = Field(1.0, env="SLOW_REQUEST_THRESHOLD")
    
    class Config:
        case_sensitive = False
        env_file = ".env"


def _parse_origins(value: List[str]) -> List[str]:
    if len(value) == 1 and "," in value[0]:
        return [v.strip() for v in value[0].split(",") if v.strip()]
    return value


settings = AppSettings()
settings.allowed_origins = _parse_origins(settings.allowed_origins)
settings.features = _parse_origins(settings.features)


