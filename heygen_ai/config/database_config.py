from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from api.core.async_database import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Database Configuration for HeyGen AI API
Environment-based configuration for multiple database backends
"""


    DatabaseConfig,
    DatabaseType,
    create_postgresql_config,
    create_mysql_config,
    create_sqlite_config
)


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Database type and connection
    database_type: str = Field(default="sqlite", env="DATABASE_TYPE")
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_database: str = Field(default="heygen_ai", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    
    # MySQL settings
    mysql_host: str = Field(default="localhost", env="MYSQL_HOST")
    mysql_port: int = Field(default=3306, env="MYSQL_PORT")
    mysql_database: str = Field(default="heygen_ai", env="MYSQL_DB")
    mysql_user: str = Field(default="root", env="MYSQL_USER")
    mysql_password: str = Field(default="", env="MYSQL_PASSWORD")
    
    # SQLite settings
    sqlite_database: str = Field(default="heygen_ai.db", env="SQLITE_DATABASE")
    
    # Connection pool settings
    pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DATABASE_POOL_RECYCLE")
    pool_pre_ping: bool = Field(default=True, env="DATABASE_POOL_PRE_PING")
    
    # Development settings
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    echo_pool: bool = Field(default=False, env="DATABASE_ECHO_POOL")
    
    # Health monitoring
    health_check_interval: int = Field(default=30, env="DATABASE_HEALTH_CHECK_INTERVAL")
    health_check_enabled: bool = Field(default=True, env="DATABASE_HEALTH_CHECK_ENABLED")
    
    @dataclass
class Config:
        env_prefix = "HEYGEN_"
        case_sensitive = False


def get_database_config() -> DatabaseConfig:
    """Get database configuration based on environment settings."""
    settings = DatabaseSettings()
    
    # If DATABASE_URL is provided, use it directly
    if settings.database_url:
        if settings.database_url.startswith("postgresql"):
            return DatabaseConfig(
                url=settings.database_url,
                type=DatabaseType.POSTGRESQL,
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
                pool_timeout=settings.pool_timeout,
                pool_recycle=settings.pool_recycle,
                pool_pre_ping=settings.pool_pre_ping,
                echo=settings.echo,
                echo_pool=settings.echo_pool
            )
        elif settings.database_url.startswith("mysql"):
            return DatabaseConfig(
                url=settings.database_url,
                type=DatabaseType.MYSQL,
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
                pool_timeout=settings.pool_timeout,
                pool_recycle=settings.pool_recycle,
                pool_pre_ping=settings.pool_pre_ping,
                echo=settings.echo,
                echo_pool=settings.echo_pool
            )
        elif settings.database_url.startswith("sqlite"):
            return DatabaseConfig(
                url=settings.database_url,
                type=DatabaseType.SQLITE,
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
                pool_timeout=settings.pool_timeout,
                pool_recycle=settings.pool_recycle,
                pool_pre_ping=settings.pool_pre_ping,
                echo=settings.echo,
                echo_pool=settings.echo_pool
            )
    
    # Otherwise, use type-specific configuration
    if settings.database_type.lower() == "postgresql":
        return create_postgresql_config(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            username=settings.postgres_user,
            password=settings.postgres_password,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=settings.pool_pre_ping,
            echo=settings.echo,
            echo_pool=settings.echo_pool
        )
    
    elif settings.database_type.lower() == "mysql":
        return create_mysql_config(
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_database,
            username=settings.mysql_user,
            password=settings.mysql_password,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=settings.pool_pre_ping,
            echo=settings.echo,
            echo_pool=settings.echo_pool
        )
    
    else:  # Default to SQLite
        return create_sqlite_config(
            database_path=settings.sqlite_database,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=settings.pool_pre_ping,
            echo=settings.echo,
            echo_pool=settings.echo_pool
        )


def get_development_config() -> DatabaseConfig:
    """Get development database configuration."""
    return create_sqlite_config(
        database_path="heygen_ai_dev.db",
        pool_size=10,
        max_overflow=20,
        echo=True,  # Enable SQL logging for development
        echo_pool=True
    )


def get_testing_config() -> DatabaseConfig:
    """Get testing database configuration."""
    return create_sqlite_config(
        database_path=":memory:",  # In-memory database for testing
        pool_size=5,
        max_overflow=10,
        echo=False
    )


def get_production_config() -> DatabaseConfig:
    """Get production database configuration."""
    settings = DatabaseSettings()
    
    if settings.database_type.lower() == "postgresql":
        return create_postgresql_config(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_database,
            username=settings.postgres_user,
            password=settings.postgres_password,
            pool_size=50,  # Higher pool size for production
            max_overflow=100,
            pool_timeout=60,
            pool_recycle=1800,  # Recycle every 30 minutes
            pool_pre_ping=True,
            echo=False,  # Disable SQL logging for production
            connect_args={
                "server_settings": {
                    "application_name": "heygen_ai",
                    "timezone": "UTC"
                },
                "command_timeout": 60,
                "statement_timeout": 30000
            }
        )
    
    elif settings.database_type.lower() == "mysql":
        return create_mysql_config(
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_database,
            username=settings.mysql_user,
            password=settings.mysql_password,
            pool_size=50,
            max_overflow=100,
            pool_timeout=60,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False,
            connect_args={
                "charset": "utf8mb4",
                "autocommit": False,
                "sql_mode": "STRICT_TRANS_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO"
            }
        )
    
    else:
        raise ValueError(f"Unsupported database type for production: {settings.database_type}")


def get_database_config_by_environment(environment: str) -> DatabaseConfig:
    """Get database configuration based on environment."""
    if environment == "development":
        return get_development_config()
    elif environment == "testing":
        return get_testing_config()
    elif environment == "production":
        return get_production_config()
    else:
        return get_database_config()


# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    "development": {
        "database_type": "sqlite",
        "database_url": "sqlite+aiosqlite:///heygen_ai_dev.db",
        "pool_size": 10,
        "max_overflow": 20,
        "echo": True,
        "echo_pool": True
    },
    "testing": {
        "database_type": "sqlite",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "pool_size": 5,
        "max_overflow": 10,
        "echo": False
    },
    "production": {
        "database_type": "postgresql",
        "database_url": "postgresql+asyncpg://user:pass@localhost/heygen_ai",
        "pool_size": 50,
        "max_overflow": 100,
        "pool_timeout": 60,
        "pool_recycle": 1800,
        "echo": False
    }
}


def get_environment_database_config(environment: str) -> Dict[str, Any]:
    """Get environment-specific database configuration."""
    return ENVIRONMENT_CONFIGS.get(environment, ENVIRONMENT_CONFIGS["development"])


# Example usage functions
def setup_development_database():
    """Setup development database configuration."""
    os.environ.setdefault("HEYGEN_DATABASE_TYPE", "sqlite")
    os.environ.setdefault("HEYGEN_DATABASE_ECHO", "true")
    return get_development_config()


def setup_production_database():
    """Setup production database configuration."""
    os.environ.setdefault("HEYGEN_DATABASE_TYPE", "postgresql")
    os.environ.setdefault("HEYGEN_DATABASE_ECHO", "false")
    return get_production_config()


def setup_testing_database():
    """Setup testing database configuration."""
    os.environ.setdefault("HEYGEN_DATABASE_TYPE", "sqlite")
    os.environ.setdefault("HEYGEN_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    return get_testing_config() 