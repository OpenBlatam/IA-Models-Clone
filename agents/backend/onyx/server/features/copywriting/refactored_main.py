from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any
import uvicorn
from contextlib import asynccontextmanager
from refactored_architecture import RefactoredCopywritingAPI
from pydantic_settings import BaseSettings
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Refactored Main Entry Point
===========================

Modern, clean main entry point with:
- Dependency injection
- Configuration management
- Error handling
- Graceful shutdown
- Health monitoring
"""


# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('refactored_copywriting.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class APIConfig(BaseSettings):
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    
    @dataclass
class Config:
        env_file = ".env"
        env_prefix = "API_"

class ModelConfig(BaseSettings):
    """Model configuration settings"""
    model_name: str = "gpt2"
    max_length: int = 512
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_distributed: bool = True
    gpu_memory_fraction: float = 0.8
    
    @dataclass
class Config:
        env_file = ".env"
        env_prefix = "MODEL_"

class CacheConfig(BaseSettings):
    """Cache configuration settings"""
    enable_caching: bool = True
    cache_size: int = 10000
    redis_url: str = "redis://localhost"
    
    @dataclass
class Config:
        env_file = ".env"
        env_prefix = "CACHE_"

class PerformanceConfig(BaseSettings):
    """Performance configuration settings"""
    enable_profiling: bool = True
    enable_monitoring: bool = True
    max_workers: int = 8
    batch_size: int = 32
    
    @dataclass
class Config:
        env_file = ".env"
        env_prefix = "PERF_"

def load_config() -> Dict[str, Any]:
    """Load configuration from environment and files"""
    api_config = APIConfig()
    model_config = ModelConfig()
    cache_config = CacheConfig()
    perf_config = PerformanceConfig()
    
    return {
        "api": api_config.dict(),
        "model": model_config.dict(),
        "cache": cache_config.dict(),
        "performance": perf_config.dict(),
        # Combined config for services
        "model_name": model_config.model_name,
        "max_length": model_config.max_length,
        "enable_gpu": model_config.enable_gpu,
        "enable_caching": cache_config.enable_caching,
        "enable_profiling": perf_config.enable_profiling,
        "enable_monitoring": perf_config.enable_monitoring,
        "enable_distributed": model_config.enable_distributed,
        "enable_quantization": model_config.enable_quantization,
        "cache_size": cache_config.cache_size,
        "gpu_memory_fraction": model_config.gpu_memory_fraction,
        "max_workers": perf_config.max_workers,
        "batch_size": perf_config.batch_size,
        "redis_url": cache_config.redis_url
    }

@asynccontextmanager
async def lifespan_manager():
    """Manage application lifespan"""
    logger.info("Starting Refactored Copywriting System...")
    
    try:
        # Startup tasks
        logger.info("Initializing services...")
        yield
    finally:
        # Shutdown tasks
        logger.info("Shutting down Refactored Copywriting System...")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame) -> Any:
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def create_app() -> RefactoredCopywritingAPI:
    """Create and configure the application"""
    try:
        # Load configuration
        config = load_config()
        
        # Create API
        api = RefactoredCopywritingAPI(config)
        
        logger.info("Application created successfully")
        return api
        
    except Exception as e:
        logger.error(f"Failed to create application: {e}")
        raise

def main():
    """Main application entry point"""
    
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Load configuration
        config = load_config()
        api_config = config["api"]
        
        # Create application
        api = create_app()
        app = api.get_app()
        
        # Start server
        logger.info(f"Starting server on {api_config['host']}:{api_config['port']}")
        
        uvicorn.run(
            app,
            host=api_config["host"],
            port=api_config["port"],
            workers=api_config["workers"],
            reload=api_config["reload"],
            log_level=api_config["log_level"],
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        sys.exit(1)

match __name__:
    case "__main__":
    main() 