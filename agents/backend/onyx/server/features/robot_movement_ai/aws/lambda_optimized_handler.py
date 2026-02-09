"""
Optimized Lambda Handler
=========================

Serverless-optimized handler with:
- Cold start reduction
- Connection pooling
- Lazy loading
- Memory optimization
"""

import json
import logging
import os
from typing import Dict, Any
from mangum import Mangum
from aws.optimization.serverless_optimizer import get_serverless_optimizer
from aws.core.app_factory import create_modular_robot_app
from robot_movement_ai.config.robot_config import RobotConfig, RobotBrand

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global app instance (reused across invocations)
_app = None
_handler = None
_optimizer = None


async def warm_up():
    """Warm up connections and caches."""
    global _optimizer
    if _optimizer is None:
        _optimizer = get_serverless_optimizer()
    
    await _optimizer.warm_up({
        "redis_url": os.getenv("REDIS_URL"),
    })


def get_app():
    """Get or create FastAPI app instance."""
    global _app
    
    if _app is None:
        # Lazy load heavy dependencies
        from aws.optimization.serverless_optimizer import LazyLoader
        
        # Load configuration
        config = RobotConfig(
            robot_brand=RobotBrand(os.getenv("ROBOT_BRAND", "generic")),
            ros_enabled=os.getenv("ROS_ENABLED", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            feedback_frequency=int(os.getenv("FEEDBACK_FREQUENCY", "1000")),
            robot_ip=os.getenv("ROBOT_IP"),
            robot_port=int(os.getenv("ROBOT_PORT", "30001")),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_api_key=os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
            api_cors_enabled=True,
            api_cors_origins=["*"]
        )
        
        # Create app with modular system
        from aws.core.config_manager import AppConfig
        app_config = AppConfig.from_env()
        _app = create_modular_robot_app(config, app_config)
        
        logger.info("FastAPI app initialized for Lambda (optimized)")
    
    return _app


def get_handler():
    """Get or create Mangum handler."""
    global _handler
    
    if _handler is None:
        app = get_app()
        _handler = Mangum(app, lifespan="off")
        logger.info("Mangum handler initialized")
    
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Optimized AWS Lambda handler.
    
    Features:
    - Connection pooling
    - Lazy loading
    - Cold start optimization
    """
    try:
        # Warm up on first invocation
        if _optimizer is None:
            import asyncio
            asyncio.run(warm_up())
        
        handler = get_handler()
        response = handler(event, context)
        return response
    except Exception as e:
        logger.error(f"Error in Lambda handler: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }















