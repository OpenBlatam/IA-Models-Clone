"""
AWS Lambda Handler for Robot Movement AI
=========================================

Serverless handler for stateless API endpoints.
Note: WebSocket endpoints require ECS/Fargate deployment.
"""

import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mangum import Mangum
    from robot_movement_ai.config.robot_config import RobotConfig, RobotBrand
    from robot_movement_ai.api.robot_api import create_robot_app
except ImportError as e:
    # Fallback for local development
    logging.warning(f"Import error: {e}. Make sure all dependencies are installed.")
    Mangum = None

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global app instance (reused across invocations)
_app: Optional[Any] = None
_handler: Optional[Any] = None


def get_app():
    """Get or create FastAPI app instance."""
    global _app
    
    if _app is None:
        # Load configuration from environment variables
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
            api_cors_origins=["*"]  # Configure based on your API Gateway domain
        )
        
        _app = create_robot_app(config)
        logger.info("FastAPI app initialized for Lambda")
    
    return _app


def get_handler():
    """Get or create Mangum handler."""
    global _handler
    
    if _handler is None:
        if Mangum is None:
            raise ImportError("Mangum is required for Lambda deployment. Install with: pip install mangum")
        app = get_app()
        _handler = Mangum(app, lifespan="off")  # Disable lifespan events for Lambda
        logger.info("Mangum handler initialized")
    
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler entry point.
    
    Args:
        event: Lambda event (API Gateway event)
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
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

