"""
AWS Lambda handler for Music Analyzer AI
Optimized for serverless deployment with cold start reduction
"""

import json
import logging
from typing import Dict, Any, Optional
from mangum import Mangum
from contextlib import asynccontextmanager

# Configure logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app instance (reused across invocations to reduce cold starts)
_app: Optional[Any] = None
_handler: Optional[Any] = None


def get_app():
    """Lazy load FastAPI app to reduce cold start time"""
    global _app
    if _app is None:
        # Import here to reduce cold start
        import sys
        import os
        
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from main import app
        _app = app
        logger.info("FastAPI app initialized")
    return _app


def get_handler():
    """Get or create Mangum handler (reused across invocations)"""
    global _handler
    if _handler is None:
        from mangum import Mangum
        _handler = Mangum(
            get_app(),
            lifespan="off",  # Disable lifespan events for Lambda
            log_level="info"
        )
        logger.info("Mangum handler initialized")
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler entry point
    
    Args:
        event: Lambda event (API Gateway event)
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Log request for CloudWatch
        logger.info(f"Received event: {json.dumps(event.get('path', 'unknown'))}")
        
        # Get handler and process request
        handler = get_handler()
        response = handler(event, context)
        
        # Log response status
        if isinstance(response, dict) and 'statusCode' in response:
            logger.info(f"Response status: {response['statusCode']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
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


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        "httpMethod": "GET",
        "path": "/",
        "headers": {},
        "queryStringParameters": None,
        "body": None
    }
    
    class TestContext:
        def __init__(self):
            self.function_name = "music-analyzer-ai"
            self.memory_limit_in_mb = 512
            self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:music-analyzer-ai"
            self.aws_request_id = "test-request-id"
    
    result = lambda_handler(test_event, TestContext())
    print(json.dumps(result, indent=2))




