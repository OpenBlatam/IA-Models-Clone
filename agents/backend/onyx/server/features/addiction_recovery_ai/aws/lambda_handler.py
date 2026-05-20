"""
AWS Lambda Handler for Addiction Recovery AI
Optimized for serverless deployment with cold start minimization
"""

import json
import logging
from typing import Any, Dict, Optional
from mangum import Mangum
from fastapi import FastAPI

# Configure logging for CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global app instance (reused across invocations)
_app: Optional[FastAPI] = None
_handler: Optional[Mangum] = None


def get_app() -> FastAPI:
    """Get or create FastAPI app instance (singleton pattern for Lambda)"""
    global _app
    
    if _app is None:
        logger.info("Initializing FastAPI app (cold start)...")
        
        # Import here to minimize cold start time
        try:
            from main import app
            _app = app
        except ImportError:
            # Fallback for relative imports
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from main import app
            _app = app
        
        logger.info("FastAPI app initialized successfully")
    
    return _app


def get_handler() -> Mangum:
    """Get or create Mangum handler instance"""
    global _handler
    
    if _handler is None:
        logger.info("Initializing Mangum handler...")
        app = get_app()
        _handler = Mangum(
            app,
            lifespan="off",  # Disable lifespan events in Lambda
            api_gateway_base_path="/",  # Adjust if using custom domain
            text_mime_types=["application/json", "text/plain"]
        )
        logger.info("Mangum handler initialized successfully")
    
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function
    
    Args:
        event: Lambda event (API Gateway event)
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Log request for CloudWatch
        logger.info(f"Received event: {json.dumps(event, default=str)}")
        
        # Get handler and process request
        handler = get_handler()
        response = handler(event, context)
        
        # Log response status
        if isinstance(response, dict) and "statusCode" in response:
            logger.info(f"Response status: {response['statusCode']}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        
        # Return error response
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e) if logger.level <= logging.DEBUG else "An error occurred"
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
        "body": None,
        "requestContext": {
            "requestId": "test-request-id",
            "stage": "test"
        }
    }
    
    class TestContext:
        function_name = "test-function"
        function_version = "$LATEST"
        invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"
        memory_limit_in_mb = "512"
        aws_request_id = "test-request-id"
    
    context = TestContext()
    response = lambda_handler(test_event, context)
    print(json.dumps(response, indent=2))















