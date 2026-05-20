"""
AWS Lambda handler for Faceless Video AI (Serverless option)
Optimized for minimal cold start times
"""

import json
import os
import sys
from pathlib import Path

# Add application to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mangum import Mangum
from api.main import app

# Initialize handler
handler = Mangum(app, lifespan="off")


def lambda_handler(event, context):
    """
    AWS Lambda handler
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    try:
        # Handle API Gateway event
        response = handler(event, context)
        return response
    except Exception as e:
        # Return error response
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
        "path": "/health",
        "headers": {},
        "queryStringParameters": None,
        "body": None,
        "isBase64Encoded": False
    }
    
    class TestContext:
        aws_request_id = "test-request-id"
        function_name = "test-function"
        function_version = "$LATEST"
        memory_limit_in_mb = 512
    
    context = TestContext()
    response = lambda_handler(test_event, context)
    print(json.dumps(response, indent=2))




