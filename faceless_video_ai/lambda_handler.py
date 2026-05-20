"""
AWS Lambda handler for serverless deployment
"""
import json
import os
from mangum import Mangum
from api.main import app

# Initialize Mangum adapter for FastAPI
handler = Mangum(app, lifespan="off")


def lambda_handler(event, context):
    """
    AWS Lambda handler function
    
    This handler wraps the FastAPI application using Mangum,
    which provides an ASGI-to-AWS Lambda adapter.
    """
    try:
        # Set environment variables if needed
        os.environ.setdefault("ENVIRONMENT", "lambda")
        
        # Process the event through Mangum
        response = handler(event, context)
        
        return response
    except Exception as e:
        # Return error response
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e)
            })
        }




