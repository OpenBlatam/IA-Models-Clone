"""
Deployment configurations for cloud platforms
"""

from .aws.lambda_handler import lambda_handler
from .aws.serverless_config import ServerlessConfig, optimize_for_lambda
from .azure.function_app import function_app
from .azure.serverless_config import AzureServerlessConfig, optimize_for_azure_functions

__all__ = [
    "lambda_handler",
    "ServerlessConfig",
    "optimize_for_lambda",
    "function_app",
    "AzureServerlessConfig",
    "optimize_for_azure_functions"
]
