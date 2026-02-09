"""
AWS Lambda Handler para Cursor Agent 24/7
==========================================

Handler optimizado para AWS Lambda con:
- Cold start optimization
- Mangum adapter para FastAPI
- CloudWatch logging
- Error handling robusto
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Configurar logging para CloudWatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variables de entorno AWS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_LAMBDA_FUNCTION_NAME = os.getenv("AWS_LAMBDA_FUNCTION_NAME", "cursor-agent-24-7")

# Lazy loading para optimizar cold starts
_app: Optional[Any] = None
_handler: Optional[Any] = None


def get_app():
    """
    Obtener aplicación FastAPI (lazy loading para cold start optimization).
    
    Returns:
        Aplicación FastAPI configurada para Lambda.
    """
    global _app
    
    if _app is None:
        try:
            # Importar después de verificar que estamos en Lambda
            from mangum import Mangum
            from .api.app_config import create_app
            from .core.agent import CursorAgent, AgentConfig
            from .core.aws_adapter import AWSStateManager, AWSCacheAdapter
            
            # Configurar para ambiente serverless
            config = AgentConfig(
                persistent_storage=True,  # Usar DynamoDB en lugar de archivo
                auto_restart=False,  # Lambda se reinicia automáticamente
                storage_path="dynamodb://cursor-agent-state"  # Usar DynamoDB
            )
            
            # Crear agente con adaptadores AWS
            agent = CursorAgent(config)
            
            # Reemplazar state manager con AWS adapter
            if hasattr(agent, 'state_manager'):
                agent.state_manager = AWSStateManager(
                    agent=agent,
                    table_name=os.getenv("DYNAMODB_TABLE_NAME", "cursor-agent-state")
                )
            
            # Reemplazar cache con AWS adapter
            if hasattr(agent, 'cache'):
                agent.cache = AWSCacheAdapter(
                    cache_type=os.getenv("CACHE_TYPE", "elasticache"),  # elasticache o dynamodb
                    endpoint=os.getenv("REDIS_ENDPOINT")
                )
            
            # Crear aplicación FastAPI
            _app = create_app(agent)
            
            logger.info("FastAPI app initialized for Lambda")
            
        except Exception as e:
            logger.error(f"Error initializing app: {e}", exc_info=True)
            raise
    
    return _app


def get_handler():
    """
    Obtener handler Mangum (lazy loading).
    
    Returns:
        Handler Mangum configurado.
    """
    global _handler
    
    if _handler is None:
        try:
            from mangum import Mangum
            
            app = get_app()
            _handler = Mangum(
                app,
                lifespan="off",  # Deshabilitar lifespan events para Lambda
                log_level="info"
            )
            
            logger.info("Mangum handler initialized")
            
        except Exception as e:
            logger.error(f"Error initializing handler: {e}", exc_info=True)
            raise
    
    return _handler


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handler principal de AWS Lambda.
    
    Args:
        event: Evento de Lambda (API Gateway, etc.)
        context: Contexto de Lambda
    
    Returns:
        Respuesta HTTP
    """
    try:
        # Obtener handler (lazy loading)
        lambda_handler = get_handler()
        
        # Procesar request
        response = lambda_handler(event, context)
        
        # Logging para CloudWatch
        logger.info(
            f"Request processed",
            extra={
                "function_name": AWS_LAMBDA_FUNCTION_NAME,
                "request_id": context.aws_request_id if context else None,
                "status_code": response.get("statusCode") if isinstance(response, dict) else None
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Error in Lambda handler: {e}",
            exc_info=True,
            extra={
                "function_name": AWS_LAMBDA_FUNCTION_NAME,
                "request_id": context.aws_request_id if context else None
            }
        )
        
        # Retornar error 500
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": '{"error": "Internal server error", "message": "An error occurred processing the request"}'
        }


# Para testing local
if __name__ == "__main__":
    import json
    
    # Evento de prueba
    test_event = {
        "httpMethod": "GET",
        "path": "/api/health",
        "headers": {},
        "queryStringParameters": None,
        "body": None
    }
    
    class TestContext:
        aws_request_id = "test-request-id"
        function_name = "test-function"
    
    response = handler(test_event, TestContext())
    print(json.dumps(response, indent=2))




