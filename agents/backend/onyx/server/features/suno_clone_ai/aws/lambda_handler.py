"""
AWS Lambda Handler para Suno Clone AI
Optimizado para serverless deployment con cold start minimization
"""

import json
import logging
from typing import Dict, Any, Optional
from mangum import Mangum
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from main import app as fastapi_app
from config.settings import settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Pre-inicializar la aplicación para reducir cold starts
_handler: Optional[Mangum] = None


def get_handler() -> Mangum:
    """Obtiene o crea el handler Lambda (singleton pattern)"""
    global _handler
    if _handler is None:
        # Configurar Mangum con optimizaciones
        _handler = Mangum(
            fastapi_app,
            lifespan="off",  # Desactivar lifespan en Lambda (se maneja diferente)
            api_gateway_base_path="/",  # Ajustar según tu API Gateway
            text_mime_types=["application/json", "text/plain"]
        )
    return _handler


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Handler principal para AWS Lambda
    
    Args:
        event: Evento de Lambda (API Gateway, SQS, etc.)
        context: Contexto de Lambda
        
    Returns:
        Respuesta formateada para API Gateway
    """
    try:
        # Log del evento (útil para debugging)
        logger.info(f"Lambda event: {json.dumps(event, default=str)}")
        
        # Obtener handler
        handler = get_handler()
        
        # Procesar request
        response = handler(event, context)
        
        # Asegurar formato correcto para API Gateway
        if isinstance(response, dict):
            return response
        elif hasattr(response, 'body'):
            return {
                "statusCode": response.get("statusCode", 200),
                "headers": response.get("headers", {}),
                "body": response.get("body", "")
            }
        else:
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(response) if not isinstance(response, str) else response
            }
            
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "error": "Internal server error",
                "message": str(e) if settings.debug else "An error occurred"
            })
        }


# Handler para eventos SQS
def sqs_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handler para procesar mensajes de SQS"""
    from services.task_queue import process_sqs_message
    
    results = []
    for record in event.get("Records", []):
        try:
            body = json.loads(record["body"])
            result = process_sqs_message(body)
            results.append({"status": "success", "result": result})
        except Exception as e:
            logger.error(f"SQS processing error: {str(e)}", exc_info=True)
            results.append({"status": "error", "error": str(e)})
    
    return {"processed": len(results), "results": results}


# Handler para eventos S3
def s3_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handler para procesar eventos de S3 (uploads, etc.)"""
    from services.audio_streaming import process_s3_event
    
    results = []
    for record in event.get("Records", []):
        try:
            s3_info = record["s3"]
            bucket = s3_info["bucket"]["name"]
            key = s3_info["object"]["key"]
            result = process_s3_event(bucket, key)
            results.append({"status": "success", "result": result})
        except Exception as e:
            logger.error(f"S3 processing error: {str(e)}", exc_info=True)
            results.append({"status": "error", "error": str(e)})
    
    return {"processed": len(results), "results": results}















