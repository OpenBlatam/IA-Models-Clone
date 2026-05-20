"""
AWS API Gateway Integration
===========================

Integración con AWS API Gateway para:
- Serverless deployment
- Request/Response transformation
- Authentication/Authorization
- Rate limiting
- API versioning
"""

import logging
import json
from typing import Dict, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AWSAPIGatewayConfig:
    """Configuración para AWS API Gateway"""
    
    def __init__(self, region: str = "us-east-1", stage: str = "prod"):
        self.region = region
        self.stage = stage
        self.api_id: Optional[str] = None
        self.resources: Dict[str, Dict] = {}
    
    def generate_swagger_spec(self, base_path: str = "/") -> Dict:
        """Genera especificación Swagger/OpenAPI para AWS API Gateway"""
        return {
            "swagger": "2.0",
            "info": {
                "title": "3D Prototype AI API",
                "version": "2.0.0"
            },
            "host": f"{{api_id}}.execute-api.{self.region}.amazonaws.com",
            "basePath": base_path,
            "schemes": ["https"],
            "paths": self._generate_paths(),
            "definitions": self._generate_definitions(),
            "securityDefinitions": {
                "JWT": {
                    "type": "apiKey",
                    "name": "Authorization",
                    "in": "header",
                    "x-amazon-apigateway-authtype": "oauth2"
                }
            }
        }
    
    def _generate_paths(self) -> Dict:
        """Genera paths para la API"""
        return {
            "/api/v1/generate": {
                "post": {
                    "summary": "Generate prototype",
                    "operationId": "generatePrototype",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {"$ref": "#/definitions/PrototypeRequest"}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Success",
                            "schema": {"$ref": "#/definitions/PrototypeResponse"}
                        }
                    },
                    "x-amazon-apigateway-integration": {
                        "type": "http_proxy",
                        "httpMethod": "POST",
                        "uri": "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations",
                        "responses": {
                            "default": {
                                "statusCode": "200"
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Healthy"
                        }
                    },
                    "x-amazon-apigateway-integration": {
                        "type": "http_proxy",
                        "httpMethod": "GET",
                        "uri": "arn:aws:apigateway:us-east-1:lambda:path/2015-03-31/functions/{lambda_arn}/invocations",
                        "responses": {
                            "default": {
                                "statusCode": "200"
                            }
                        }
                    }
                }
            }
        }
    
    def _generate_definitions(self) -> Dict:
        """Genera definiciones de modelos"""
        return {
            "PrototypeRequest": {
                "type": "object",
                "properties": {
                    "product_description": {"type": "string"},
                    "product_type": {"type": "string"},
                    "budget": {"type": "number"}
                }
            },
            "PrototypeResponse": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "materials": {"type": "array"},
                    "cad_parts": {"type": "array"}
                }
            }
        }
    
    def generate_lambda_handler_template(self) -> str:
        """Genera template para Lambda handler"""
        return '''
import json
from mangum import Mangum
from api.prototype_api import app

# Wrap FastAPI app with Mangum for Lambda
handler = Mangum(app, lifespan="off")

def lambda_handler(event, context):
    """Lambda handler para AWS API Gateway"""
    return handler(event, context)
'''


class AWSAPIGatewayManager:
    """Gestor de AWS API Gateway"""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.config = AWSAPIGatewayConfig(region=region)
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Configura cliente de boto3"""
        try:
            import boto3
            self.client = boto3.client('apigateway', region_name=self.region)
            logger.info("AWS API Gateway client configured")
        except ImportError:
            logger.warning("boto3 not available. Install with: pip install boto3")
        except Exception as e:
            logger.error(f"Failed to setup AWS API Gateway client: {e}")
    
    def create_rest_api(self, name: str, description: str = "") -> Optional[str]:
        """Crea un REST API en AWS API Gateway"""
        if not self.client:
            return None
        
        try:
            response = self.client.create_rest_api(
                name=name,
                description=description,
                endpointConfiguration={
                    'types': ['REGIONAL']
                }
            )
            api_id = response['id']
            self.config.api_id = api_id
            logger.info(f"REST API created: {api_id}")
            return api_id
        except Exception as e:
            logger.error(f"Failed to create REST API: {e}")
            return None
    
    def deploy_api(self, api_id: str, stage_name: str = "prod") -> bool:
        """Despliega la API a un stage"""
        if not self.client:
            return False
        
        try:
            self.client.create_deployment(
                restApiId=api_id,
                stageName=stage_name
            )
            logger.info(f"API deployed to stage: {stage_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to deploy API: {e}")
            return False
    
    def setup_usage_plan(self, api_id: str, 
                        throttle_rate: float = 100.0,
                        throttle_burst: int = 200) -> Optional[str]:
        """Configura un usage plan con rate limiting"""
        if not self.client:
            return None
        
        try:
            response = self.client.create_usage_plan(
                name="3d-prototype-ai-plan",
                description="Usage plan for 3D Prototype AI",
                apiStages=[{
                    'apiId': api_id,
                    'stage': self.config.stage
                }],
                throttle={
                    'rateLimit': throttle_rate,
                    'burstLimit': throttle_burst
                }
            )
            plan_id = response['id']
            logger.info(f"Usage plan created: {plan_id}")
            return plan_id
        except Exception as e:
            logger.error(f"Failed to create usage plan: {e}")
            return None
    
    def generate_swagger_export(self) -> Dict:
        """Genera exportación Swagger para importar en AWS API Gateway"""
        return self.config.generate_swagger_spec()


# Instancia global
aws_api_gateway_manager = AWSAPIGatewayManager()




