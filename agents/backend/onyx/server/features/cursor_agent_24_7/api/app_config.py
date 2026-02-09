"""
App Configuration - Configuración de la aplicación FastAPI
==========================================================

Módulo para configurar la aplicación FastAPI con middleware,
CORS, y otros componentes.
"""

import logging
import os
from typing import Optional
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from ..core.agent import CursorAgent
from ..core.error_handling import safe_import, error_context

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """
    Configurar middleware de la aplicación.
    
    Args:
        app: Aplicación FastAPI.
    """
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # OpenTelemetry instrumentation
    def setup_opentelemetry_instrumentation():
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry instrumentation enabled")
    
    safe_import(
        "opentelemetry.instrumentation.fastapi",
        setup_opentelemetry_instrumentation,
        logger_instance=logger
    )
    
    # Middleware adicional opcional
    def setup_additional_middleware():
        from ..core.middleware import (
            LoggingMiddleware,
            SecurityHeadersMiddleware,
            ErrorHandlingMiddleware
        )
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(SecurityHeadersMiddleware)
        app.add_middleware(ErrorHandlingMiddleware)
        logger.debug("Additional middleware loaded")
    
    safe_import(
        "..core.middleware",
        setup_additional_middleware,
        logger_instance=logger
    )


def setup_websocket_handlers(app: FastAPI) -> None:
    """
    Configurar handlers de WebSocket.
    
    Args:
        app: Aplicación FastAPI.
    """
    from fastapi import WebSocket
    from ..core.websocket_handler import WebSocketManager
    
    ws_manager = WebSocketManager()
    app.state.ws_manager = ws_manager
    
    # Handler para comandos desde WebSocket
    async def handle_command_message(websocket: WebSocket, message: dict):
        """Manejar mensaje de comando desde WebSocket"""
        command = message.get("command")
        if command:
            agent = app.state.agent
            task_id = await agent.add_task(command)
            await ws_manager.send_personal_message({
                "type": "task_added",
                "task_id": task_id,
                "command": command[:50]
            }, websocket)
    
    ws_manager.register_handler("command", handle_command_message)
    logger.debug("WebSocket handlers configured")


def create_app(agent: Optional[CursorAgent] = None, use_aws: Optional[bool] = None) -> FastAPI:
    """
    Crear aplicación FastAPI configurada.
    
    Args:
        agent: Instancia del agente. Si es None, se crea una nueva.
        use_aws: Forzar uso de AWS (None = auto-detect).
    
    Returns:
        Aplicación FastAPI configurada.
    """
    import os
    from ..core.agent import AgentConfig
    from .routes import agent_router, task_router, notification_router
    from fastapi import WebSocket, Request
    
    # Auto-detectar AWS si no se especifica
    if use_aws is None:
        use_aws = bool(os.getenv("AWS_REGION") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    
    # Crear aplicación
    app = FastAPI(
        title="Cursor Agent 24/7 API",
        description="API para controlar el agente persistente de Cursor",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        openapi_tags=[
            {"name": "agent", "description": "Operaciones del agente"},
            {"name": "tasks", "description": "Gestión de tareas"},
            {"name": "notifications", "description": "Notificaciones"},
            {"name": "search", "description": "Búsqueda avanzada"},
            {"name": "bulk", "description": "Operaciones en lote"},
            {"name": "webhooks", "description": "Webhooks"},
            {"name": "perplexity", "description": "Perplexity-style query processing"},
        ]
    )
    
    # Configurar middleware
    setup_middleware(app)
    
    # Agregar compresión
    def setup_compression():
        from ..core.compression import CompressionMiddleware
        app.add_middleware(CompressionMiddleware)
        logger.info("Compression middleware enabled")
    
    safe_import(
        "..core.compression",
        setup_compression,
        logger_instance=logger
    )
    
    # Configurar agente
    if agent is None:
        config = AgentConfig()
        
        # Si estamos en AWS, usar adaptadores AWS
        if use_aws:
            try:
                from ..core.aws_adapter import AWSStateManager, AWSCacheAdapter
                
                # Crear agente con configuración AWS
                agent = CursorAgent(config)
                
                # Reemplazar state manager con AWS adapter
                dynamodb_table = os.getenv("DYNAMODB_TABLE_NAME", "cursor-agent-state")
                if dynamodb_table:
                    try:
                        agent.state_manager = AWSStateManager(
                            agent=agent,
                            table_name=dynamodb_table,
                            region=os.getenv("AWS_REGION", "us-east-1")
                        )
                        logger.info(f"Using DynamoDB for state: {dynamodb_table}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize DynamoDB state manager: {e}")
                
                # Reemplazar cache con AWS adapter
                cache_type = os.getenv("CACHE_TYPE", "elasticache")
                if cache_type in ("elasticache", "dynamodb"):
                    try:
                        agent.cache = AWSCacheAdapter(
                            cache_type=cache_type,
                            endpoint=os.getenv("REDIS_ENDPOINT"),
                            table_name=os.getenv("DYNAMODB_CACHE_TABLE", "cursor-agent-cache"),
                            region=os.getenv("AWS_REGION", "us-east-1")
                        )
                        logger.info(f"Using {cache_type} for cache")
                    except Exception as e:
                        logger.warning(f"Failed to initialize AWS cache adapter: {e}")
                
            except ImportError as e:
                logger.warning(f"AWS adapters not available: {e}. Using default storage.")
                agent = CursorAgent(config)
        else:
            agent = CursorAgent(config)
    
    app.state.agent = agent
    
    # Configurar WebSocket
    setup_websocket_handlers(app)
    
    # Registrar routers
    app.include_router(agent_router)
    app.include_router(task_router)
    app.include_router(notification_router)
    
    # Router de búsqueda (opcional, requiere Elasticsearch)
    def setup_search_routes():
        from .routes.search_routes import router as search_router
        app.include_router(search_router)
        logger.info("Search routes configured")
    
    safe_import(
        ".routes.search_routes",
        setup_search_routes,
        logger_instance=logger
    )
    
    # Router de operaciones en lote
    def setup_bulk_routes():
        from .routes.bulk_routes import router as bulk_router
        app.include_router(bulk_router)
        logger.info("Bulk routes configured")
    
    safe_import(
        ".routes.bulk_routes",
        setup_bulk_routes,
        logger_instance=logger
    )
    
    # Router de webhooks
    def setup_webhook_routes():
        from .routes.webhook_routes import router as webhook_router
        app.include_router(webhook_router)
        logger.info("Webhook routes configured")
    
    safe_import(
        ".routes.webhook_routes",
        setup_webhook_routes,
        logger_instance=logger
    )
    
    # Router de Perplexity (query processing)
    def setup_perplexity_routes():
        from .routes.perplexity_routes import router as perplexity_router
        app.include_router(perplexity_router)
        logger.info("Perplexity routes configured")
    
    safe_import(
        ".routes.perplexity_routes",
        setup_perplexity_routes,
        logger_instance=logger
    )
    
    # Configurar graceful shutdown
    def setup_graceful_shutdown():
        from ..core.graceful_shutdown import get_shutdown_manager
        shutdown_manager = get_shutdown_manager()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            await shutdown_manager.shutdown()
        
        # Registrar handlers de shutdown
        async def close_webhook_manager():
            from ..core.webhooks import get_webhook_manager
            manager = get_webhook_manager()
            await manager.close()
        
        shutdown_manager.register(close_webhook_manager)
        logger.info("Graceful shutdown configured")
    
    safe_import(
        "..core.graceful_shutdown",
        setup_graceful_shutdown,
        logger_instance=logger
    )
    
    # Endpoint WebSocket
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """Endpoint WebSocket para comunicación en tiempo real"""
        await app.state.ws_manager.listen(websocket)
    
    # Health check mejorado
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint con información de AWS"""
        health_data = {
            "status": "healthy",
            "agent_status": agent.status.value,
            "aws": {
                "enabled": use_aws,
                "region": os.getenv("AWS_REGION"),
                "lambda": bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
            }
        }
        return health_data
    
    # Prometheus metrics endpoint
    @app.get("/metrics")
    async def metrics_endpoint():
        """Endpoint de métricas Prometheus"""
        try:
            from ..core.observability import get_prometheus_metrics
            
            metrics = get_prometheus_metrics()
            if metrics:
                from fastapi.responses import Response
                return Response(
                    content=metrics['generate_latest'](),
                    media_type=metrics['CONTENT_TYPE_LATEST']
                )
            else:
                return {"message": "Prometheus metrics not available"}
        except Exception as e:
            logger.warning(f"Error generating metrics: {e}")
            return {"error": "Metrics unavailable"}
    
    # OAuth2 routes
    try:
        from ..core.oauth2 import (
            authenticate_user,
            create_access_token,
            get_current_active_user,
            OAuth2PasswordRequestForm
        )
        from datetime import timedelta
        
        @app.post("/api/auth/token")
        async def login(form_data: OAuth2PasswordRequestForm = Depends()):
            """Endpoint de autenticación OAuth2"""
            user = authenticate_user(form_data.username, form_data.password)
            if not user:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            access_token_expires = timedelta(minutes=30)
            access_token = create_access_token(
                data={"sub": user.username, "roles": user.roles},
                expires_delta=access_token_expires
            )
            return {"access_token": access_token, "token_type": "bearer"}
        
        @app.get("/api/auth/me")
        async def read_users_me(current_user = Depends(get_current_active_user)):
            """Obtener información del usuario actual"""
            return {
                "username": current_user.username,
                "email": current_user.email,
                "roles": current_user.roles
            }
        
        logger.info("OAuth2 routes configured")
    except ImportError:
        logger.debug("OAuth2 not available")
    except Exception as e:
        logger.warning(f"Failed to setup OAuth2: {e}")
    
    # Inicializar observability
    try:
        from ..core.observability import setup_prometheus, setup_opentelemetry
        
        setup_prometheus()
        setup_opentelemetry(
            service_name="cursor-agent-24-7",
            service_version="1.0.0"
        )
        logger.info("Observability configured")
    except Exception as e:
        logger.warning(f"Failed to setup observability: {e}")
    
    # Configurar API Gateway middleware
    try:
        from ..core.api_gateway import setup_api_gateway_middleware
        setup_api_gateway_middleware(app)
        logger.info("API Gateway middleware configured")
    except Exception as e:
        logger.debug(f"API Gateway middleware not configured: {e}")
    
    # Endpoint para Celery task status
    try:
        from ..core.celery_worker import get_task_status
        
        @app.get("/api/tasks/{task_id}/status")
        async def get_celery_task_status(task_id: str):
            """Obtener estado de tarea Celery"""
            status = get_task_status(task_id)
            return status
        
        logger.info("Celery integration configured")
    except Exception as e:
        logger.debug(f"Celery integration not available: {e}")
    
    logger.info(f"FastAPI application created and configured (AWS: {use_aws})")
    return app

