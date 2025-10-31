"""
API Gateway - Gateway de API
===========================

API Gateway que actúa como punto de entrada único para todos los microservicios
con funcionalidades avanzadas de routing, load balancing, rate limiting,
autenticación y autorización.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import time
import uuid

from .service_discovery import ServiceDiscovery
from .load_balancer import LoadBalancer
from .rate_limiter import RateLimiter
from .authentication import AuthenticationService
from .authorization import AuthorizationService
from .request_router import RequestRouter
from .response_aggregator import ResponseAggregator
from ..monitoring.metrics import MetricsCollector
from ..monitoring.tracing import DistributedTracer
from ..resilience.circuit_breaker import CircuitBreaker


class APIGateway:
    """
    API Gateway principal del sistema.
    
    Funcionalidades:
    - Service discovery y routing
    - Load balancing
    - Rate limiting
    - Authentication y authorization
    - Request/response transformation
    - Circuit breaking
    - Metrics y tracing
    - Response aggregation
    """
    
    def __init__(
        self,
        service_discovery: ServiceDiscovery,
        load_balancer: LoadBalancer,
        rate_limiter: RateLimiter,
        auth_service: AuthenticationService,
        authz_service: AuthorizationService,
        request_router: RequestRouter,
        response_aggregator: ResponseAggregator,
        metrics_collector: MetricsCollector,
        tracer: DistributedTracer,
        circuit_breaker: CircuitBreaker
    ):
        self.service_discovery = service_discovery
        self.load_balancer = load_balancer
        self.rate_limiter = rate_limiter
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.request_router = request_router
        self.response_aggregator = response_aggregator
        self.metrics_collector = metrics_collector
        self.tracer = tracer
        self.circuit_breaker = circuit_breaker
        
        # Configurar FastAPI
        self.app = FastAPI(
            title="AI History Comparison API Gateway",
            description="API Gateway para el sistema de comparación de historial de IA",
            version="3.0.0"
        )
        
        # Configurar middleware
        self._setup_middleware()
        
        # Configurar rutas
        self._setup_routes()
        
        # Cliente HTTP
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    def _setup_middleware(self):
        """Configurar middleware del gateway."""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Middleware personalizado
        @self.app.middleware("http")
        async def gateway_middleware(request: Request, call_next):
            """Middleware principal del gateway."""
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Agregar request ID
            request.state.request_id = request_id
            
            # Iniciar tracing
            with self.tracer.start_span("gateway_request", request_id=request_id):
                try:
                    # Rate limiting
                    client_ip = request.client.host if request.client else "unknown"
                    if not await self.rate_limiter.is_allowed(client_ip):
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "request_id": request_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    
                    # Authentication
                    auth_result = await self.auth_service.authenticate(request)
                    if not auth_result.success:
                        return JSONResponse(
                            status_code=401,
                            content={
                                "error": "Authentication failed",
                                "request_id": request_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    
                    # Authorization
                    authz_result = await self.authz_service.authorize(
                        request, auth_result.user
                    )
                    if not authz_result.success:
                        return JSONResponse(
                            status_code=403,
                            content={
                                "error": "Authorization failed",
                                "request_id": request_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    
                    # Procesar request
                    response = await call_next(request)
                    
                    # Métricas
                    process_time = time.time() - start_time
                    self.metrics_collector.record_histogram(
                        "gateway_request_duration", process_time
                    )
                    self.metrics_collector.increment_counter(
                        "gateway_requests_total",
                        tags={"status": str(response.status_code)}
                    )
                    
                    # Agregar headers de respuesta
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Process-Time"] = str(process_time)
                    
                    return response
                    
                except Exception as e:
                    # Error handling
                    process_time = time.time() - start_time
                    self.metrics_collector.increment_counter(
                        "gateway_errors_total",
                        tags={"error_type": type(e).__name__}
                    )
                    
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "Internal gateway error",
                            "request_id": request_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
    
    def _setup_routes(self):
        """Configurar rutas del gateway."""
        
        @self.app.get("/")
        async def root():
            """Endpoint raíz del gateway."""
            return {
                "message": "AI History Comparison API Gateway",
                "version": "3.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "services": await self.service_discovery.get_available_services()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check del gateway."""
            services_health = await self._check_services_health()
            
            return {
                "status": "healthy",
                "gateway": "operational",
                "services": services_health,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Obtener métricas del gateway."""
            return await self.metrics_collector.get_metrics()
        
        # Rutas de microservicios
        @self.app.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_request(service_name: str, path: str, request: Request):
            """Proxy requests a microservicios."""
            return await self._handle_proxy_request(service_name, path, request)
        
        # Rutas agregadas
        @self.app.get("/api/v1/history")
        async def get_history_aggregated(request: Request):
            """Obtener historial con datos agregados."""
            return await self._handle_aggregated_request("history", request)
        
        @self.app.get("/api/v1/analytics/dashboard")
        async def get_analytics_dashboard(request: Request):
            """Obtener dashboard de analytics."""
            return await self._handle_aggregated_request("analytics_dashboard", request)
    
    async def _handle_proxy_request(
        self, 
        service_name: str, 
        path: str, 
        request: Request
    ) -> Response:
        """Manejar request proxy a microservicio."""
        try:
            # Obtener servicio
            service = await self.service_discovery.get_service(service_name)
            if not service:
                raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
            # Load balancing
            service_instance = await self.load_balancer.get_instance(service)
            if not service_instance:
                raise HTTPException(status_code=503, detail="No available service instances")
            
            # Circuit breaker
            if not await self.circuit_breaker.is_available(service_name):
                raise HTTPException(status_code=503, detail="Service temporarily unavailable")
            
            # Construir URL
            target_url = f"{service_instance.url}/{path}"
            
            # Preparar request
            headers = dict(request.headers)
            headers["X-Request-ID"] = request.state.request_id
            headers["X-User-ID"] = getattr(request.state, "user_id", "anonymous")
            
            # Procesar request
            async with self.circuit_breaker.execute(service_name):
                response = await self.http_client.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.query_params,
                    content=await request.body()
                )
            
            # Transformar response
            transformed_response = await self._transform_response(response, service_name)
            
            return JSONResponse(
                content=transformed_response,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Service timeout")
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Service unavailable")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _handle_aggregated_request(
        self, 
        aggregation_type: str, 
        request: Request
    ) -> Dict[str, Any]:
        """Manejar requests agregados."""
        try:
            if aggregation_type == "history":
                return await self._aggregate_history_data(request)
            elif aggregation_type == "analytics_dashboard":
                return await self._aggregate_analytics_dashboard(request)
            else:
                raise HTTPException(status_code=404, detail="Unknown aggregation type")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _aggregate_history_data(self, request: Request) -> Dict[str, Any]:
        """Agregar datos de historial de múltiples servicios."""
        try:
            # Obtener datos de historial
            history_service = await self.service_discovery.get_service("history")
            history_instance = await self.load_balancer.get_instance(history_service)
            
            # Obtener datos de calidad
            quality_service = await self.service_discovery.get_service("quality")
            quality_instance = await self.load_balancer.get_instance(quality_service)
            
            # Hacer requests paralelos
            tasks = []
            
            # Request a servicio de historial
            history_task = self.http_client.get(
                f"{history_instance.url}/history",
                params=request.query_params
            )
            tasks.append(("history", history_task))
            
            # Request a servicio de calidad
            quality_task = self.http_client.get(
                f"{quality_instance.url}/quality/reports",
                params=request.query_params
            )
            tasks.append(("quality", quality_task))
            
            # Ejecutar requests
            results = {}
            for service_name, task in tasks:
                try:
                    response = await task
                    results[service_name] = response.json()
                except Exception as e:
                    results[service_name] = {"error": str(e)}
            
            # Agregar resultados
            aggregated_data = await self.response_aggregator.aggregate_history_data(results)
            
            return aggregated_data
            
        except Exception as e:
            raise Exception(f"Failed to aggregate history data: {e}")
    
    async def _aggregate_analytics_dashboard(self, request: Request) -> Dict[str, Any]:
        """Agregar datos para dashboard de analytics."""
        try:
            # Obtener servicios
            services = ["history", "comparison", "quality", "analytics"]
            service_instances = {}
            
            for service_name in services:
                service = await self.service_discovery.get_service(service_name)
                if service:
                    instance = await self.load_balancer.get_instance(service)
                    if instance:
                        service_instances[service_name] = instance
            
            # Hacer requests paralelos
            tasks = []
            for service_name, instance in service_instances.items():
                task = self.http_client.get(f"{instance.url}/metrics")
                tasks.append((service_name, task))
            
            # Ejecutar requests
            results = {}
            for service_name, task in tasks:
                try:
                    response = await task
                    results[service_name] = response.json()
                except Exception as e:
                    results[service_name] = {"error": str(e)}
            
            # Agregar resultados
            dashboard_data = await self.response_aggregator.aggregate_dashboard_data(results)
            
            return dashboard_data
            
        except Exception as e:
            raise Exception(f"Failed to aggregate dashboard data: {e}")
    
    async def _transform_response(
        self, 
        response: httpx.Response, 
        service_name: str
    ) -> Dict[str, Any]:
        """Transformar response del microservicio."""
        try:
            data = response.json()
            
            # Agregar metadatos del gateway
            transformed_data = {
                "data": data,
                "metadata": {
                    "service": service_name,
                    "gateway_version": "3.0.0",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return transformed_data
            
        except Exception:
            # Si no es JSON, devolver como está
            return {"data": response.text}
    
    async def _check_services_health(self) -> Dict[str, str]:
        """Verificar salud de los servicios."""
        services_health = {}
        
        try:
            available_services = await self.service_discovery.get_available_services()
            
            for service_name in available_services:
                try:
                    service = await self.service_discovery.get_service(service_name)
                    if service:
                        instance = await self.load_balancer.get_instance(service)
                        if instance:
                            # Hacer health check
                            health_response = await self.http_client.get(
                                f"{instance.url}/health",
                                timeout=5.0
                            )
                            if health_response.status_code == 200:
                                services_health[service_name] = "healthy"
                            else:
                                services_health[service_name] = "unhealthy"
                        else:
                            services_health[service_name] = "no_instances"
                    else:
                        services_health[service_name] = "not_found"
                        
                except Exception:
                    services_health[service_name] = "unreachable"
                    
        except Exception as e:
            services_health["error"] = str(e)
        
        return services_health
    
    async def start(self):
        """Iniciar el gateway."""
        await self.service_discovery.start()
        await self.load_balancer.start()
        await self.rate_limiter.start()
        await self.auth_service.start()
        await self.authz_service.start()
        await self.metrics_collector.start()
        await self.tracer.start()
    
    async def stop(self):
        """Detener el gateway."""
        await self.http_client.aclose()
        await self.service_discovery.stop()
        await self.load_balancer.stop()
        await self.rate_limiter.stop()
        await self.auth_service.stop()
        await self.authz_service.stop()
        await self.metrics_collector.stop()
        await self.tracer.stop()




