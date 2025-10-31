from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple, Type, TypeVar
from enum import Enum
import threading
from contextlib import contextmanager
from collections import defaultdict
import argparse
import click
import typer
from datetime import datetime, timedelta
import uuid
import hashlib
import base64
    import fastapi
    from fastapi import FastAPI, HTTPException, Depends, Request, Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel, Field, validator
    import uvicorn
    import redis
from typing import Any, List, Dict, Optional
"""
RORO Pattern Tool Control - CLI and RESTful API Interfaces
==========================================================

This module provides comprehensive CLI and RESTful API interfaces using the
RORO (Receive Object, Return Object) pattern for tool control.

Features:
- RORO pattern implementation for consistent data flow
- CLI interface with command parsing and validation
- RESTful API with FastAPI integration
- Tool registry and management system
- Request/response validation and sanitization
- Authentication and authorization
- Rate limiting and throttling
- Error handling and logging
- Performance monitoring and metrics
- Documentation and OpenAPI generation

Author: AI Assistant
License: MIT
"""


try:
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class ToolStatus(Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ToolPriority(Enum):
    """Tool execution priority."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AuthLevel(Enum):
    """Authentication levels."""
    NONE = "none"
    BASIC = "basic"
    TOKEN = "token"
    JWT = "jwt"
    OAUTH = "oauth"


@dataclass
class ToolRequest:
    """RORO pattern request object."""
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: ToolPriority = ToolPriority.NORMAL
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolResponse:
    """RORO pattern response object."""
    request_id: str
    tool_name: str
    status: ToolStatus
    result: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ToolDefinition:
    """Tool definition and metadata."""
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_parameters: List[str] = field(default_factory=list)
    optional_parameters: List[str] = field(default_factory=list)
    return_type: str = "any"
    auth_required: AuthLevel = AuthLevel.NONE
    rate_limit: Optional[float] = None
    timeout: Optional[float] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolExecution:
    """Tool execution context."""
    request: ToolRequest
    definition: ToolDefinition
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: ToolStatus = ToolStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    cors_origins: List[str] = field(default_factory=list)
    trusted_hosts: List[str] = field(default_factory=list)
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    auth_enabled: bool = True
    auth_tokens: Dict[str, str] = field(default_factory=dict)
    log_requests: bool = True
    metrics_enabled: bool = True


@dataclass
class CLIConfig:
    """CLI configuration."""
    verbose: bool = False
    quiet: bool = False
    output_format: str = "json"
    config_file: Optional[str] = None
    timeout: Optional[float] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0


class ToolRegistryError(Exception):
    """Custom exception for tool registry errors."""
    pass


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""
    pass


class AuthenticationError(Exception):
    """Custom exception for authentication errors."""
    pass


class RateLimitError(Exception):
    """Custom exception for rate limiting errors."""
    pass


class ToolRegistry:
    """Tool registry and management system."""
    
    def __init__(self) -> Any:
        """Initialize tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.executions: Dict[str, ToolExecution] = {}
        self.handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        self._metrics = defaultdict(int)
    
    def register_tool(self, definition: ToolDefinition, handler: Callable) -> bool:
        """Register a tool with its handler."""
        with self._lock:
            if definition.name in self.tools:
                logger.warning(f"Tool {definition.name} already registered, overwriting")
            
            self.tools[definition.name] = definition
            self.handlers[definition.name] = handler
            
            logger.info(f"Registered tool: {definition.name} v{definition.version}")
            return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        with self._lock:
            if tool_name not in self.tools:
                return False
            
            del self.tools[tool_name]
            if tool_name in self.handlers:
                del self.handlers[tool_name]
            
            logger.info(f"Unregistered tool: {tool_name}")
            return True
    
    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition."""
        with self._lock:
            return self.tools.get(tool_name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all registered tools."""
        with self._lock:
            return list(self.tools.values())
    
    def get_tool_categories(self) -> Dict[str, List[str]]:
        """Get tools grouped by category."""
        with self._lock:
            categories = defaultdict(list)
            for tool in self.tools.values():
                categories[tool.category].append(tool.name)
            return dict(categories)
    
    async def validate_request(self, request: ToolRequest) -> List[str]:
        """Validate tool request."""
        errors = []
        
        # Check if tool exists
        tool = self.get_tool(request.tool_name)
        if not tool:
            errors.append(f"Tool '{request.tool_name}' not found")
            return errors
        
        # Check required parameters
        for param in tool.required_parameters:
            if param not in request.parameters:
                errors.append(f"Required parameter '{param}' missing")
        
        # Check parameter types
        for param_name, param_value in request.parameters.items():
            if param_name in tool.parameters:
                param_def = tool.parameters[param_name]
                if not self._validate_parameter(param_value, param_def):
                    errors.append(f"Parameter '{param_name}' has invalid type or value")
        
        return errors
    
    def _validate_parameter(self, value: Any, definition: Dict[str, Any]) -> bool:
        """Validate parameter value against definition."""
        param_type = definition.get('type', 'any')
        
        if param_type == 'string':
            return isinstance(value, str)
        elif param_type == 'integer':
            return isinstance(value, int)
        elif param_type == 'float':
            return isinstance(value, (int, float))
        elif param_type == 'boolean':
            return isinstance(value, bool)
        elif param_type == 'list':
            return isinstance(value, list)
        elif param_type == 'dict':
            return isinstance(value, dict)
        else:
            return True  # Accept any type for 'any'
    
    async def execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool using RORO pattern."""
        # Validate request
        errors = self.validate_request(request)
        if errors:
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error_message="; ".join(errors),
                error_code="VALIDATION_ERROR"
            )
        
        # Get tool definition and handler
        tool = self.get_tool(request.tool_name)
        handler = self.handlers.get(request.tool_name)
        
        if not tool or not handler:
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error_message=f"Tool '{request.tool_name}' not found or handler missing",
                error_code="TOOL_NOT_FOUND"
            )
        
        # Create execution context
        execution = ToolExecution(request=request, definition=tool)
        self.executions[request.request_id] = execution
        
        # Update metrics
        self._metrics[f"tool_{request.tool_name}_executions"] += 1
        
        try:
            # Execute tool
            execution.status = ToolStatus.RUNNING
            start_time = time.time()
            
            if asyncio.iscoroutinefunction(handler):
                result = await handler(request)
            else:
                result = handler(request)
            
            execution_time = time.time() - start_time
            
            # Update execution
            execution.end_time = datetime.now()
            execution.status = ToolStatus.COMPLETED
            execution.result = result
            
            # Create response
            response = ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                metadata={'tool_version': tool.version}
            )
            
            logger.info(f"Tool {request.tool_name} executed successfully in {execution_time:.2f}s")
            return response
        
        except Exception as e:
            execution_time = time.time() - start_time
            execution.end_time = datetime.now()
            execution.status = ToolStatus.FAILED
            execution.error = str(e)
            
            logger.error(f"Tool {request.tool_name} execution failed: {e}")
            
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error_message=str(e),
                error_code="EXECUTION_ERROR",
                execution_time=execution_time
            )
    
    def get_execution(self, request_id: str) -> Optional[ToolExecution]:
        """Get execution by request ID."""
        with self._lock:
            return self.executions.get(request_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get registry metrics."""
        with self._lock:
            return {
                'total_tools': len(self.tools),
                'total_executions': len(self.executions),
                'tool_metrics': dict(self._metrics),
                'categories': self.get_tool_categories()
            }


class RateLimiter:
    """Rate limiting for API requests."""
    
    def __init__(self, config: APIConfig):
        """Initialize rate limiter."""
        self.config = config
        self.request_counts: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        if REDIS_AVAILABLE and hasattr(config, 'redis_url'):
            self.redis_client = redis.from_url(config.redis_url)
        else:
            self.redis_client = None
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        if not self.config.rate_limit_enabled:
            return True
        
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        with self._lock:
            # Clean old requests
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self.request_counts[identifier]) >= self.config.rate_limit_requests:
                return False
            
            # Add current request
            self.request_counts[identifier].append(now)
            return True


class Authenticator:
    """Authentication and authorization."""
    
    def __init__(self, config: APIConfig):
        """Initialize authenticator."""
        self.config = config
        self.tokens = config.auth_tokens
        self._lock = threading.Lock()
    
    def authenticate(self, credentials: Optional[str]) -> Optional[str]:
        """Authenticate request and return user ID."""
        if not self.config.auth_enabled:
            return "anonymous"
        
        if not credentials:
            return None
        
        # Extract token from Bearer format
        if credentials.startswith("Bearer "):
            token = credentials[7:]
        else:
            token = credentials
        
        with self._lock:
            for user_id, user_token in self.tokens.items():
                if user_token == token:
                    return user_id
        
        return None
    
    def authorize(self, user_id: str, tool_name: str) -> bool:
        """Check if user is authorized to use tool."""
        # Simple authorization - can be extended with roles and permissions
        return user_id is not None


class CLIToolController:
    """CLI interface for tool control."""
    
    def __init__(self, registry: ToolRegistry, config: CLIConfig):
        """Initialize CLI controller."""
        self.registry = registry
        self.config = config
        self.app = typer.Typer()
        self._setup_commands()
    
    def _setup_commands(self) -> Any:
        """Setup CLI commands."""
        
        @self.app.command()
        def list_tools(
            category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
            verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
        ):
            """List available tools."""
            tools = self.registry.list_tools()
            
            if category:
                tools = [t for t in tools if t.category == category]
            
            if self.config.output_format == "json":
                output = json.dumps([asdict(tool) for tool in tools], indent=2)
                typer.echo(output)
            else:
                for tool in tools:
                    typer.echo(f"{tool.name} v{tool.version} - {tool.description}")
                    if verbose:
                        typer.echo(f"  Category: {tool.category}")
                        typer.echo(f"  Tags: {', '.join(tool.tags)}")
                        typer.echo(f"  Required params: {', '.join(tool.required_parameters)}")
                        typer.echo()
        
        @self.app.command()
        def execute(
            tool_name: str = typer.Argument(..., help="Name of the tool to execute"),
            parameters: str = typer.Option("{}", "--params", "-p", help="Tool parameters as JSON"),
            timeout: Optional[float] = typer.Option(None, "--timeout", "-t", help="Execution timeout")
        ):
            """Execute a tool."""
            try:
                # Parse parameters
                params = json.loads(parameters)
                
                # Create request
                request = ToolRequest(
                    tool_name=tool_name,
                    parameters=params,
                    timeout=timeout or self.config.timeout
                )
                
                # Execute tool
                response = asyncio.run(self.registry.execute_tool(request))
                
                # Output result
                if self.config.output_format == "json":
                    output = json.dumps(asdict(response), indent=2, default=str)
                    typer.echo(output)
                else:
                    if response.status == ToolStatus.COMPLETED:
                        typer.echo(f"✅ Tool executed successfully")
                        typer.echo(f"Result: {response.result}")
                        if response.execution_time:
                            typer.echo(f"Execution time: {response.execution_time:.2f}s")
                    else:
                        typer.echo(f"❌ Tool execution failed")
                        typer.echo(f"Error: {response.error_message}")
            
            except json.JSONDecodeError:
                typer.echo("❌ Invalid JSON parameters", err=True)
            except Exception as e:
                typer.echo(f"❌ Error: {e}", err=True)
        
        @self.app.command()
        def status(
            request_id: str = typer.Argument(..., help="Request ID to check")
        ):
            """Check execution status."""
            execution = self.registry.get_execution(request_id)
            
            if not execution:
                typer.echo("❌ Execution not found", err=True)
                return
            
            if self.config.output_format == "json":
                output = json.dumps(asdict(execution), indent=2, default=str)
                typer.echo(output)
            else:
                typer.echo(f"Status: {execution.status.value}")
                typer.echo(f"Tool: {execution.request.tool_name}")
                typer.echo(f"Started: {execution.start_time}")
                if execution.end_time:
                    typer.echo(f"Ended: {execution.end_time}")
                if execution.result:
                    typer.echo(f"Result: {execution.result}")
                if execution.error:
                    typer.echo(f"Error: {execution.error}")
        
        @self.app.command()
        def metrics():
            """Show registry metrics."""
            metrics = self.registry.get_metrics()
            
            if self.config.output_format == "json":
                output = json.dumps(metrics, indent=2)
                typer.echo(output)
            else:
                typer.echo(f"Total tools: {metrics['total_tools']}")
                typer.echo(f"Total executions: {metrics['total_executions']}")
                typer.echo("Categories:")
                for category, tools in metrics['categories'].items():
                    typer.echo(f"  {category}: {len(tools)} tools")
    
    def run(self) -> Any:
        """Run CLI application."""
        self.app()


class APIToolController:
    """RESTful API interface for tool control."""
    
    def __init__(self, registry: ToolRegistry, config: APIConfig):
        """Initialize API controller."""
        self.registry = registry
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.authenticator = Authenticator(config)
        
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for API interface")
        
        self.app = FastAPI(
            title="Tool Control API",
            description="RESTful API for tool control using RORO pattern",
            version="1.0.0"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self) -> Any:
        """Setup API middleware."""
        # CORS middleware
        if self.config.cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Trusted hosts middleware
        if self.config.trusted_hosts:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.config.trusted_hosts
            )
    
    def _setup_routes(self) -> Any:
        """Setup API routes."""
        
        # Pydantic models for request/response
        class ToolExecuteRequest(BaseModel):
            tool_name: str = Field(..., description="Name of the tool to execute")
            parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
            priority: ToolPriority = Field(default=ToolPriority.NORMAL, description="Execution priority")
            timeout: Optional[float] = Field(None, description="Execution timeout")
            metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
        
        class ToolExecuteResponse(BaseModel):
            request_id: str
            tool_name: str
            status: ToolStatus
            result: Optional[Any] = None
            error_message: Optional[str] = None
            error_code: Optional[str] = None
            execution_time: Optional[float] = None
            metadata: Dict[str, Any] = Field(default_factory=dict)
            timestamp: datetime
        
        class ToolListResponse(BaseModel):
            tools: List[Dict[str, Any]]
            total: int
            categories: Dict[str, List[str]]
        
        class MetricsResponse(BaseModel):
            total_tools: int
            total_executions: int
            tool_metrics: Dict[str, int]
            categories: Dict[str, List[str]]
        
        # Authentication dependency
        def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
            if not self.config.auth_enabled:
                return "anonymous"
            
            token = credentials.credentials if credentials else None
            user_id = self.authenticator.authenticate(token)
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid authentication")
            
            return user_id
        
        # Rate limiting dependency
        def check_rate_limit(request: Request, user_id: str = Depends(get_current_user)):
            if not self.config.rate_limit_enabled:
                return user_id
            
            identifier = f"{user_id}:{request.client.host}"
            if not self.rate_limiter.check_rate_limit(identifier):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            return user_id
        
        # Routes
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """API root endpoint."""
            return {
                "message": "Tool Control API",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        @self.app.get("/tools", response_model=ToolListResponse)
        async def list_tools(
            category: Optional[str] = None,
            user_id: str = Depends(check_rate_limit)
        ):
            """List available tools."""
            tools = self.registry.list_tools()
            
            if category:
                tools = [t for t in tools if t.category == category]
            
            categories = self.registry.get_tool_categories()
            
            return ToolListResponse(
                tools=[asdict(tool) for tool in tools],
                total=len(tools),
                categories=categories
            )
        
        @self.app.get("/tools/{tool_name}")
        async def get_tool(
            tool_name: str,
            user_id: str = Depends(check_rate_limit)
        ):
            """Get tool definition."""
            tool = self.registry.get_tool(tool_name)
            
            if not tool:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            return asdict(tool)
        
        @self.app.post("/execute", response_model=ToolExecuteResponse)
        async def execute_tool(
            request: ToolExecuteRequest,
            user_id: str = Depends(check_rate_limit)
        ):
            """Execute a tool."""
            # Create tool request
            tool_request = ToolRequest(
                tool_name=request.tool_name,
                parameters=request.parameters,
                user_id=user_id,
                priority=request.priority,
                timeout=request.timeout,
                metadata=request.metadata
            )
            
            # Check authorization
            if not self.authenticator.authorize(user_id, request.tool_name):
                raise HTTPException(status_code=403, detail="Not authorized to use this tool")
            
            # Execute tool
            response = await self.registry.execute_tool(tool_request)
            
            # Log request if enabled
            if self.config.log_requests:
                logger.info(f"API request: {user_id} executed {request.tool_name}")
            
            return ToolExecuteResponse(**asdict(response))
        
        @self.app.get("/status/{request_id}")
        async def get_status(
            request_id: str,
            user_id: str = Depends(check_rate_limit)
        ):
            """Get execution status."""
            execution = self.registry.get_execution(request_id)
            
            if not execution:
                raise HTTPException(status_code=404, detail="Execution not found")
            
            return asdict(execution)
        
        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics(
            user_id: str = Depends(check_rate_limit)
        ):
            """Get registry metrics."""
            metrics = self.registry.get_metrics()
            return MetricsResponse(**metrics)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "tools_registered": len(self.registry.list_tools())
            }
    
    def run(self) -> Any:
        """Run API server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            workers=self.config.workers
        )


# Example tool implementations
def example_calculator_tool(request: ToolRequest) -> Any:
    """Example calculator tool."""
    operation = request.parameters.get('operation')
    a = request.parameters.get('a', 0)
    b = request.parameters.get('b', 0)
    
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("Division by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


async def example_async_tool(request: ToolRequest) -> Any:
    """Example async tool."""
    delay = request.parameters.get('delay', 1.0)
    message = request.parameters.get('message', 'Hello World')
    
    await asyncio.sleep(delay)
    
    return {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'delay': delay
    }


def example_file_processor_tool(request: ToolRequest) -> Any:
    """Example file processing tool."""
    file_path = request.parameters.get('file_path')
    operation = request.parameters.get('operation', 'read')
    
    if not file_path:
        raise ValueError("file_path parameter is required")
    
    path = Path(file_path)
    
    if operation == 'read':
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return {
            'content': content,
            'size': len(content),
            'lines': len(content.splitlines())
        }
    
    elif operation == 'write':
        content = request.parameters.get('content', '')
        
        with open(path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        return {
            'message': f"File written successfully: {file_path}",
            'size': len(content)
        }
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Example usage functions
def demonstrate_cli_interface():
    """Demonstrate CLI interface."""
    # Create registry and register tools
    registry = ToolRegistry()
    
    # Register calculator tool
    calc_definition = ToolDefinition(
        name="calculator",
        description="Simple calculator with basic operations",
        category="math",
        tags=["math", "calculator"],
        parameters={
            'operation': {'type': 'string', 'description': 'Operation to perform'},
            'a': {'type': 'number', 'description': 'First number'},
            'b': {'type': 'number', 'description': 'Second number'}
        },
        required_parameters=['operation', 'a', 'b'],
        examples=[
            {'operation': 'add', 'a': 5, 'b': 3},
            {'operation': 'multiply', 'a': 4, 'b': 7}
        ]
    )
    registry.register_tool(calc_definition, example_calculator_tool)
    
    # Register async tool
    async_definition = ToolDefinition(
        name="async_demo",
        description="Async tool demonstration",
        category="demo",
        tags=["async", "demo"],
        parameters={
            'delay': {'type': 'number', 'description': 'Delay in seconds'},
            'message': {'type': 'string', 'description': 'Message to return'}
        },
        required_parameters=['delay'],
        examples=[
            {'delay': 2.0, 'message': 'Hello from async tool'}
        ]
    )
    registry.register_tool(async_definition, example_async_tool)
    
    # Create CLI controller
    cli_config = CLIConfig(verbose=True, output_format="json")
    cli = CLIToolController(registry, cli_config)
    
    print("CLI Tool Controller Demo")
    print("Available commands:")
    print("  list-tools")
    print("  execute <tool_name> --params '{\"param\": \"value\"}'")
    print("  status <request_id>")
    print("  metrics")
    
    # Note: In a real application, you would call cli.run()
    # For demonstration, we'll show the structure


def demonstrate_api_interface():
    """Demonstrate API interface."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI not available, skipping API demonstration")
        return
    
    # Create registry and register tools
    registry = ToolRegistry()
    
    # Register tools
    calc_definition = ToolDefinition(
        name="calculator",
        description="Simple calculator with basic operations",
        category="math",
        parameters={
            'operation': {'type': 'string', 'description': 'Operation to perform'},
            'a': {'type': 'number', 'description': 'First number'},
            'b': {'type': 'number', 'description': 'Second number'}
        },
        required_parameters=['operation', 'a', 'b']
    )
    registry.register_tool(calc_definition, example_calculator_tool)
    
    file_definition = ToolDefinition(
        name="file_processor",
        description="File processing operations",
        category="file",
        parameters={
            'file_path': {'type': 'string', 'description': 'Path to file'},
            'operation': {'type': 'string', 'description': 'Operation (read/write)'},
            'content': {'type': 'string', 'description': 'Content to write'}
        },
        required_parameters=['file_path', 'operation']
    )
    registry.register_tool(file_definition, example_file_processor_tool)
    
    # Create API controller
    api_config = APIConfig(
        host="0.0.0.0",
        port=8000,
        debug=True,
        cors_origins=["*"],
        auth_tokens={"demo_user": "demo_token_123"},
        rate_limit_requests=100,
        rate_limit_window=60
    )
    
    api = APIToolController(registry, api_config)
    
    print("API Tool Controller Demo")
    print("API will be available at:")
    print("  - Main API: http://localhost:8000")
    print("  - Documentation: http://localhost:8000/docs")
    print("  - Health check: http://localhost:8000/health")
    print("  - List tools: http://localhost:8000/tools")
    print("  - Execute tool: POST http://localhost:8000/execute")
    
    # Note: In a real application, you would call api.run()
    # For demonstration, we'll show the structure


def demonstrate_roro_pattern():
    """Demonstrate RORO pattern usage."""
    registry = ToolRegistry()
    
    # Register a simple tool
    tool_definition = ToolDefinition(
        name="greeter",
        description="Simple greeting tool",
        category="demo",
        parameters={
            'name': {'type': 'string', 'description': 'Name to greet'},
            'language': {'type': 'string', 'description': 'Language for greeting'}
        },
        required_parameters=['name'],
        optional_parameters=['language']
    )
    
    def greeter_tool(request: ToolRequest) -> Any:
        name = request.parameters.get('name', 'World')
        language = request.parameters.get('language', 'en')
        
        greetings = {
            'en': f"Hello, {name}!",
            'es': f"¡Hola, {name}!",
            'fr': f"Bonjour, {name}!",
            'de': f"Hallo, {name}!"
        }
        
        return greetings.get(language, greetings['en'])
    
    registry.register_tool(tool_definition, greeter_tool)
    
    # Create and execute requests
    requests = [
        ToolRequest(
            tool_name="greeter",
            parameters={'name': 'Alice', 'language': 'en'}
        ),
        ToolRequest(
            tool_name="greeter",
            parameters={'name': 'Bob', 'language': 'es'}
        ),
        ToolRequest(
            tool_name="greeter",
            parameters={'name': 'Charlie', 'language': 'fr'}
        )
    ]
    
    print("RORO Pattern Demo")
    print("Executing multiple tool requests...")
    
    async def run_requests():
        
    """run_requests function."""
for request in requests:
            response = await registry.execute_tool(request)
            print(f"Request {response.request_id}: {response.result}")
    
    asyncio.run(run_requests())


def main():
    """Main function demonstrating RORO tool control."""
    logger.info("Starting RORO tool control examples")
    
    # Demonstrate RORO pattern
    try:
        demonstrate_roro_pattern()
    except Exception as e:
        logger.error(f"RORO pattern demonstration failed: {e}")
    
    # Demonstrate CLI interface
    try:
        demonstrate_cli_interface()
    except Exception as e:
        logger.error(f"CLI interface demonstration failed: {e}")
    
    # Demonstrate API interface
    try:
        demonstrate_api_interface()
    except Exception as e:
        logger.error(f"API interface demonstration failed: {e}")
    
    logger.info("RORO tool control examples completed")


match __name__:
    case "__main__":
    main() 