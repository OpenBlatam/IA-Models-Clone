# RORO Pattern Tool Control - Implementation Summary

## Overview

This implementation provides comprehensive CLI and RESTful API interfaces using the RORO (Receive Object, Return Object) pattern for tool control. The system follows the established patterns of guard clauses, early returns, structured logging, and modular design.

## Key Features

### 1. RORO Pattern Implementation
- **Consistent Data Flow**: All tools receive and return structured objects
- **Type Safety**: Strong typing with dataclasses and Pydantic models
- **Request/Response Objects**: Standardized ToolRequest and ToolResponse classes
- **Metadata Support**: Rich metadata for tracking and debugging
- **Async Support**: Both synchronous and asynchronous tool execution

### 2. CLI Interface
- **Command Parsing**: Advanced command-line argument parsing with Typer
- **Validation**: Request validation before execution
- **Output Formats**: JSON and human-readable output formats
- **Interactive Commands**: List, execute, status, and metrics commands
- **Error Handling**: Comprehensive error reporting and recovery

### 3. RESTful API Interface
- **FastAPI Integration**: Modern, fast web framework with automatic documentation
- **OpenAPI Generation**: Automatic API documentation and client generation
- **Middleware Support**: CORS, trusted hosts, and custom middleware
- **Authentication**: Bearer token authentication with extensible auth system
- **Rate Limiting**: Configurable rate limiting per user/IP

### 4. Tool Registry and Management
- **Tool Registration**: Dynamic tool registration with metadata
- **Validation**: Request validation against tool definitions
- **Categories and Tags**: Organized tool management with categories and tags
- **Versioning**: Tool versioning and compatibility tracking
- **Examples**: Built-in examples for each tool

### 5. Security and Performance
- **Authentication**: Multiple authentication levels (none, basic, token, JWT, OAuth)
- **Authorization**: User-based authorization for tool access
- **Rate Limiting**: Configurable rate limiting to prevent abuse
- **Input Validation**: Comprehensive input validation and sanitization
- **Performance Monitoring**: Execution time tracking and metrics

## Core Classes

### ToolRequest (RORO Input)
```python
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
```

### ToolResponse (RORO Output)
```python
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
```

### ToolRegistry
```python
class ToolRegistry:
    """Tool registry and management system."""
    
    async def execute_tool(self, request: ToolRequest) -> ToolResponse:
        """Execute a tool using RORO pattern."""
        # Guard clauses for early returns
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
                error_message=f"Tool '{request.tool_name}' not found",
                error_code="TOOL_NOT_FOUND"
            )
        
        # Happy path - execute tool
        try:
            start_time = time.time()
            if asyncio.iscoroutinefunction(handler):
                result = await handler(request)
            else:
                result = handler(request)
            
            execution_time = time.time() - start_time
            
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.COMPLETED,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            return ToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                status=ToolStatus.FAILED,
                error_message=str(e),
                error_code="EXECUTION_ERROR"
            )
```

## Design Patterns Applied

### 1. RORO Pattern
- **Receive Object**: All tools receive a standardized ToolRequest object
- **Return Object**: All tools return a standardized ToolResponse object
- **Consistent Interface**: Uniform data flow across all tools
- **Metadata Rich**: Rich metadata for tracking and debugging

### 2. Guard Clauses and Early Returns
- All functions start with validation checks that return early on failure
- Prevents deep nesting and keeps the happy path at the end
- Improves code readability and maintainability

### 3. Registry Pattern
- Centralized tool registration and management
- Dynamic tool discovery and loading
- Consistent tool interface and metadata

### 4. Dependency Injection
- Configuration objects injected into controllers
- Authentication and rate limiting as dependencies
- Modular and testable architecture

### 5. Middleware Pattern
- CORS, authentication, and rate limiting as middleware
- Chainable middleware for different concerns
- Clean separation of cross-cutting concerns

## CLI Interface

### Command Structure
```bash
# List available tools
tool-control list-tools --category math --verbose

# Execute a tool
tool-control execute calculator --params '{"operation": "add", "a": 5, "b": 3}'

# Check execution status
tool-control status <request_id>

# View metrics
tool-control metrics
```

### Features
- **Interactive Commands**: Easy-to-use command-line interface
- **JSON Parameters**: Structured parameter passing
- **Output Formats**: JSON and human-readable output
- **Error Handling**: Clear error messages and recovery
- **Help System**: Built-in help and documentation

## RESTful API Interface

### Endpoints
```
GET  /                    - API root and information
GET  /tools              - List all available tools
GET  /tools/{tool_name}  - Get specific tool definition
POST /execute            - Execute a tool
GET  /status/{request_id} - Get execution status
GET  /metrics            - Get system metrics
GET  /health             - Health check
```

### Request/Response Examples
```python
# Execute tool request
POST /execute
{
    "tool_name": "calculator",
    "parameters": {
        "operation": "add",
        "a": 5,
        "b": 3
    },
    "priority": "normal",
    "timeout": 30.0
}

# Tool response
{
    "request_id": "uuid-1234",
    "tool_name": "calculator",
    "status": "completed",
    "result": 8,
    "execution_time": 0.001,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Features
- **OpenAPI Documentation**: Automatic API documentation
- **Authentication**: Bearer token authentication
- **Rate Limiting**: Configurable rate limiting
- **CORS Support**: Cross-origin resource sharing
- **Error Handling**: Standardized error responses

## Tool Implementation Examples

### Synchronous Tool
```python
def calculator_tool(request: ToolRequest) -> Any:
    """Simple calculator tool."""
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
```

### Asynchronous Tool
```python
async def async_tool(request: ToolRequest) -> Any:
    """Async tool example."""
    delay = request.parameters.get('delay', 1.0)
    message = request.parameters.get('message', 'Hello World')
    
    await asyncio.sleep(delay)
    
    return {
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'delay': delay
    }
```

### File Processing Tool
```python
def file_processor_tool(request: ToolRequest) -> Any:
    """File processing tool."""
    file_path = request.parameters.get('file_path')
    operation = request.parameters.get('operation', 'read')
    
    if not file_path:
        raise ValueError("file_path parameter is required")
    
    path = Path(file_path)
    
    if operation == 'read':
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r') as f:
            content = f.read()
        
        return {
            'content': content,
            'size': len(content),
            'lines': len(content.splitlines())
        }
    
    elif operation == 'write':
        content = request.parameters.get('content', '')
        
        with open(path, 'w') as f:
            f.write(content)
        
        return {
            'message': f"File written successfully: {file_path}",
            'size': len(content)
        }
    
    else:
        raise ValueError(f"Unknown operation: {operation}")
```

## Security Features

### 1. Authentication
- **Bearer Token**: Simple token-based authentication
- **Extensible**: Support for JWT, OAuth, and custom auth
- **User Management**: User-based access control
- **Session Tracking**: Session-based authentication

### 2. Authorization
- **Tool-Level**: Authorization per tool
- **User-Based**: User-specific permissions
- **Role-Based**: Role-based access control (extensible)
- **Resource-Based**: Resource-level permissions

### 3. Rate Limiting
- **Per-User**: Rate limiting per user
- **Per-IP**: Rate limiting per IP address
- **Configurable**: Adjustable limits and windows
- **Redis Support**: Distributed rate limiting with Redis

### 4. Input Validation
- **Parameter Validation**: Validate all input parameters
- **Type Checking**: Strong type checking
- **Sanitization**: Input sanitization and cleaning
- **Schema Validation**: JSON schema validation

## Performance Features

### 1. Async Support
- **Async Tools**: Support for asynchronous tool execution
- **Concurrent Execution**: Multiple tools can run concurrently
- **Non-Blocking**: Non-blocking I/O operations
- **Scalability**: Horizontal scaling capabilities

### 2. Caching
- **Result Caching**: Cache tool results
- **Metadata Caching**: Cache tool metadata
- **Redis Integration**: Distributed caching with Redis
- **TTL Support**: Time-to-live for cached data

### 3. Monitoring
- **Execution Time**: Track tool execution time
- **Metrics Collection**: Collect performance metrics
- **Health Checks**: System health monitoring
- **Logging**: Comprehensive logging and debugging

## Usage Examples

### CLI Usage
```python
# Create registry and register tools
registry = ToolRegistry()
registry.register_tool(calc_definition, calculator_tool)

# Create CLI controller
cli_config = CLIConfig(verbose=True, output_format="json")
cli = CLIToolController(registry, cli_config)

# Run CLI
cli.run()
```

### API Usage
```python
# Create API controller
api_config = APIConfig(
    host="0.0.0.0",
    port=8000,
    auth_tokens={"user1": "token123"},
    rate_limit_requests=100
)

api = APIToolController(registry, api_config)

# Run API server
api.run()
```

### Direct RORO Usage
```python
# Create request
request = ToolRequest(
    tool_name="calculator",
    parameters={"operation": "add", "a": 5, "b": 3}
)

# Execute tool
response = await registry.execute_tool(request)

# Handle response
if response.status == ToolStatus.COMPLETED:
    print(f"Result: {response.result}")
else:
    print(f"Error: {response.error_message}")
```

## Integration Examples

### FastAPI Integration
```python
from fastapi import FastAPI, Depends
from roro_tool_control_examples import ToolRegistry, ToolRequest

app = FastAPI()
registry = ToolRegistry()

@app.post("/custom-execute")
async def custom_execute(request: ToolRequest):
    response = await registry.execute_tool(request)
    return response
```

### Django Integration
```python
from django.http import JsonResponse
from roro_tool_control_examples import ToolRegistry, ToolRequest

registry = ToolRegistry()

def execute_tool_view(request):
    tool_name = request.POST.get('tool_name')
    parameters = json.loads(request.POST.get('parameters', '{}'))
    
    tool_request = ToolRequest(
        tool_name=tool_name,
        parameters=parameters
    )
    
    response = asyncio.run(registry.execute_tool(tool_request))
    return JsonResponse(asdict(response))
```

### Celery Integration
```python
from celery import Celery
from roro_tool_control_examples import ToolRegistry, ToolRequest

app = Celery('tool_control')
registry = ToolRegistry()

@app.task
def execute_tool_task(tool_name, parameters):
    request = ToolRequest(tool_name=tool_name, parameters=parameters)
    response = asyncio.run(registry.execute_tool(request))
    return asdict(response)
```

## Best Practices

### 1. Tool Development
- **RORO Pattern**: Always use ToolRequest and ToolResponse
- **Error Handling**: Proper exception handling and error codes
- **Validation**: Validate all input parameters
- **Documentation**: Document tool purpose and parameters
- **Testing**: Comprehensive unit and integration tests

### 2. Security
- **Input Validation**: Validate and sanitize all inputs
- **Authentication**: Implement proper authentication
- **Authorization**: Use appropriate authorization levels
- **Rate Limiting**: Implement rate limiting for public APIs
- **Logging**: Log security-relevant events

### 3. Performance
- **Async Tools**: Use async for I/O-bound operations
- **Caching**: Cache frequently accessed data
- **Resource Management**: Proper resource cleanup
- **Monitoring**: Monitor performance and errors
- **Scaling**: Design for horizontal scaling

### 4. Deployment
- **Configuration**: Use environment-based configuration
- **Health Checks**: Implement health check endpoints
- **Logging**: Structured logging for production
- **Monitoring**: Comprehensive monitoring and alerting
- **Documentation**: Keep API documentation updated

## Conclusion

This implementation provides a robust, scalable, and secure foundation for tool control using the RORO pattern. The modular design, comprehensive security features, and multiple interface options make it suitable for production use while maintaining flexibility and ease of use.

The system follows established patterns and best practices, ensuring maintainability, testability, and extensibility. The RORO pattern provides consistent data flow, while the CLI and API interfaces offer multiple ways to interact with tools.

Key benefits:
- **Consistency**: RORO pattern ensures consistent data flow
- **Flexibility**: Multiple interface options (CLI, API, direct)
- **Security**: Comprehensive security features
- **Performance**: Async support and performance monitoring
- **Maintainability**: Clean, modular design with clear interfaces
- **Extensibility**: Easy to add new tools and interfaces 