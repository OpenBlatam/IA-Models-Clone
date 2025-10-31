# Route Structure Guide

A comprehensive guide to structuring routes and dependencies clearly to optimize readability and maintainability in the HeyGen AI FastAPI application.

## 🎯 Overview

This guide covers the complete route structure system designed to:
- **Clear Organization**: Well-structured routes with logical grouping
- **Dependency Injection**: Centralized dependency management
- **Modular Design**: Separate route modules for different features
- **Consistent Patterns**: Standardized route patterns and responses
- **Performance Monitoring**: Built-in metrics and monitoring
- **Documentation**: Automatic API documentation generation
- **Maintainability**: Easy to understand and modify structure

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Route Organization](#route-organization)
3. [Dependency Management](#dependency-management)
4. [Base Route Class](#base-route-class)
5. [Route Categories](#route-categories)
6. [User Routes](#user-routes)
7. [Video Routes](#video-routes)
8. [System Routes](#system-routes)
9. [Middleware Integration](#middleware-integration)
10. [Error Handling](#error-handling)
11. [Performance Monitoring](#performance-monitoring)
12. [Best Practices](#best-practices)
13. [Integration Examples](#integration-examples)
14. [Development Workflow](#development-workflow)

## 🏗️ System Architecture

### **Route Structure Architecture**

```
FastAPI Application
├── Route Registry (Central route management)
├── Dependency Container (Service management)
├── Base Route Class (Common functionality)
├── Route Categories (Logical grouping)
│   ├── User Routes (Authentication & Management)
│   ├── Video Routes (Processing & Management)
│   ├── AI Routes (AI & ML Operations)
│   ├── Analytics Routes (Reporting & Metrics)
│   ├── System Routes (Health & Monitoring)
│   └── External Routes (API Integration)
├── Middleware (Request/Response processing)
├── Error Handling (Centralized error management)
└── Performance Monitoring (Metrics & analytics)
```

### **Core Components**

1. **RouteRegistry**: Central route management and registration
2. **DependencyContainer**: Service dependency injection
3. **BaseRoute**: Common route functionality and patterns
4. **Route Categories**: Logical grouping of related routes
5. **Middleware**: Request/response processing
6. **Error Handling**: Centralized error management

## 📁 Route Organization

### **1. Route Registry Setup**

```python
from api.routes import route_registry, dependency_container

# Register routes with clear organization
route_registry.register_route(
    "users",
    user_routes.router,
    prefix="/api/v1/users",
    tags=["users", "authentication"]
)

route_registry.register_route(
    "videos",
    video_routes.router,
    prefix="/api/v1/videos",
    tags=["videos", "processing"]
)

# Setup all routes on FastAPI app
route_registry.setup_app(app)
```

### **2. Route Categories**

```python
ROUTE_CATEGORIES = {
    "AUTH": "Authentication and Authorization",
    "USERS": "User Management",
    "VIDEOS": "Video Processing",
    "AI": "AI and Machine Learning",
    "ANALYTICS": "Analytics and Reporting",
    "SYSTEM": "System and Health",
    "EXTERNAL": "External API Integration",
    "UTILS": "Utility and Helper Functions"
}
```

### **3. Route Structure**

```
/api/v1/
├── users/                    # User management
│   ├── GET /                 # List users
│   ├── POST /                # Create user
│   ├── GET /{user_id}        # Get user
│   ├── PUT /{user_id}        # Update user
│   ├── DELETE /{user_id}     # Delete user
│   ├── POST /login           # User login
│   ├── POST /logout          # User logout
│   ├── GET /{user_id}/profile # Get profile
│   └── GET /{user_id}/statistics # Get statistics
├── videos/                   # Video processing
│   ├── GET /                 # List videos
│   ├── POST /                # Create video
│   ├── GET /{video_id}       # Get video
│   ├── PUT /{video_id}       # Update video
│   ├── DELETE /{video_id}    # Delete video
│   ├── POST /upload          # Upload video
│   ├── POST /{video_id}/process # Start processing
│   ├── GET /{video_id}/status # Get status
│   ├── POST /{video_id}/cancel # Cancel processing
│   ├── GET /{video_id}/download # Download video
│   └── GET /{video_id}/analytics # Get analytics
└── system/                   # System routes
    ├── GET /health           # Health check
    ├── GET /api/info         # API information
    └── GET /api/metrics      # Application metrics
```

## 🔧 Dependency Management

### **1. Dependency Container Setup**

```python
from api.routes import dependency_container

# Register services
dependency_container.register_singleton("db_operations", db_operations)
dependency_container.register_singleton("api_operations", api_operations)
dependency_container.register_singleton("file_storage", file_storage)

# Get services
db_ops = dependency_container.get_service("db_operations")
api_ops = dependency_container.get_service("api_operations")
```

### **2. Route Dependencies**

```python
class UserRoutes(BaseRoute):
    def __init__(self, db_operations, api_operations):
        super().__init__(
            name="User Management",
            description="User management operations",
            category=RouteCategory.USERS,
            tags=["users", "authentication"],
            prefix="/users",
            dependencies={
                "db_ops": db_operations,
                "api_ops": api_operations
            }
        )
```

### **3. Dependency Injection in Routes**

```python
@app.get("/users")
async def get_users():
    db_ops = self.get_dependency("db_ops")
    api_ops = self.get_dependency("api_ops")
    
    # Use dependencies
    users = await db_ops.execute_query("SELECT * FROM users")
    return users
```

## 🏛️ Base Route Class

### **1. Base Route Features**

```python
class BaseRoute:
    def __init__(self, name, description, category, tags, prefix, dependencies):
        self.name = name
        self.description = description
        self.category = category
        self.tags = tags
        self.prefix = prefix
        self.dependencies = dependencies
        
        # Create router with standard responses
        self.router = APIRouter(
            prefix=prefix,
            tags=tags,
            responses={
                200: {"description": "Success"},
                400: {"description": "Bad Request"},
                401: {"description": "Unauthorized"},
                403: {"description": "Forbidden"},
                404: {"description": "Not Found"},
                500: {"description": "Internal Server Error"}
            }
        )
```

### **2. Standard Response Methods**

```python
# Success response
def success_response(self, data=None, message="Success", request_id=None):
    return BaseResponse(
        success=True,
        message=message,
        data=data,
        request_id=request_id
    )

# Error response
def error_response(self, message, error_code=None, error_details=None, request_id=None):
    return ErrorResponse(
        success=False,
        message=message,
        error_code=error_code,
        error_details=error_details,
        request_id=request_id
    )

# Paginated response
def paginated_response(self, data, total_count, page=1, page_size=10, request_id=None):
    return PaginatedResponse(
        success=True,
        data=data,
        total_count=total_count,
        page=page,
        page_size=page_size,
        request_id=request_id
    )
```

### **3. Route Decorators**

```python
# Performance monitoring
@route_metrics
async def get_users():
    # Route implementation
    pass

# Authentication requirement
@require_auth
async def update_user():
    # Route implementation
    pass

# Rate limiting
@rate_limit(requests_per_minute=100)
async def create_user():
    # Route implementation
    pass

# Response caching
@cache_response(ttl=300)
async def get_user_profile():
    # Route implementation
    pass
```

## 👥 User Routes

### **1. User Route Structure**

```python
class UserRoutes(BaseRoute):
    def _register_routes(self):
        # CRUD Operations
        self.router.get("/", response_model=PaginatedResponse)
        self.router.get("/{user_id}", response_model=UserResponse)
        self.router.post("/", response_model=UserResponse)
        self.router.put("/{user_id}", response_model=UserResponse)
        self.router.delete("/{user_id}", response_model=BaseResponse)
        
        # Profile Operations
        self.router.get("/{user_id}/profile", response_model=Dict[str, Any])
        self.router.put("/{user_id}/profile", response_model=Dict[str, Any])
        
        # Authentication Operations
        self.router.post("/login", response_model=Dict[str, Any])
        self.router.post("/logout", response_model=BaseResponse)
        
        # Statistics Operations
        self.router.get("/{user_id}/statistics", response_model=Dict[str, Any])
```

### **2. User CRUD Operations**

```python
@app.get("/users")
@route_metrics
@rate_limit(requests_per_minute=100)
@cache_response(ttl=300)
async def get_users(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get paginated list of users with filtering."""
    try:
        db_ops = self.get_dependency("db_ops")
        
        # Build query with filters
        query = "SELECT * FROM users WHERE 1=1"
        params = {}
        
        if search:
            query += " AND (name ILIKE :search OR email ILIKE :search)"
            params["search"] = f"%{search}%"
        
        if status:
            query += " AND status = :status"
            params["status"] = status
        
        # Add pagination
        query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
        params["limit"] = pagination.page_size
        params["offset"] = (pagination.page - 1) * pagination.page_size
        
        # Execute query
        users = await db_ops.execute_query(query, parameters=params)
        
        return self.paginated_response(
            data=users,
            total_count=len(users),
            page=pagination.page,
            page_size=pagination.page_size
        )
        
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")
```

### **3. User Authentication**

```python
@app.post("/login")
@route_metrics
@rate_limit(requests_per_minute=5)
async def login(email: str, password: str):
    """User login authentication."""
    try:
        db_ops = self.get_dependency("db_ops")
        
        # Get user by email
        users = await db_ops.execute_query(
            "SELECT * FROM users WHERE email = :email AND status = 'active'",
            parameters={"email": email}
        )
        
        if not users:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user = users[0]
        
        # Verify password
        if user["password"] != password:  # Use proper password verification
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate access token
        access_token = f"token_{user['id']}_{int(time.time())}"
        
        return self.success_response(
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "email": user["email"],
                    "role": user["role"]
                }
            },
            message="Login successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
```

## 🎥 Video Routes

### **1. Video Route Structure**

```python
class VideoRoutes(BaseRoute):
    def _register_routes(self):
        # CRUD Operations
        self.router.get("/", response_model=PaginatedResponse)
        self.router.get("/{video_id}", response_model=VideoResponse)
        self.router.post("/", response_model=VideoResponse)
        self.router.put("/{video_id}", response_model=VideoResponse)
        self.router.delete("/{video_id}", response_model=BaseResponse)
        
        # Upload Operations
        self.router.post("/upload", response_model=VideoResponse)
        
        # Processing Operations
        self.router.post("/{video_id}/process", response_model=VideoResponse)
        self.router.get("/{video_id}/status", response_model=Dict[str, Any])
        self.router.post("/{video_id}/cancel", response_model=BaseResponse)
        
        # Download Operations
        self.router.get("/{video_id}/download")
        
        # Analytics Operations
        self.router.get("/{video_id}/analytics", response_model=Dict[str, Any])
```

### **2. Video Processing**

```python
@app.post("/{video_id}/process")
@route_metrics
@require_auth
@rate_limit(requests_per_minute=10)
async def start_video_processing(
    video_id: int = Path(...),
    processing_options: Optional[VideoProcessingOptions] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Start or restart video processing."""
    try:
        db_ops = self.get_dependency("db_ops")
        api_ops = self.get_dependency("api_ops")
        
        # Check permissions
        videos = await db_ops.execute_query(
            "SELECT * FROM videos WHERE id = :video_id",
            parameters={"video_id": video_id}
        )
        
        if not videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video = videos[0]
        
        if current_user["user_id"] != str(video["user_id"]) and current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Start processing with external API
        processing_data = {
            "video_id": video_id,
            "title": video["title"],
            "description": video["description"],
            "type": video["type"]
        }
        
        if processing_options:
            processing_data["options"] = processing_options.dict()
        
        processing_response = await api_ops.post(
            endpoint="/process-video",
            data=processing_data
        )
        
        # Update video status
        await db_ops.execute_update(
            table="videos",
            data={
                "external_job_id": processing_response["data"]["job_id"],
                "status": "processing",
                "processing_started_at": datetime.now(timezone.utc)
            },
            where_conditions={"id": video_id}
        )
        
        return self.success_response(
            data={
                "video_id": video_id,
                "status": "processing",
                "job_id": processing_response["data"]["job_id"]
            },
            message="Video processing started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting video processing {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start video processing")
```

### **3. Video Upload**

```python
@app.post("/upload")
@route_metrics
@require_auth
@rate_limit(requests_per_minute=5)
async def upload_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload video file for processing."""
    try:
        db_ops = self.get_dependency("db_ops")
        file_storage = self.get_dependency("file_storage")
        api_ops = self.get_dependency("api_ops")
        
        # Validate file
        allowed_types = ["video/mp4", "video/avi", "video/mov"]
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Save file
        file_path = await file_storage.save_file(file, f"{uuid.uuid4()}.mp4")
        
        # Create video record
        video_data = {
            "title": title,
            "description": description,
            "user_id": int(current_user["user_id"]),
            "status": "uploaded",
            "file_path": file_path,
            "file_size": file.size,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = await db_ops.execute_insert(
            table="videos",
            data=video_data,
            returning="id, title, status, file_path, created_at"
        )
        
        # Start processing
        processing_response = await api_ops.post(
            endpoint="/process-uploaded-video",
            data={
                "video_id": result["id"],
                "file_path": file_path,
                "title": title
            }
        )
        
        return self.success_response(
            data=result,
            message="Video uploaded and processing started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload video")
```

## 🔧 System Routes

### **1. Health Check**

```python
@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "external_api": "connected",
            "file_storage": "connected"
        }
    }
```

### **2. API Information**

```python
@app.get("/api/info", tags=["system"])
async def api_info():
    """Get API information."""
    return {
        "name": "HeyGen AI API",
        "version": "1.0.0",
        "description": "Advanced AI-powered video generation and processing API",
        "categories": ROUTE_CATEGORIES,
        "endpoints": {
            "users": "/api/v1/users",
            "videos": "/api/v1/videos",
            "health": "/health",
            "docs": "/docs"
        }
    }
```

### **3. Metrics Endpoint**

```python
@app.get("/api/metrics", tags=["system"])
async def get_metrics():
    """Get application metrics."""
    try:
        db_operations = dependency_container.get_service("db_operations")
        api_operations = dependency_container.get_service("api_operations")
        
        return {
            "database": {
                "query_metrics": db_operations.get_query_metrics(),
                "cache_stats": db_operations.get_cache_stats(),
                "connection_metrics": db_operations.db_manager.get_connection_metrics()
            },
            "api": {
                "api_metrics": api_operations.get_api_metrics(),
                "cache_stats": api_operations.get_cache_stats(),
                "circuit_breaker": api_operations.api_manager.get_circuit_breaker_state()
            },
            "routes": {
                "users": user_routes.get_route_info(),
                "videos": video_routes.get_route_info()
            }
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": "Failed to retrieve metrics"}
```

## 🔄 Middleware Integration

### **1. Request Logging Middleware**

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration_ms=duration * 1000
    )
    
    return response
```

### **2. Error Handling Middleware**

```python
@app.middleware("http")
async def handle_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        
        # Return structured error response
        return Response(
            content=json.dumps({
                "success": False,
                "message": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "timestamp": time.time()
            }),
            status_code=500,
            media_type="application/json"
        )
```

### **3. CORS and Compression**

```python
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## ⚠️ Error Handling

### **1. Standardized Error Responses**

```python
# HTTP Exception handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now(timezone.utc)
        ).dict()
    )

# Validation error handling
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            success=False,
            message="Validation error",
            error_code="VALIDATION_ERROR",
            error_details=exc.errors(),
            timestamp=datetime.now(timezone.utc)
        ).dict()
    )
```

### **2. Route-Level Error Handling**

```python
@app.get("/users")
async def get_users():
    try:
        db_ops = self.get_dependency("db_ops")
        users = await db_ops.execute_query("SELECT * FROM users")
        return self.success_response(data=users)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")
```

## 📊 Performance Monitoring

### **1. Route Metrics**

```python
@route_metrics
async def get_users():
    # Route implementation
    pass

# Metrics are automatically tracked:
# - Request duration
# - Success/failure rates
# - Request count
# - Response size
```

### **2. Performance Analytics**

```python
@app.get("/api/performance")
async def get_performance_metrics():
    """Get performance metrics for all routes."""
    routes = route_registry.get_all_routes()
    
    performance_data = {}
    for name, route_info in routes.items():
        router = route_info["router"]
        metrics = router.get_metrics()
        
        performance_data[name] = {
            "total_requests": len(metrics),
            "successful_requests": sum(1 for m in metrics.values() if m.success),
            "failed_requests": sum(1 for m in metrics.values() if not m.success),
            "avg_duration_ms": sum(m.duration_ms or 0 for m in metrics.values()) / len(metrics) if metrics else 0
        }
    
    return performance_data
```

## ✅ Best Practices

### **1. Route Organization**

```python
# ✅ Good: Clear route organization
class UserRoutes(BaseRoute):
    def _register_routes(self):
        # CRUD operations
        self.router.get("/", response_model=PaginatedResponse)
        self.router.get("/{user_id}", response_model=UserResponse)
        self.router.post("/", response_model=UserResponse)
        
        # Authentication operations
        self.router.post("/login", response_model=Dict[str, Any])
        self.router.post("/logout", response_model=BaseResponse)
        
        # Profile operations
        self.router.get("/{user_id}/profile", response_model=Dict[str, Any])
        self.router.put("/{user_id}/profile", response_model=Dict[str, Any])

# ❌ Bad: Disorganized routes
@app.get("/users")
@app.post("/create_user")
@app.get("/get_user_profile")
@app.post("/user_login")
```

### **2. Dependency Management**

```python
# ✅ Good: Centralized dependency management
class UserRoutes(BaseRoute):
    def __init__(self, db_operations, api_operations):
        super().__init__(
            dependencies={
                "db_ops": db_operations,
                "api_ops": api_operations
            }
        )
    
    async def get_users(self):
        db_ops = self.get_dependency("db_ops")
        return await db_ops.execute_query("SELECT * FROM users")

# ❌ Bad: Direct dependency injection
@app.get("/users")
async def get_users(db_ops: AsyncDatabaseOperations = Depends()):
    return await db_ops.execute_query("SELECT * FROM users")
```

### **3. Error Handling**

```python
# ✅ Good: Consistent error handling
@app.get("/users")
async def get_users():
    try:
        db_ops = self.get_dependency("db_ops")
        users = await db_ops.execute_query("SELECT * FROM users")
        return self.success_response(data=users)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

# ❌ Bad: Inconsistent error handling
@app.get("/users")
async def get_users():
    users = await db_ops.execute_query("SELECT * FROM users")
    return users  # No error handling
```

### **4. Response Standardization**

```python
# ✅ Good: Standardized responses
@app.get("/users")
async def get_users():
    users = await db_ops.execute_query("SELECT * FROM users")
    return self.success_response(
        data=users,
        message="Users retrieved successfully"
    )

@app.post("/users")
async def create_user(user_data: UserCreate):
    user = await db_ops.execute_insert("users", user_data.dict())
    return self.success_response(
        data=user,
        message="User created successfully"
    )

# ❌ Bad: Inconsistent responses
@app.get("/users")
async def get_users():
    return {"users": users}

@app.post("/users")
async def create_user(user_data: UserCreate):
    return {"message": "User created", "user": user}
```

## 🔗 Integration Examples

### **1. Complete Application Setup**

```python
from fastapi import FastAPI
from api.routes import create_app, route_registry, dependency_container

# Create application
app = create_app(
    title="HeyGen AI API",
    description="Advanced AI-powered video generation and processing API",
    version="1.0.0",
    debug=True
)

# Application is automatically configured with:
# - All routes registered
# - Dependencies injected
# - Middleware configured
# - Lifecycle events set up
```

### **2. Route Development**

```python
# Create new route module
from api.routes.base import BaseRoute, RouteCategory

class AnalyticsRoutes(BaseRoute):
    def __init__(self, db_operations, api_operations):
        super().__init__(
            name="Analytics",
            description="Analytics and reporting operations",
            category=RouteCategory.ANALYTICS,
            tags=["analytics", "reporting"],
            prefix="/analytics",
            dependencies={
                "db_ops": db_operations,
                "api_ops": api_operations
            }
        )
        self._register_routes()
    
    def _register_routes(self):
        @self.router.get("/dashboard")
        @route_metrics
        @require_auth
        async def get_dashboard():
            # Implementation
            pass
        
        @self.router.get("/reports")
        @route_metrics
        @require_auth
        async def get_reports():
            # Implementation
            pass

# Register new routes
analytics_routes = AnalyticsRoutes(db_operations, api_operations)
route_registry.register_route(
    "analytics",
    analytics_routes.router,
    prefix="/api/v1/analytics",
    tags=["analytics", "reporting"]
)
```

### **3. Custom Middleware**

```python
# Create custom middleware
class CustomMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        # Pre-processing
        if scope["type"] == "http":
            # Add custom headers
            scope["headers"].append((b"x-custom-header", b"custom-value"))
        
        await self.app(scope, receive, send)

# Register middleware
app.add_middleware(CustomMiddleware)
```

## 🛠️ Development Workflow

### **1. Route Development Process**

```python
# 1. Create route class
class NewFeatureRoutes(BaseRoute):
    def __init__(self, dependencies):
        super().__init__(
            name="New Feature",
            description="New feature operations",
            category=RouteCategory.UTILS,
            tags=["new-feature"],
            prefix="/new-feature",
            dependencies=dependencies
        )
        self._register_routes()
    
    def _register_routes(self):
        # 2. Define routes
        @self.router.get("/")
        @route_metrics
        @require_auth
        async def get_feature():
            # 3. Implement route logic
            pass

# 4. Register routes
new_feature_routes = NewFeatureRoutes(dependencies)
route_registry.register_route("new-feature", new_feature_routes.router)
```

### **2. Testing Routes**

```python
# Test route structure
def test_route_structure():
    """Test that all routes are properly structured."""
    routes = route_registry.get_all_routes()
    
    for name, route_info in routes.items():
        router = route_info["router"]
        
        # Check that all routes have proper decorators
        for route in router.routes:
            assert hasattr(route, "endpoint"), f"Route {route.name} missing endpoint"
            assert route.tags, f"Route {route.name} missing tags"

# Test route dependencies
def test_route_dependencies():
    """Test that all routes have proper dependencies."""
    routes = route_registry.get_all_routes()
    
    for name, route_info in routes.items():
        router = route_info["router"]
        
        # Check dependencies are available
        for route in router.routes:
            # Verify dependencies are properly injected
            pass
```

### **3. Route Documentation**

```python
# Generate route documentation
def generate_route_docs():
    """Generate comprehensive route documentation."""
    organized_routes = get_route_organization()
    
    docs = "# API Route Documentation\n\n"
    
    for category, info in organized_routes.items():
        docs += f"## {category}: {info['description']}\n\n"
        
        for route_group in info["routes"]:
            docs += f"### {route_group['name']}\n\n"
            docs += f"**Prefix:** `{route_group['prefix']}`\n\n"
            docs += f"**Tags:** {', '.join(route_group['tags'])}\n\n"
            
            for route in route_group["routes"]:
                methods = ", ".join(route["method"])
                docs += f"- **{methods}** `{route['path']}` - {route['name']}\n"
            
            docs += "\n"
    
    return docs
```

## 📊 Summary

### **Key Benefits**

1. **Clear Organization**: Routes are logically grouped by category
2. **Dependency Injection**: Centralized dependency management
3. **Consistent Patterns**: Standardized route patterns and responses
4. **Performance Monitoring**: Built-in metrics and monitoring
5. **Error Handling**: Centralized error management
6. **Documentation**: Automatic API documentation generation
7. **Maintainability**: Easy to understand and modify structure
8. **Scalability**: Modular design supports growth

### **Implementation Checklist**

- [ ] **Setup Route Registry**: Configure central route management
- [ ] **Setup Dependency Container**: Configure dependency injection
- [ ] **Create Base Route Class**: Implement common functionality
- [ ] **Organize Route Categories**: Group routes logically
- [ ] **Implement User Routes**: Create user management routes
- [ ] **Implement Video Routes**: Create video processing routes
- [ ] **Add System Routes**: Create health and monitoring routes
- [ ] **Configure Middleware**: Add request/response processing
- [ ] **Setup Error Handling**: Implement error management
- [ ] **Add Performance Monitoring**: Track route metrics
- [ ] **Generate Documentation**: Create API documentation
- [ ] **Test Route Structure**: Verify organization and dependencies

### **Next Steps**

1. **Integration**: Integrate with existing HeyGen AI services
2. **Customization**: Customize routes for specific needs
3. **Testing**: Implement comprehensive route testing
4. **Documentation**: Generate detailed API documentation
5. **Monitoring**: Set up production monitoring
6. **Optimization**: Optimize route performance

This comprehensive route structure system ensures your HeyGen AI API is well-organized, maintainable, and scalable with clear dependencies and logical organization. 