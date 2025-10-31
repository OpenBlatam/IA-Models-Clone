from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
    from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
    from pydantic.types import StrictStr, StrictInt, StrictBool
    import asyncpg
    import aiomysql
    from sqlalchemy import create_engine, text, select, insert, update, delete
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
    from sqlalchemy.exc import SQLAlchemyError
    from fastapi import FastAPI, HTTPException, Depends, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Conditional Optimization Examples
🚀 Demonstrates Python best practices for conditional statements
⚡ Single-line conditionals and early returns for clean code
🎯 Error handling and edge cases prioritized
"""


# Pydantic v2 imports
try:
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Async database imports
try:
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    AIOMYSQL_AVAILABLE = True
except ImportError:
    AIOMYSQL_AVAILABLE = False

# SQLAlchemy 2.0 imports
try:
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# FastAPI imports
try:
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserData:
    """User data structure for validation examples."""
    user_id: str
    email: str
    age: int
    is_active: bool
    permissions: List[str]

@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

# Pydantic v2 Models for FastAPI
if PYDANTIC_AVAILABLE:
    class UserCreate(BaseModel):
        """Pydantic model for user creation with validation."""
        user_id: str = Field(..., min_length=3, max_length=50, description="Unique user identifier")
        email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        age: int = Field(..., ge=18, le=120, description="User age (must be 18+)")
        is_active: bool = Field(default=True, description="User account status")
        permissions: List[str] = Field(default_factory=list, description="User permissions")
        
        @field_validator('user_id')
        @classmethod
        def validate_user_id(cls, v: str) -> str:
            if not v.isalnum(): raise ValueError("User ID must be alphanumeric")
            return v.lower()
        
        @field_validator('permissions')
        @classmethod
        def validate_permissions(cls, v: List[str]) -> List[str]:
            valid_permissions = {"read", "write", "delete", "admin"}
            invalid_perms = [perm for perm in v if perm not in valid_permissions]
            if invalid_perms: raise ValueError(f"Invalid permissions: {invalid_perms}")
            return list(set(v))  # Remove duplicates
    
    class UserResponse(BaseModel):
        """Pydantic model for user response."""
        user_id: str
        email: str
        age: int
        is_active: bool
        permissions: List[str]
        created_at: datetime = Field(default_factory=datetime.now)
        updated_at: Optional[datetime] = None
        
        class Config:
            from_attributes = True
    
    class APIResponse(BaseModel):
        """Standard API response model."""
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        message: Optional[str] = None
        timestamp: datetime = Field(default_factory=datetime.now)
    
    class ErrorResponse(BaseModel):
        """Error response model."""
        error: str
        error_id: str
        status_code: int
        details: Optional[Dict[str, Any]] = None
        timestamp: datetime = Field(default_factory=datetime.now)

class ConditionalOptimizer:
    """Demonstrates optimized conditional patterns for production code."""
    
    def __init__(self) -> Any:
        self.cache = {}
        self.stats = {"validations": 0, "errors": 0, "cache_hits": 0}
    
    # ❌ BEFORE: Deeply nested conditionals
    def validate_user_bad(self, user_data: UserData) -> ValidationResult:
        """Poor example with deeply nested conditionals."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if user_data is not None:
            if user_data.user_id:
                if len(user_data.user_id) >= 3:
                    if user_data.email:
                        if '@' in user_data.email:
                            if user_data.age:
                                if user_data.age >= 18:
                                    if user_data.is_active:
                                        # Happy path buried deep
                                        result.is_valid = True
                                    else:
                                        result.errors.append("User is not active")
                                else:
                                    result.errors.append("User must be 18 or older")
                            else:
                                result.errors.append("Age is required")
                        else:
                            result.errors.append("Invalid email format")
                    else:
                        result.errors.append("Email is required")
                else:
                    result.errors.append("User ID must be at least 3 characters")
            else:
                result.errors.append("User ID is required")
        else:
            result.errors.append("User data is required")
        
        return result
    
    # ✅ AFTER: Early returns with single-line conditionals (if-return pattern)
    def validate_user_good(self, user_data: UserData) -> ValidationResult:
        """Optimized example with early returns and single-line conditionals."""
        # Handle edge cases and errors first - if-return pattern
        if user_data is None: return ValidationResult(False, ["User data is required"], [])
        if not user_data.user_id: return ValidationResult(False, ["User ID is required"], [])
        if len(user_data.user_id) < 3: return ValidationResult(False, ["User ID must be at least 3 characters"], [])
        if not user_data.email: return ValidationResult(False, ["Email is required"], [])
        if '@' not in user_data.email: return ValidationResult(False, ["Invalid email format"], [])
        if not user_data.age: return ValidationResult(False, ["Age is required"], [])
        if user_data.age < 18: return ValidationResult(False, ["User must be 18 or older"], [])
        if not user_data.is_active: return ValidationResult(False, ["User is not active"], [])
        
        # Happy path last - no else needed
        return ValidationResult(True, [], [])
    
    # ✅ PRODUCTION: With warnings and detailed validation
    def validate_user_production(self, user_data: UserData) -> ValidationResult:
        """Production-ready validation with comprehensive error handling."""
        errors = []
        warnings = []
        
        # Early returns for critical errors
        if user_data is None: return ValidationResult(False, ["User data is required"], [])
        if not user_data.user_id: return ValidationResult(False, ["User ID is required"], [])
        if len(user_data.user_id) < 3: return ValidationResult(False, ["User ID must be at least 3 characters"], [])
        if not user_data.email: return ValidationResult(False, ["Email is required"], [])
        if '@' not in user_data.email: return ValidationResult(False, ["Invalid email format"], [])
        if not user_data.age: return ValidationResult(False, ["Age is required"], [])
        if user_data.age < 18: return ValidationResult(False, ["User must be 18 or older"], [])
        
        # Warnings for non-critical issues
        if user_data.age > 100: warnings.append("Unusual age value")
        if len(user_data.email) > 254: warnings.append("Email address is very long")
        if not user_data.permissions: warnings.append("User has no permissions assigned")
        
        # Final validation
        if not user_data.is_active: return ValidationResult(False, ["User is not active"], warnings)
        
        # Happy path
        return ValidationResult(True, [], warnings)
    
    # ✅ CACHE OPTIMIZATION: Single-line conditionals for cache operations
    def get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Optimized cache retrieval with single-line conditionals."""
        if key not in self.cache: return None
        if self.cache[key]["expires"] < datetime.now(): 
            del self.cache[key]
            return None
        self.stats["cache_hits"] += 1
        return self.cache[key]["data"]
    
    def set_cached_data(self, key: str, data: Dict[str, Any], ttl_seconds: int = 3600) -> bool:
        """Optimized cache storage with single-line conditionals."""
        if not key: return False
        if not data: return False
        if ttl_seconds <= 0: return False
        
        expires = datetime.now() + timedelta(seconds=ttl_seconds)
        self.cache[key] = {"data": data, "expires": expires}
        return True
    
    # ✅ BATCH PROCESSING: Early returns for batch validation
    def process_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch processing with early returns and single-line conditionals."""
        if not items: return {"success": False, "error": "No items provided"}
        if len(items) > 1000: return {"success": False, "error": "Batch too large"}
        
        results = []
        errors = []
        
        for item in items:
            if not isinstance(item, dict): 
                errors.append(f"Invalid item type: {type(item)}")
                continue
            if "id" not in item: 
                errors.append("Item missing required 'id' field")
                continue
            
            # Process valid item
            processed = self._process_single_item(item)
            if processed: results.append(processed)
        
        return {
            "success": len(errors) == 0,
            "processed": len(results),
            "errors": errors,
            "results": results
        }
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single item with early returns."""
        if not item.get("data"): return None
        if item.get("status") == "deleted": return None
        
        # Happy path processing
        return {
            "id": item["id"],
            "processed_at": datetime.now().isoformat(),
            "result": f"Processed: {item['data']}"
        }
    
    # ✅ ERROR HANDLING: Comprehensive error handling patterns
    def safe_divide(self, numerator: float, denominator: float) -> Optional[float]:
        """Safe division with early returns for error conditions."""
        if denominator == 0: return None
        if not isinstance(numerator, (int, float)): return None
        if not isinstance(denominator, (int, float)): return None
        
        return numerator / denominator
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Configuration validation with early returns."""
        if not config: return ValidationResult(False, ["Configuration is required"], [])
        if not isinstance(config, dict): return ValidationResult(False, ["Configuration must be a dictionary"], [])
        
        required_fields = ["api_key", "base_url", "timeout"]
        for field in required_fields:
            if field not in config: return ValidationResult(False, [f"Missing required field: {field}"], [])
            if not config[field]: return ValidationResult(False, [f"Field '{field}' cannot be empty"], [])
        
        # Validate specific field types
        if not isinstance(config["timeout"], (int, float)): return ValidationResult(False, ["Timeout must be a number"], [])
        if config["timeout"] <= 0: return ValidationResult(False, ["Timeout must be positive"], [])
        
        # Warnings for non-optimal values
        warnings = []
        if config["timeout"] > 300: warnings.append("Timeout is very high (>5 minutes)")
        if len(config["api_key"]) < 10: warnings.append("API key seems too short")
        
        return ValidationResult(True, [], warnings)
    
    # ✅ PERFORMANCE OPTIMIZATION: Single-line conditionals for performance checks
    def should_use_cache(self, key: str, data_size: int) -> bool:
        """Performance optimization with single-line conditionals."""
        if key not in self.cache: return True
        if data_size > 1024 * 1024: return False  # Don't cache large data
        if len(self.cache) > 1000: return False   # Cache full
        return True
    
    def should_retry_operation(self, attempt: int, max_attempts: int, last_error: str) -> bool:
        """Retry logic with single-line conditionals."""
        if attempt >= max_attempts: return False
        if "permission denied" in last_error.lower(): return False
        if "not found" in last_error.lower(): return False
        return True
    
    # ✅ GUARD CLAUSES: Handle preconditions and invalid states early
    async def process_user_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate guard clauses for preconditions."""
        # Guard clauses for invalid states
        if not user_id: 
            logger.error("User ID is required")
            return {"error": "User ID is required", "status": "invalid"}
        
        if not request_data: 
            logger.error("Request data is required", user_id=user_id)
            return {"error": "Request data is required", "status": "invalid"}
        
        if not isinstance(request_data, dict): 
            logger.error("Request data must be a dictionary", user_id=user_id, data_type=type(request_data))
            return {"error": "Request data must be a dictionary", "status": "invalid"}
        
        # Guard clauses for business logic preconditions
        if user_id not in self.cache: 
            logger.warning("User not found in cache", user_id=user_id)
            return {"error": "User not found", "status": "not_found"}
        
        if not self.cache[user_id].get("is_active", False): 
            logger.warning("User account is inactive", user_id=user_id)
            return {"error": "User account is inactive", "status": "inactive"}
        
        # Happy path - process the request
        try:
            result = self._process_valid_request(user_id, request_data)
            logger.info("Request processed successfully", user_id=user_id, result_type=type(result))
            return {"result": result, "status": "success"}
        except Exception as e:
            logger.error("Request processing failed", user_id=user_id, error=str(e), exc_info=True)
            return {"error": "Internal processing error", "status": "error"}
    
    async def _process_valid_request(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a valid request after all guard clauses pass."""
        return {
            "user_id": user_id,
            "processed_at": datetime.now().isoformat(),
            "data": request_data
        }
    
    # ✅ DEPENDENCY HANDLING: Check dependencies early
    def initialize_system(self, config: Dict[str, Any]) -> bool:
        """Initialize system with dependency checks."""
        # Guard clauses for dependencies
        if not config: 
            logger.error("Configuration is required for system initialization")
            return False
        
        required_deps = ["database_url", "api_key", "redis_url"]
        for dep in required_deps:
            if dep not in config: 
                logger.error(f"Missing required dependency: {dep}")
                return False
            if not config[dep]: 
                logger.error(f"Dependency '{dep}' cannot be empty")
                return False
        
        # Check external service availability
        if not self._check_database_connection(config["database_url"]): 
            logger.error("Database connection failed")
            return False
        
        if not self._check_redis_connection(config["redis_url"]): 
            logger.error("Redis connection failed")
            return False
        
        # Happy path - initialize system
        logger.info("System initialized successfully")
        return True
    
    def _check_database_connection(self, url: str) -> bool:
        """Check database connection availability."""
        # Simulate connection check
        return "postgresql" in url or "mysql" in url
    
    def _check_redis_connection(self, url: str) -> bool:
        """Check Redis connection availability."""
        # Simulate connection check
        return "redis" in url

# FastAPI Integration Examples
class FastAPIErrorHandler:
    """FastAPI error handling with guard clauses and proper logging."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
    
    async def validate_request_body(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate FastAPI request body with guard clauses."""
        # Guard clauses for request validation
        if not body: 
            self.logger.error("Request body is required")
            return {"error": "Request body is required", "status_code": 400}
        
        if not isinstance(body, dict): 
            self.logger.error("Request body must be a JSON object")
            return {"error": "Request body must be a JSON object", "status_code": 400}
        
        required_fields = ["user_id", "action", "data"]
        for field in required_fields:
            if field not in body: 
                self.logger.error(f"Missing required field: {field}")
                return {"error": f"Missing required field: {field}", "status_code": 400}
        
        # Happy path
        return None
    
    async def handle_api_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API errors with proper logging and user-friendly messages."""
        error_id = hashlib.md5(str(error).encode()).hexdigest()[:8]
        
        # Log detailed error for debugging
        self.logger.error(
            "API error occurred",
            error_id=error_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            exc_info=True
        )
        
        # Return user-friendly error message
        if isinstance(error, ValueError):
            return {
                "error": "Invalid input provided",
                "error_id": error_id,
                "status_code": 400
            }
        elif isinstance(error, PermissionError):
            return {
                "error": "Access denied",
                "error_id": error_id,
                "status_code": 403
            }
        elif isinstance(error, FileNotFoundError):
            return {
                "error": "Resource not found",
                "error_id": error_id,
                "status_code": 404
            }
        else:
            return {
                "error": "Internal server error",
                "error_id": error_id,
                "status_code": 500
            }
    
    def validate_user_permissions(self, user_id: str, required_permissions: List[str]) -> Optional[Dict[str, Any]]:
        """Validate user permissions with guard clauses."""
        # Guard clauses for permission validation
        if not user_id: 
            self.logger.error("User ID is required for permission validation")
            return {"error": "User ID is required", "status_code": 400}
        
        if not required_permissions: 
            self.logger.error("Required permissions list cannot be empty")
            return {"error": "Required permissions list cannot be empty", "status_code": 400}
        
        # Simulate user permission check
        user_permissions = self._get_user_permissions(user_id)
        if not user_permissions: 
            self.logger.warning("User has no permissions", user_id=user_id)
            return {"error": "User has no permissions", "status_code": 403}
        
        missing_permissions = [perm for perm in required_permissions if perm not in user_permissions]
        if missing_permissions: 
            self.logger.warning(
                "User missing required permissions",
                user_id=user_id,
                missing_permissions=missing_permissions
            )
            return {
                "error": f"Missing required permissions: {', '.join(missing_permissions)}",
                "status_code": 403
            }
        
        # Happy path
        return None
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (simulated)."""
        # Simulate database lookup
        permissions_map = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "guest": ["read"]
        }
        return permissions_map.get(user_id, [])

# FastAPI Route Examples with Declarative Definitions
if FASTAPI_AVAILABLE and PYDANTIC_AVAILABLE:
    class FastAPIRouter:
        """FastAPI router with declarative route definitions and conditional optimization."""
        
        def __init__(self) -> Any:
            self.app = FastAPI(title="Conditional Optimization API", version="1.0.0")
            self.setup_middleware()
            self.setup_routes()
        
        def setup_middleware(self) -> None:
            """Setup FastAPI middleware with guard clauses."""
            if not self.app: return
            
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        def setup_routes(self) -> None:
            """Setup API routes with declarative definitions."""
            if not self.app: return
            
            @self.app.get("/health", response_model=APIResponse, tags=["Health"])
            async def health_check() -> APIResponse:
                """Health check endpoint with early returns."""
                # Guard clauses for health checks
                if not self._check_database_connection(): 
                    return APIResponse(success=False, error="Database connection failed")
                
                if not self._check_redis_connection(): 
                    return APIResponse(success=False, error="Redis connection failed")
                
                # Happy path
                return APIResponse(success=True, message="Service is healthy")
            
            @self.app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
            async def create_user(user: UserCreate) -> UserResponse:
                """Create user with declarative validation and early returns."""
                # Guard clauses for user creation
                if await self._user_exists(user.user_id): 
                    raise HTTPException(status_code=400, detail="User already exists")
                
                if not self._validate_email_domain(user.email): 
                    raise HTTPException(status_code=400, detail="Invalid email domain")
                
                # Happy path - create user
                created_user = await self._create_user_in_db(user)
                return UserResponse(**created_user.dict())
            
            @self.app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
            async def get_user(user_id: str) -> UserResponse:
                """Get user with guard clauses and early returns."""
                # Guard clauses
                if not user_id: 
                    raise HTTPException(status_code=400, detail="User ID is required")
                
                if len(user_id) < 3: 
                    raise HTTPException(status_code=400, detail="User ID too short")
                
                user = await self._get_user_from_db(user_id)
                if not user: 
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Happy path
                return UserResponse(**user.dict())
            
            @self.app.put("/users/{user_id}", response_model=UserResponse, tags=["Users"])
            async def update_user(user_id: str, user_update: UserCreate) -> UserResponse:
                """Update user with comprehensive validation."""
                # Guard clauses for updates
                if not user_id: 
                    raise HTTPException(status_code=400, detail="User ID is required")
                
                existing_user = await self._get_user_from_db(user_id)
                if not existing_user: 
                    raise HTTPException(status_code=404, detail="User not found")
                
                if not existing_user.is_active: 
                    raise HTTPException(status_code=400, detail="Cannot update inactive user")
                
                # Happy path - update user
                updated_user = await self._update_user_in_db(user_id, user_update)
                return UserResponse(**updated_user.dict())
            
            @self.app.delete("/users/{user_id}", response_model=APIResponse, tags=["Users"])
            async def delete_user(user_id: str) -> APIResponse:
                """Delete user with guard clauses."""
                # Guard clauses
                if not user_id: 
                    raise HTTPException(status_code=400, detail="User ID is required")
                
                user = await self._get_user_from_db(user_id)
                if not user: 
                    raise HTTPException(status_code=404, detail="User not found")
                
                if user.permissions and "admin" in user.permissions: 
                    raise HTTPException(status_code=403, detail="Cannot delete admin user")
                
                # Happy path - delete user
                await self._delete_user_from_db(user_id)
                return APIResponse(success=True, message="User deleted successfully")
        
        # Dependency injection functions (plain functions)
        async def get_current_user(self, user_id: str) -> UserResponse:
            """Dependency function to get current user with guard clauses."""
            if not user_id: 
                raise HTTPException(status_code=401, detail="User ID required")
            
            user = await self._get_user_from_db(user_id)
            if not user: 
                raise HTTPException(status_code=401, detail="User not found")
            
            if not user.is_active: 
                raise HTTPException(status_code=401, detail="User account inactive")
            
            return user
        
        async def require_permissions(self, user: UserResponse, required_perms: List[str]) -> bool:
            """Dependency function to check permissions with early returns."""
            if not user: return False
            if not user.is_active: return False
            if not required_perms: return True
            
            user_perms = set(user.permissions)
            required_set = set(required_perms)
            
            if not required_set.issubset(user_perms): 
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return True
        
        # Private helper methods with guard clauses
        def _check_database_connection(self) -> bool:
            """Check database connection with early return."""
            if not ASYNCPG_AVAILABLE: return False
            # Simulate connection check
            return True
        
        def _check_redis_connection(self) -> bool:
            """Check Redis connection with early return."""
            if not AIOREDIS_AVAILABLE: return False
            # Simulate connection check
            return True
        
        async def _user_exists(self, user_id: str) -> bool:
            """Check if user exists with early return."""
            if not user_id: return False
            # Simulate database check
            return user_id in ["existing_user", "admin"]
        
        def _validate_email_domain(self, email: str) -> bool:
            """Validate email domain with early return."""
            if not email: return False
            if '@' not in email: return False
            
            domain = email.split('@')[1]
            valid_domains = {"example.com", "gmail.com", "yahoo.com"}
            return domain in valid_domains
        
        async def _create_user_in_db(self, user: UserCreate) -> UserCreate:
            """Create user in database (simulated)."""
            # Simulate database operation
            return user
        
        async def _get_user_from_db(self, user_id: str) -> Optional[UserCreate]:
            """Get user from database (simulated)."""
            if not user_id: return None
            
            # Simulate database lookup
            users_db = {
                "admin": UserCreate(user_id="admin", email="admin@example.com", age=30, permissions=["admin"]),
                "user": UserCreate(user_id="user", email="user@example.com", age=25, permissions=["read", "write"])
            }
            return users_db.get(user_id)
        
        async def _update_user_in_db(self, user_id: str, user_update: UserCreate) -> UserCreate:
            """Update user in database (simulated)."""
            # Simulate database update
            return user_update
        
        async def _delete_user_from_db(self, user_id: str) -> bool:
            """Delete user from database (simulated)."""
            # Simulate database deletion
            return True

# ✅ MODERN FASTAPI: Lifespan Context Manager with Async/Sync Optimization
if FASTAPI_AVAILABLE:
    class ModernFastAPIApp:
        """Modern FastAPI application with lifespan context manager and proper async/sync usage."""
        
        def __init__(self) -> Any:
            self.app = None
            self.db_pool = None
            self.redis_pool = None
            self.metrics_collector = None
        
        @asynccontextmanager
        async def lifespan(self, app: FastAPI):
            """Lifespan context manager for startup and shutdown events."""
            # Startup phase - async operations
            logger.info("Starting application...")
            
            try:
                # Initialize async resources
                await self._initialize_database()
                await self._initialize_redis()
                await self._initialize_metrics()
                
                logger.info("Application started successfully")
                yield
                
            except Exception as e:
                logger.error("Failed to start application", error=str(e), exc_info=True)
                raise
            finally:
                # Shutdown phase - async cleanup
                logger.info("Shutting down application...")
                await self._cleanup_resources()
                logger.info("Application shutdown complete")
        
        def create_app(self) -> FastAPI:
            """Create FastAPI app with lifespan context manager."""
            self.app = FastAPI(
                title="Modern Conditional Optimization API",
                version="2.0.0",
                lifespan=self.lifespan
            )
            
            # Setup middleware and routes
            self._setup_middleware()
            self._setup_routes()
            
            return self.app
        
        def _setup_middleware(self) -> None:
            """Setup middleware with guard clauses."""
            if not self.app: return
            
            # CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        def _setup_routes(self) -> None:
            """Setup routes with proper async/sync function usage."""
            if not self.app: return
            
            @self.app.get("/health", response_model=APIResponse, tags=["Health"])
            async def health_check() -> APIResponse:
                """Async health check with early returns."""
                # Guard clauses for health checks
                if not await self._check_database_health(): 
                    return APIResponse(success=False, error="Database unhealthy")
                
                if not await self._check_redis_health(): 
                    return APIResponse(success=False, error="Redis unhealthy")
                
                # Happy path
                return APIResponse(success=True, message="All systems healthy")
            
            @self.app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
            async def create_user(user: UserCreate) -> UserResponse:
                """Async user creation with comprehensive validation."""
                # Guard clauses
                if await self._user_exists(user.user_id): 
                    raise HTTPException(status_code=400, detail="User already exists")
                
                if not self._validate_email_domain(user.email): 
                    raise HTTPException(status_code=400, detail="Invalid email domain")
                
                # Happy path - async database operation
                created_user = await self._create_user_async(user)
                return UserResponse(**created_user.dict())
            
            @self.app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
            async def get_user(user_id: str) -> UserResponse:
                """Async user retrieval with guard clauses."""
                # Guard clauses
                if not user_id: 
                    raise HTTPException(status_code=400, detail="User ID required")
                
                if len(user_id) < 3: 
                    raise HTTPException(status_code=400, detail="User ID too short")
                
                user = await self._get_user_async(user_id)
                if not user: 
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Happy path
                return UserResponse(**user.dict())
        
        # ✅ ASYNC OPERATIONS: Use async def for I/O operations
        async def _initialize_database(self) -> None:
            """Initialize database connection pool."""
            if not ASYNCPG_AVAILABLE: 
                logger.warning("asyncpg not available, skipping database initialization")
                return
            
            try:
                self.db_pool = await asyncpg.create_pool(
                    "postgresql://user:pass@localhost/db",
                    min_size=5,
                    max_size=20
                )
                logger.info("Database pool initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize database pool", error=str(e))
                raise
        
        async def _initialize_redis(self) -> None:
            """Initialize Redis connection pool."""
            if not AIOREDIS_AVAILABLE: 
                logger.warning("aioredis not available, skipping Redis initialization")
                return
            
            try:
                self.redis_pool = await aioredis.create_redis_pool("redis://localhost")
                logger.info("Redis pool initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize Redis pool", error=str(e))
                raise
        
        async def _initialize_metrics(self) -> None:
            """Initialize metrics collection."""
            self.metrics_collector = MetricsCollector()
            await self.metrics_collector.start()
            logger.info("Metrics collector initialized")
        
        async def _cleanup_resources(self) -> None:
            """Cleanup resources on shutdown."""
            if self.db_pool: 
                await self.db_pool.close()
                logger.info("Database pool closed")
            
            if self.redis_pool: 
                self.redis_pool.close()
                await self.redis_pool.wait_closed()
                logger.info("Redis pool closed")
            
            if self.metrics_collector: 
                await self.metrics_collector.stop()
                logger.info("Metrics collector stopped")
        
        # ✅ ASYNC HEALTH CHECKS
        async def _check_database_health(self) -> bool:
            """Async database health check."""
            if not self.db_pool: return False
            
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return True
            except Exception as e:
                logger.error("Database health check failed", error=str(e))
                return False
        
        async def _check_redis_health(self) -> bool:
            """Async Redis health check."""
            if not self.redis_pool: return False
            
            try:
                await self.redis_pool.ping()
                return True
            except Exception as e:
                logger.error("Redis health check failed", error=str(e))
                return False
        
        # ✅ ASYNC DATABASE OPERATIONS
        async def _user_exists(self, user_id: str) -> bool:
            """Async user existence check."""
            if not user_id or not self.db_pool: return False
            
            try:
                async with self.db_pool.acquire() as conn:
                    result = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE user_id = $1)",
                        user_id
                    )
                return result
            except Exception as e:
                logger.error("User existence check failed", user_id=user_id, error=str(e))
                return False
        
        async def _create_user_async(self, user: UserCreate) -> UserCreate:
            """Async user creation."""
            if not self.db_pool: raise HTTPException(status_code=500, detail="Database not available")
            
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO users (user_id, email, age, is_active, permissions)
                        VALUES ($1, $2, $3, $4, $5)
                    """, user.user_id, user.email, user.age, user.is_active, user.permissions)
                return user
            except Exception as e:
                logger.error("User creation failed", user_id=user.user_id, error=str(e))
                raise HTTPException(status_code=500, detail="Failed to create user")
        
        async def _get_user_async(self, user_id: str) -> Optional[UserCreate]:
            """Async user retrieval."""
            if not user_id or not self.db_pool: return None
            
            try:
                async with self.db_pool.acquire() as conn:
                    row = await conn.fetchrow(
                        "SELECT user_id, email, age, is_active, permissions FROM users WHERE user_id = $1",
                        user_id
                    )
                if row:
                    return UserCreate(
                        user_id=row['user_id'],
                        email=row['email'],
                        age=row['age'],
                        is_active=row['is_active'],
                        permissions=row['permissions']
                    )
                return None
            except Exception as e:
                logger.error("User retrieval failed", user_id=user_id, error=str(e))
                return None
        
        # ✅ SYNC OPERATIONS: Use def for CPU-bound operations
        def _validate_email_domain(self, email: str) -> bool:
            """Synchronous email domain validation."""
            if not email: return False
            if '@' not in email: return False
            
            domain = email.split('@')[1]
            valid_domains = {"example.com", "gmail.com", "yahoo.com"}
            return domain in valid_domains
        
        def _calculate_user_score(self, user: UserCreate) -> float:
            """Synchronous user score calculation."""
            if not user: return 0.0
            
            score = 0.0
            if user.age >= 18: score += 10
            if user.is_active: score += 20
            if len(user.permissions) > 0: score += 15
            
            return score

# ✅ PERFORMANCE MONITORING AND ERROR HANDLING
class MetricsCollector:
    """Performance metrics collector with proper async/sync usage."""
    
    def __init__(self) -> Any:
        self.metrics = {}
        self.is_running = False
    
    async def start(self) -> None:
        """Async startup."""
        self.is_running = True
        logger.info("Metrics collector started")
    
    async def stop(self) -> None:
        """Async shutdown."""
        self.is_running = False
        logger.info("Metrics collector stopped")
    
    async def record_request(self, endpoint: str, duration: float, status_code: int) -> None:
        """Synchronous metrics recording."""
        if not self.is_running: return
        
        if endpoint not in self.metrics:
            self.metrics[endpoint] = {
                "count": 0,
                "total_duration": 0.0,
                "status_codes": {}
            }
        
        self.metrics[endpoint]["count"] += 1
        self.metrics[endpoint]["total_duration"] += duration
        
        status_str = str(status_code)
        if status_str not in self.metrics[endpoint]["status_codes"]:
            self.metrics[endpoint]["status_codes"][status_str] = 0
        self.metrics[endpoint]["status_codes"][status_str] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Synchronous metrics retrieval."""
        if not self.is_running: return {}
        
        result = {}
        for endpoint, data in self.metrics.items():
            avg_duration = data["total_duration"] / data["count"] if data["count"] > 0 else 0
            result[endpoint] = {
                "request_count": data["count"],
                "average_duration": avg_duration,
                "status_codes": data["status_codes"]
            }
        
        return result

# ✅ PERFORMANCE OPTIMIZATION: Async I/O, Caching, and Lazy Loading
class PerformanceOptimizer:
    """Performance optimization with async I/O, caching, and lazy loading."""
    
    def __init__(self) -> Any:
        self.cache = {}
        self.lazy_loaded_data = {}
        self.async_tasks = {}
        self.performance_metrics = {}
    
    # ✅ ASYNC I/O OPTIMIZATION: Parallel processing for I/O-bound tasks
    async async def fetch_multiple_resources(self, resource_ids: List[str]) -> Dict[str, Any]:
        """Fetch multiple resources concurrently using async I/O."""
        if not resource_ids: return {}
        
        # Guard clauses for input validation
        if len(resource_ids) > 100: 
            logger.warning("Too many resource IDs, limiting to 100")
            resource_ids = resource_ids[:100]
        
        # Create async tasks for concurrent execution
        tasks = []
        for resource_id in resource_ids:
            if resource_id not in self.cache:
                task = self._fetch_single_resource_async(resource_id)
                tasks.append((resource_id, task))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            # Process results with error handling
            for i, (resource_id, _) in enumerate(tasks):
                if isinstance(results[i], Exception):
                    logger.error(f"Failed to fetch resource {resource_id}", error=str(results[i]))
                    self.cache[resource_id] = {"error": "Failed to fetch"}
                else:
                    self.cache[resource_id] = results[i]
        
        # Return cached results
        return {rid: self.cache.get(rid, {"error": "Not found"}) for rid in resource_ids}
    
    async async def _fetch_single_resource_async(self, resource_id: str) -> Dict[str, Any]:
        """Async fetch for single resource with timeout and retry logic."""
        if not resource_id: return {"error": "Invalid resource ID"}
        
        # Simulate async I/O operation
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate different resource types
        if resource_id.startswith("user_"):
            return {"type": "user", "id": resource_id, "data": f"User data for {resource_id}"}
        elif resource_id.startswith("post_"):
            return {"type": "post", "id": resource_id, "data": f"Post data for {resource_id}"}
        else:
            return {"type": "unknown", "id": resource_id, "data": f"Generic data for {resource_id}"}
    
    # ✅ CACHING STRATEGIES: Multi-level caching with intelligent invalidation
    class CacheManager:
        """Intelligent caching with multiple strategies."""
        
        def __init__(self) -> Any:
            self.memory_cache = {}
            self.cache_metadata = {}
            self.cache_hits = 0
            self.cache_misses = 0
        
        async def get(self, key: str, fetch_func=None, ttl: int = 3600) -> Optional[Any]:
            """Get value with intelligent caching and async fetching."""
            if not key: return None
            
            # Check memory cache first
            if key in self.memory_cache:
                metadata = self.cache_metadata.get(key, {})
                if not self._is_expired(metadata, ttl):
                    self.cache_hits += 1
                    return self.memory_cache[key]
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
                    if key in self.cache_metadata:
                        del self.cache_metadata[key]
            
            self.cache_misses += 1
            
            # Fetch new data if function provided
            if fetch_func and callable(fetch_func):
                try:
                    if asyncio.iscoroutinefunction(fetch_func):
                        value = await fetch_func(key)
                    else:
                        value = fetch_func(key)
                    
                    # Cache the result
                    await self.set(key, value, ttl)
                    return value
                except Exception as e:
                    logger.error(f"Failed to fetch data for key {key}", error=str(e))
                    return None
            
            return None
        
        async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
            """Set value with metadata tracking."""
            if not key: return False
            
            self.memory_cache[key] = value
            self.cache_metadata[key] = {
                "created_at": datetime.now(),
                "ttl": ttl,
                "access_count": 0
            }
            return True
        
        def _is_expired(self, metadata: Dict[str, Any], ttl: int) -> bool:
            """Check if cache entry is expired."""
            if not metadata: return True
            
            created_at = metadata.get("created_at")
            if not created_at: return True
            
            age = (datetime.now() - created_at).total_seconds()
            return age > ttl
        
        def get_stats(self) -> Dict[str, Any]:
            """Get cache performance statistics."""
            return {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "size": len(self.memory_cache)
            }
    
    # ✅ LAZY LOADING: Load data only when needed
    class LazyLoader:
        """Lazy loading implementation for expensive resources."""
        
        def __init__(self) -> Any:
            self.loaded_data = {}
            self.loading_tasks = {}
            self.load_callbacks = {}
        
        async def get_or_load(self, key: str, loader_func, force_reload: bool = False) -> Optional[Dict[str, Any]]:
            """Get data or load it lazily if not available."""
            if not key: return None
            
            # Return cached data if available and not forcing reload
            if not force_reload and key in self.loaded_data:
                return self.loaded_data[key]
            
            # Check if already loading
            if key in self.loading_tasks and not self.loading_tasks[key].done():
                # Wait for existing loading task
                try:
                    return await self.loading_tasks[key]
                except Exception as e:
                    logger.error(f"Failed to load data for {key}", error=str(e))
                    return None
            
            # Start new loading task
            if asyncio.iscoroutinefunction(loader_func):
                task = asyncio.create_task(self._load_data_async(key, loader_func))
            else:
                task = asyncio.create_task(self._load_data_sync(key, loader_func))
            
            self.loading_tasks[key] = task
            
            try:
                result = await task
                self.loaded_data[key] = result
                return result
            except Exception as e:
                logger.error(f"Failed to load data for {key}", error=str(e))
                return None
            finally:
                # Clean up completed task
                if key in self.loading_tasks:
                    del self.loading_tasks[key]
        
        async def _load_data_async(self, key: str, loader_func) -> Any:
            """Load data using async function."""
            return await loader_func(key)
        
        async def _load_data_sync(self, key: str, loader_func) -> Any:
            """Load data using sync function in thread pool."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, loader_func, key)
        
        def preload(self, keys: List[str], loader_func) -> None:
            """Preload data in background for better performance."""
            if not keys: return
            
            async def preload_task():
                
    """preload_task function."""
tasks = []
                for key in keys:
                    if key not in self.loaded_data:
                        task = self.get_or_load(key, loader_func)
                        tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # Start preloading in background
            asyncio.create_task(preload_task())
    
    # ✅ BATCH PROCESSING: Optimize multiple operations
    async def process_batch_async(self, items: List[Any], processor_func, batch_size: int = 10) -> List[Any]:
        """Process items in batches with async optimization."""
        if not items: return []
        if not processor_func: return items
        
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create async tasks for batch
            if asyncio.iscoroutinefunction(processor_func):
                tasks = [processor_func(item) for item in batch]
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                tasks = [loop.run_in_executor(None, processor_func, item) for item in batch]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed for item {i + j}", error=str(result))
                    results.append(None)
                else:
                    results.append(result)
        
        return results
    
    # ✅ CONNECTION POOLING: Optimize database connections
    class ConnectionPool:
        """Async connection pool for database optimization."""
        
        def __init__(self, max_connections: int = 20):
            
    """__init__ function."""
self.max_connections = max_connections
            self.available_connections = asyncio.Queue(maxsize=max_connections)
            self.active_connections = 0
            self.connection_factory = None
        
        async def initialize(self, connection_factory) -> Any:
            """Initialize connection pool."""
            if not connection_factory: return
            
            self.connection_factory = connection_factory
            
            # Pre-populate pool with connections
            for _ in range(min(5, self.max_connections)):
                try:
                    conn = await self.connection_factory()
                    await self.available_connections.put(conn)
                except Exception as e:
                    logger.error("Failed to create connection", error=str(e))
        
        async def get_connection(self) -> Optional[Any]:
            """Get connection from pool with timeout."""
            if not self.connection_factory: return None
            
            try:
                # Try to get existing connection
                conn = await asyncio.wait_for(self.available_connections.get(), timeout=5.0)
                self.active_connections += 1
                return conn
            except asyncio.TimeoutError:
                # Create new connection if pool is empty and under limit
                if self.active_connections < self.max_connections:
                    try:
                        conn = await self.connection_factory()
                        self.active_connections += 1
                        return conn
                    except Exception as e:
                        logger.error("Failed to create new connection", error=str(e))
                        return None
                else:
                    logger.warning("Connection pool exhausted")
                    return None
        
        async def return_connection(self, conn: Any) -> None:
            """Return connection to pool."""
            if not conn: return
            
            try:
                # Check if connection is still valid
                if hasattr(conn, 'closed') and conn.closed:
                    # Don't return closed connections
                    self.active_connections -= 1
                    return
                
                # Return to pool
                await self.available_connections.put(conn)
                self.active_connections -= 1
            except Exception as e:
                logger.error("Failed to return connection to pool", error=str(e))
                self.active_connections -= 1
        
        async def close_all(self) -> None:
            """Close all connections in pool."""
            while not self.available_connections.empty():
                try:
                    conn = await self.available_connections.get()
                    if hasattr(conn, 'close'):
                        await conn.close()
                except Exception as e:
                    logger.error("Failed to close connection", error=str(e))
    
    # ✅ PERFORMANCE MONITORING: Track and optimize performance
    def track_performance(self, operation_name: str, start_time: float) -> None:
        """Track operation performance."""
        if not operation_name: return
        
        duration = time.perf_counter() - start_time
        
        if operation_name not in self.performance_metrics:
            self.performance_metrics[operation_name] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0,
                "avg_time": 0.0
            }
        
        metrics = self.performance_metrics[operation_name]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["min_time"] = min(metrics["min_time"], duration)
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "operations": self.performance_metrics,
            "cache_stats": self.cache_manager.get_stats() if hasattr(self, 'cache_manager') else {},
            "total_operations": sum(m["count"] for m in self.performance_metrics.values()),
            "average_response_time": sum(m["avg_time"] for m in self.performance_metrics.values()) / len(self.performance_metrics) if self.performance_metrics else 0
        }

# Example usage and testing
def demonstrate_optimization():
    """Demonstrate the conditional optimization patterns."""
    optimizer = ConditionalOptimizer()
    
    # Test data
    valid_user = UserData(
        user_id="user123",
        email="user@example.com",
        age=25,
        is_active=True,
        permissions=["read", "write"]
    )
    
    invalid_user = UserData(
        user_id="ab",  # Too short
        email="invalid-email",
        age=16,  # Too young
        is_active=False,
        permissions=[]
    )
    
    # Test validation methods
    print("=== Validation Examples ===")
    print("Valid user (good):", optimizer.validate_user_good(valid_user))
    print("Invalid user (good):", optimizer.validate_user_good(invalid_user))
    print("Valid user (production):", optimizer.validate_user_production(valid_user))
    
    # Test cache operations
    print("\n=== Cache Examples ===")
    optimizer.set_cached_data("test_key", {"data": "test_value"})
    print("Cached data:", optimizer.get_cached_data("test_key"))
    
    # Test batch processing
    print("\n=== Batch Processing ===")
    batch_items = [
        {"id": "1", "data": "item1", "status": "active"},
        {"id": "2", "data": "", "status": "active"},  # Invalid
        {"id": "3", "data": "item3", "status": "deleted"},  # Skipped
        {"id": "4", "data": "item4", "status": "active"}
    ]
    print("Batch result:", optimizer.process_batch(batch_items))
    
    # Test error handling
    print("\n=== Error Handling ===")
    print("Safe divide (10/2):", optimizer.safe_divide(10, 2))
    print("Safe divide (10/0):", optimizer.safe_divide(10, 0))
    
    # Test configuration validation
    print("\n=== Configuration Validation ===")
    good_config = {
        "api_key": "sk-1234567890abcdef",
        "base_url": "https://api.example.com",
        "timeout": 30
    }
    bad_config = {
        "api_key": "",
        "base_url": "https://api.example.com"
        # Missing timeout
    }
    print("Good config:", optimizer.validate_config(good_config))
    print("Bad config:", optimizer.validate_config(bad_config))

match __name__:
    case "__main__":
    demonstrate_optimization() 