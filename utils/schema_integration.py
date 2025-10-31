from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic, Callable
from datetime import datetime
import asyncio
from functools import wraps
import time
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError, BaseModel
import structlog
from .pydantic_schemas import (
from .schema_validators import validation_registry, ValidationResult
from .http_response_models import SuccessResponse, ErrorResponse, ListResponse
from .http_exception_system import HTTPExceptionFactory, OnyxHTTPException
    from .pydantic_schemas import UserCreateInput, UserOutput, UserListOutput
    from .pydantic_schemas import BlogPostCreateInput, BlogPostOutput
from fastapi import FastAPI
from .schema_integration import setup_schema_integration
from typing import Any, List, Dict, Optional
import logging
"""
Schema Integration - FastAPI and Onyx Integration
Integration system for Pydantic schemas with FastAPI, Onyx, and comprehensive API patterns.
"""



    BaseInputModel, BaseOutputModel, PaginatedOutputModel, 
    schema_registry, SchemaFactory, validate_input, validate_output
)

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)
InputT = TypeVar('InputT', bound=BaseInputModel)
OutputT = TypeVar('OutputT', bound=BaseOutputModel)

class SchemaIntegrationManager:
    """
    Manager for integrating Pydantic schemas with FastAPI and Onyx.
    """
    
    def __init__(self) -> Any:
        self._routers: Dict[str, APIRouter] = {}
        self._middleware: List[Callable] = []
        self._dependencies: Dict[str, Callable] = {}
        self._response_models: Dict[str, Type[BaseModel]] = {}
    
    def register_router(self, name: str, router: APIRouter) -> None:
        """Register a FastAPI router."""
        self._routers[name] = router
        logger.info(f"Registered router: {name}")
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware function."""
        self._middleware.append(middleware)
        logger.info(f"Added middleware: {middleware.__name__}")
    
    def register_dependency(self, name: str, dependency: Callable) -> None:
        """Register a dependency injection function."""
        self._dependencies[name] = dependency
        logger.info(f"Registered dependency: {name}")
    
    def register_response_model(self, name: str, model: Type[BaseModel]) -> None:
        """Register a response model."""
        self._response_models[name] = model
        schema_registry.register_schema(name, model)
        logger.info(f"Registered response model: {name}")

# Global integration manager
integration_manager = SchemaIntegrationManager()

class FastAPISchemaDecorator:
    """
    Decorator for FastAPI endpoints with schema validation.
    """
    
    def __init__(self, 
                 input_model: Optional[Type[BaseInputModel]] = None,
                 output_model: Optional[Type[BaseOutputModel]] = None,
                 validate_input: bool = True,
                 validate_output: bool = True,
                 enable_caching: bool = False,
                 cache_ttl: int = 300):
        
    """__init__ function."""
self.input_model = input_model
        self.output_model = output_model
        self.validate_input = validate_input
        self.validate_output = validate_output
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            
            try:
                # Input validation
                if self.validate_input and self.input_model:
                    validated_input = await self._validate_input(kwargs)
                    kwargs.update(validated_input)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Output validation
                if self.validate_output and self.output_model:
                    result = await self._validate_output(result)
                
                # Log performance
                duration = time.perf_counter() - start_time
                logger.info(
                    "Endpoint executed successfully",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    input_model=self.input_model.__name__ if self.input_model else None,
                    output_model=self.output_model.__name__ if self.output_model else None
                )
                
                return result
                
            except ValidationError as e:
                duration = time.perf_counter() - start_time
                logger.warning(
                    "Validation error in endpoint",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    errors=[str(error) for error in e.errors()]
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Validation error: {str(e)}"
                )
            
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    "Error in endpoint",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    error=str(e)
                )
                raise
        
        return wrapper
    
    async def _validate_input(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters."""
        # Extract input data
        input_data = {}
        for key, value in kwargs.items():
            if not key.startswith('_') and not callable(value):
                input_data[key] = value
        
        # Create input model instance
        input_instance = self.input_model(**input_data)
        
        # Validate business rules
        business_errors = input_instance.validate_business_rules()
        if business_errors:
            raise ValidationError(f"Business validation failed: {', '.join(business_errors)}")
        
        return input_instance.model_dump()
    
    async def _validate_output(self, result: Any) -> bool:
        """Validate output data."""
        if isinstance(result, dict):
            # Convert dict to output model
            output_instance = self.output_model(**result)
            return output_instance.model_dump()
        elif isinstance(result, BaseModel):
            # Validate existing model
            if not isinstance(result, self.output_model):
                # Convert to output model
                output_instance = self.output_model(**result.model_dump())
                return output_instance.model_dump()
            return result.model_dump()
        
        return result

def schema_endpoint(
    input_model: Optional[Type[BaseInputModel]] = None,
    output_model: Optional[Type[BaseOutputModel]] = None,
    validate_input: bool = True,
    validate_output: bool = True,
    enable_caching: bool = False,
    cache_ttl: int = 300
):
    """Decorator for FastAPI endpoints with schema validation."""
    decorator = FastAPISchemaDecorator(
        input_model=input_model,
        output_model=output_model,
        validate_input=validate_input,
        validate_output=validate_output,
        enable_caching=enable_caching,
        cache_ttl=cache_ttl
    )
    return decorator

class OnyxSchemaIntegration:
    """
    Integration between Pydantic schemas and Onyx models.
    """
    
    @staticmethod
    def onyx_to_pydantic(onyx_model: Any, pydantic_class: Type[BaseModel]) -> BaseModel:
        """Convert Onyx model to Pydantic model."""
        try:
            # Extract data from Onyx model
            if hasattr(onyx_model, 'to_dict'):
                data = onyx_model.to_dict()
            elif hasattr(onyx_model, 'model_dump'):
                data = onyx_model.model_dump()
            else:
                data = onyx_model.__dict__
            
            # Create Pydantic instance
            return pydantic_class(**data)
            
        except Exception as e:
            logger.error(
                "Error converting Onyx model to Pydantic",
                onyx_model_type=type(onyx_model).__name__,
                pydantic_class=pydantic_class.__name__,
                error=str(e)
            )
            raise
    
    @staticmethod
    def pydantic_to_onyx(pydantic_model: BaseModel, onyx_class: Type) -> Any:
        """Convert Pydantic model to Onyx model."""
        try:
            # Extract data from Pydantic model
            data = pydantic_model.model_dump()
            
            # Create Onyx instance
            return onyx_class(**data)
            
        except Exception as e:
            logger.error(
                "Error converting Pydantic model to Onyx",
                pydantic_model_type=type(pydantic_model).__name__,
                onyx_class=onyx_class.__name__,
                error=str(e)
            )
            raise
    
    @staticmethod
    def validate_onyx_model(onyx_model: Any, schema_name: str) -> ValidationResult:
        """Validate Onyx model using registered schema."""
        try:
            # Convert to dict
            if hasattr(onyx_model, 'to_dict'):
                data = onyx_model.to_dict()
            else:
                data = onyx_model.__dict__
            
            # Validate using schema registry
            validated_data = schema_registry.validate_data(schema_name, data)
            
            return ValidationResult(True)
            
        except Exception as e:
            return ValidationResult(False, [str(e)])

class SchemaMiddleware:
    """
    Middleware for schema validation and monitoring.
    """
    
    def __init__(self, app) -> Any:
        self.app = app
    
    async def __call__(self, scope, receive, send) -> Any:
        if scope["type"] == "http":
            # Add schema validation to request
            request = Request(scope, receive)
            
            # Log request with schema info
            logger.info(
                "HTTP request received",
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers)
            )
            
            # Process request
            await self.app(scope, receive, send)
            
            # Log response
            logger.info(
                "HTTP response sent",
                method=request.method,
                url=str(request.url)
            )
        else:
            await self.app(scope, receive, send)

class SchemaResponseHandler:
    """
    Handler for standardized API responses.
    """
    
    @staticmethod
    def success_response(
        data: Any,
        message: str = "Success",
        status_code: int = 200,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SuccessResponse:
        """Create a success response."""
        return SuccessResponse(
            success=True,
            data=data,
            message=message,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    @staticmethod
    def error_response(
        error: str,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ) -> ErrorResponse:
        """Create an error response."""
        return ErrorResponse(
            success=False,
            error=error,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            details=details or {}
        )
    
    @staticmethod
    def list_response(
        items: List[Any],
        total_count: int,
        page: int = 1,
        page_size: int = 10,
        message: str = "Items retrieved successfully"
    ) -> ListResponse:
        """Create a list response."""
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
        return ListResponse(
            success=True,
            data=items,
            pagination=pagination,
            total_count=total_count,
            message=message,
            timestamp=datetime.utcnow()
        )

class SchemaDependencyInjection:
    """
    Dependency injection for schema validation.
    """
    
    @staticmethod
    def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
        """Get current user from request."""
        # Mock implementation - replace with actual auth logic
        return {
            "id": "user-123",
            "username": "testuser",
            "email": "test@example.com"
        }
    
    @staticmethod
    def validate_user_permission(user: Dict[str, Any], permission: str) -> bool:
        """Validate user permission."""
        # Mock implementation - replace with actual permission logic
        return True
    
    @staticmethod
    def get_schema_validator(schema_name: str):
        """Get schema validator dependency."""
        def validator(data: Dict[str, Any]) -> Dict[str, Any]:
            return schema_registry.validate_data(schema_name, data)
        return validator

# Example FastAPI router with schema integration
def create_user_router() -> APIRouter:
    """Create a user router with schema integration."""
    
    
    router = APIRouter(prefix="/users", tags=["users"])
    
    @router.post("/", response_model=SuccessResponse[UserOutput])
    @schema_endpoint(
        input_model=UserCreateInput,
        output_model=UserOutput,
        validate_input=True,
        validate_output=True
    )
    async def create_user(user_data: UserCreateInput):
        """Create a new user."""
        try:
            # Mock user creation
            user = UserOutput(
                id=str(uuid.uuid4()),
                username=user_data.username,
                email=user_data.email,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                phone=user_data.phone,
                website=user_data.website,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            return SchemaResponseHandler.success_response(
                data=user.model_dump(),
                message="User created successfully",
                status_code=201
            )
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    @router.get("/", response_model=ListResponse[UserOutput])
    async def list_users(
        page: int = 1,
        page_size: int = 10,
        search: Optional[str] = None
    ):
        """List users with pagination."""
        try:
            # Mock user list
            users = [
                UserOutput(
                    id=f"user-{i}",
                    username=f"user{i}",
                    email=f"user{i}@example.com",
                    first_name=f"User{i}",
                    last_name="Example",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                for i in range(1, 21)
            ]
            
            # Apply search filter
            if search:
                users = [u for u in users if search.lower() in u.username.lower()]
            
            # Apply pagination
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paginated_users = users[start_idx:end_idx]
            
            return SchemaResponseHandler.list_response(
                items=[user.model_dump() for user in paginated_users],
                total_count=len(users),
                page=page,
                page_size=page_size
            )
            
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list users"
            )
    
    @router.get("/{user_id}", response_model=SuccessResponse[UserOutput])
    async def get_user(user_id: str):
        """Get user by ID."""
        try:
            # Mock user retrieval
            user = UserOutput(
                id=user_id,
                username="testuser",
                email="test@example.com",
                first_name="Test",
                last_name="User",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            return SchemaResponseHandler.success_response(
                data=user.model_dump(),
                message="User retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    
    return router

# Example blog post router
def create_blog_router() -> APIRouter:
    """Create a blog post router with schema integration."""
    
    
    router = APIRouter(prefix="/blog", tags=["blog"])
    
    @router.post("/posts", response_model=SuccessResponse[BlogPostOutput])
    @schema_endpoint(
        input_model=BlogPostCreateInput,
        output_model=BlogPostOutput,
        validate_input=True,
        validate_output=True
    )
    async def create_post(post_data: BlogPostCreateInput):
        """Create a new blog post."""
        try:
            # Mock post creation
            post = BlogPostOutput(
                id=str(uuid.uuid4()),
                title=post_data.title,
                content=post_data.content,
                excerpt=post_data.excerpt,
                tags=post_data.tags,
                category=post_data.category,
                is_published=post_data.is_published,
                featured_image=post_data.featured_image,
                author_id="user-123",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                published_at=datetime.utcnow() if post_data.is_published else None
            )
            
            return SchemaResponseHandler.success_response(
                data=post.model_dump(),
                message="Blog post created successfully",
                status_code=201
            )
            
        except Exception as e:
            logger.error(f"Error creating blog post: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create blog post"
            )
    
    return router

# Integration setup function
def setup_schema_integration(app) -> Any:
    """Setup schema integration with FastAPI app."""
    
    # Register routers
    user_router = create_user_router()
    blog_router = create_blog_router()
    
    integration_manager.register_router("users", user_router)
    integration_manager.register_router("blog", blog_router)
    
    # Add routers to app
    app.include_router(user_router)
    app.include_router(blog_router)
    
    # Add middleware
    app.add_middleware(SchemaMiddleware)
    
    # Register dependencies
    integration_manager.register_dependency(
        "current_user", 
        SchemaDependencyInjection.get_current_user
    )
    
    logger.info("Schema integration setup completed")

# Example usage in FastAPI app
"""

app = FastAPI(title="Schema Integration Example")

# Setup schema integration
setup_schema_integration(app)

@app.get("/health")
async def health_check():
    
    """health_check function."""
return {"status": "healthy", "timestamp": datetime.utcnow()}
""" 