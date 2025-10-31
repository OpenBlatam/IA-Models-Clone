from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field, validator
from .http_exception_system import (
from .http_exception_handlers import (
from .http_response_models import (
from .error_system import (
from typing import Any, List, Dict, Optional
"""
📚 HTTPException Usage Examples
==============================

Comprehensive examples showing how to use the HTTPException system
in FastAPI applications with proper error handling and response modeling.
"""



    OnyxHTTPException, HTTPExceptionFactory, HTTPExceptionMapper,
    HTTPExceptionHandler, http_exception_handler,
    raise_bad_request, raise_unauthorized, raise_forbidden,
    raise_not_found, raise_conflict, raise_unprocessable_entity,
    raise_too_many_requests, raise_internal_server_error,
    raise_service_unavailable
)
    setup_exception_handlers, handle_http_exceptions
)
    SuccessResponse, ErrorResponse, ListResponse, BatchResponse,
    ResponseFactory, create_success_response, create_error_response,
    create_list_response, create_batch_response
)
    ValidationError, AuthenticationError, AuthorizationError,
    ResourceNotFoundError, BusinessLogicError, DatabaseError,
    ErrorContext, ErrorFactory
)

logger = logging.getLogger(__name__)


# Example Pydantic models
class UserCreate(BaseModel):
    """User creation model"""
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    age: int = Field(..., ge=0, le=150, description="User age")
    
    @validator('email')
    def validate_email(cls, v) -> bool:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v


class UserUpdate(BaseModel):
    """User update model"""
    name: Optional[str] = Field(None, description="User name")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")


class User(BaseModel):
    """User model"""
    id: str
    email: str
    name: str
    age: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BlogPostCreate(BaseModel):
    """Blog post creation model"""
    title: str = Field(..., min_length=1, max_length=200, description="Post title")
    content: str = Field(..., min_length=10, description="Post content")
    author_id: str = Field(..., description="Author ID")


class BlogPost(BaseModel):
    """Blog post model"""
    id: str
    title: str
    content: str
    author_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Example service layer with error handling
class UserService:
    """Example user service with proper error handling"""
    
    def __init__(self) -> Any:
        # Simulate database
        self.users: Dict[str, User] = {}
        self.request_count = 0
    
    def _generate_id(self) -> str:
        """Generate unique user ID"""
        return str(uuid4())
    
    def _validate_user_exists(self, user_id: str) -> User:
        """Validate user exists and return user"""
        if user_id not in self.users:
            raise ResourceNotFoundError(
                message=f"User with ID {user_id} not found",
                resource_type="user",
                resource_id=user_id
            )
        return self.users[user_id]
    
    def _validate_email_unique(self, email: str, exclude_id: Optional[str] = None):
        """Validate email is unique"""
        for user_id, user in self.users.items():
            if user.email == email and user_id != exclude_id:
                raise BusinessLogicError(
                    message=f"User with email {email} already exists",
                    business_rule="unique_email"
                )
    
    def create_user(self, user_data: UserCreate, request_id: Optional[str] = None) -> User:
        """Create a new user"""
        # Validate email uniqueness
        self._validate_email_unique(user_data.email)
        
        # Create user
        user_id = self._generate_id()
        now = datetime.utcnow()
        
        user = User(
            id=user_id,
            email=user_data.email,
            name=user_data.name,
            age=user_data.age,
            created_at=now,
            updated_at=now
        )
        
        self.users[user_id] = user
        
        logger.info(f"User created: {user_id} (Request: {request_id})")
        return user
    
    def get_user(self, user_id: str, request_id: Optional[str] = None) -> User:
        """Get user by ID"""
        user = self._validate_user_exists(user_id)
        
        logger.info(f"User retrieved: {user_id} (Request: {request_id})")
        return user
    
    def get_users(self, skip: int = 0, limit: int = 10, request_id: Optional[str] = None) -> List[User]:
        """Get list of users with pagination"""
        users_list = list(self.users.values())
        total_count = len(users_list)
        
        # Apply pagination
        paginated_users = users_list[skip:skip + limit]
        
        logger.info(f"Users retrieved: {len(paginated_users)}/{total_count} (Request: {request_id})")
        return paginated_users, total_count
    
    def update_user(self, user_id: str, user_data: UserUpdate, request_id: Optional[str] = None) -> User:
        """Update user"""
        user = self._validate_user_exists(user_id)
        
        # Update fields
        if user_data.name is not None:
            user.name = user_data.name
        if user_data.age is not None:
            user.age = user_data.age
        
        user.updated_at = datetime.utcnow()
        
        logger.info(f"User updated: {user_id} (Request: {request_id})")
        return user
    
    def delete_user(self, user_id: str, request_id: Optional[str] = None) -> bool:
        """Delete user"""
        user = self._validate_user_exists(user_id)
        
        del self.users[user_id]
        
        logger.info(f"User deleted: {user_id} (Request: {request_id})")
        return True
    
    def check_rate_limit(self, request_id: Optional[str] = None):
        """Check rate limit (example)"""
        self.request_count += 1
        
        if self.request_count > 100:  # Simulate rate limit
            raise BusinessLogicError(
                message="Rate limit exceeded",
                business_rule="rate_limit"
            )


class BlogPostService:
    """Example blog post service with error handling"""
    
    def __init__(self, user_service: UserService):
        
    """__init__ function."""
self.user_service = user_service
        self.posts: Dict[str, BlogPost] = {}
    
    def _generate_id(self) -> str:
        """Generate unique post ID"""
        return str(uuid4())
    
    def _validate_post_exists(self, post_id: str) -> BlogPost:
        """Validate post exists and return post"""
        if post_id not in self.posts:
            raise ResourceNotFoundError(
                message=f"Blog post with ID {post_id} not found",
                resource_type="blog_post",
                resource_id=post_id
            )
        return self.posts[post_id]
    
    def create_post(self, post_data: BlogPostCreate, request_id: Optional[str] = None) -> BlogPost:
        """Create a new blog post"""
        # Validate author exists
        self.user_service.get_user(post_data.author_id, request_id)
        
        # Create post
        post_id = self._generate_id()
        now = datetime.utcnow()
        
        post = BlogPost(
            id=post_id,
            title=post_data.title,
            content=post_data.content,
            author_id=post_data.author_id,
            created_at=now,
            updated_at=now
        )
        
        self.posts[post_id] = post
        
        logger.info(f"Blog post created: {post_id} (Request: {request_id})")
        return post
    
    def get_post(self, post_id: str, request_id: Optional[str] = None) -> BlogPost:
        """Get blog post by ID"""
        post = self._validate_post_exists(post_id)
        
        logger.info(f"Blog post retrieved: {post_id} (Request: {request_id})")
        return post


# Example FastAPI application with comprehensive error handling
def create_example_app() -> FastAPI:
    """Create example FastAPI application with HTTPException handling"""
    
    app = FastAPI(
        title="HTTPException Example App",
        description="Example application demonstrating HTTPException usage",
        version="1.0.0"
    )
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Create services
    user_service = UserService()
    blog_service = BlogPostService(user_service)
    
    # Create routers
    user_router = APIRouter(prefix="/users", tags=["users"])
    blog_router = APIRouter(prefix="/blog", tags=["blog"])
    admin_router = APIRouter(prefix="/admin", tags=["admin"])
    
    # Dependency to get request ID
    async def get_request_id(request: Request) -> str:
        return request.headers.get("X-Request-ID", str(uuid4()))
    
    # Dependency to check authentication (simplified)
    def check_authentication(request: Request) -> str:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise AuthenticationError(
                message="Invalid or missing authentication token",
                auth_method="bearer"
            )
        
        # In a real app, you would validate the token here
        token = auth_header.split(" ")[1]
        if token == "invalid":
            raise AuthenticationError(
                message="Invalid authentication token",
                auth_method="bearer"
            )
        
        return token
    
    # Dependency to check authorization (simplified)
    def check_admin_permission(token: str = Depends(check_authentication)) -> str:
        if token != "admin_token":
            raise AuthorizationError(
                message="Admin permission required",
                required_permission="admin"
            )
        return token
    
    # User endpoints
    @user_router.post("/", response_model=SuccessResponse[User])
    @handle_http_exceptions
    async def create_user(
        user_data: UserCreate,
        request_id: str = Depends(get_request_id)
    ):
        """Create a new user"""
        try:
            user = user_service.create_user(user_data, request_id)
            return create_success_response(
                data=user,
                message="User created successfully",
                request_id=request_id
            )
        except BusinessLogicError as e:
            # Convert to conflict error
            raise_conflict(
                message=str(e),
                error_code="USER_ALREADY_EXISTS",
                resource_type="user"
            )
    
    @user_router.get("/{user_id}", response_model=SuccessResponse[User])
    @handle_http_exceptions
    async def get_user(
        user_id: str,
        request_id: str = Depends(get_request_id)
    ):
        """Get user by ID"""
        user = user_service.get_user(user_id, request_id)
        return create_success_response(
            data=user,
            message="User retrieved successfully",
            request_id=request_id
        )
    
    @user_router.get("/", response_model=ListResponse[User])
    @handle_http_exceptions
    async def get_users(
        skip: int = 0,
        limit: int = 10,
        request_id: str = Depends(get_request_id)
    ):
        """Get list of users with pagination"""
        # Validate pagination parameters
        if skip < 0:
            raise_bad_request(
                message="Skip parameter must be non-negative",
                error_code="INVALID_PAGINATION",
                field="skip"
            )
        
        if limit <= 0 or limit > 100:
            raise_bad_request(
                message="Limit parameter must be between 1 and 100",
                error_code="INVALID_PAGINATION",
                field="limit"
            )
        
        users, total_count = user_service.get_users(skip, limit, request_id)
        
        return create_list_response(
            data=users,
            total_count=total_count,
            message="Users retrieved successfully",
            request_id=request_id
        )
    
    @user_router.put("/{user_id}", response_model=SuccessResponse[User])
    @handle_http_exceptions
    async def update_user(
        user_id: str,
        user_data: UserUpdate,
        request_id: str = Depends(get_request_id)
    ):
        """Update user"""
        user = user_service.update_user(user_id, user_data, request_id)
        return create_success_response(
            data=user,
            message="User updated successfully",
            request_id=request_id
        )
    
    @user_router.delete("/{user_id}", response_model=SuccessResponse[bool])
    @handle_http_exceptions
    async def delete_user(
        user_id: str,
        request_id: str = Depends(get_request_id)
    ):
        """Delete user"""
        result = user_service.delete_user(user_id, request_id)
        return create_success_response(
            data=result,
            message="User deleted successfully",
            request_id=request_id
        )
    
    # Blog endpoints
    @blog_router.post("/posts", response_model=SuccessResponse[BlogPost])
    @handle_http_exceptions
    async def create_blog_post(
        post_data: BlogPostCreate,
        request_id: str = Depends(get_request_id)
    ):
        """Create a new blog post"""
        post = blog_service.create_post(post_data, request_id)
        return create_success_response(
            data=post,
            message="Blog post created successfully",
            request_id=request_id
        )
    
    @blog_router.get("/posts/{post_id}", response_model=SuccessResponse[BlogPost])
    @handle_http_exceptions
    async def get_blog_post(
        post_id: str,
        request_id: str = Depends(get_request_id)
    ):
        """Get blog post by ID"""
        post = blog_service.get_post(post_id, request_id)
        return create_success_response(
            data=post,
            message="Blog post retrieved successfully",
            request_id=request_id
        )
    
    # Admin endpoints (require authentication and authorization)
    @admin_router.get("/stats", response_model=SuccessResponse[Dict[str, Any]])
    @handle_http_exceptions
    async def get_admin_stats(
        token: str = Depends(check_admin_permission),
        request_id: str = Depends(get_request_id)
    ):
        """Get admin statistics (requires admin permission)"""
        stats = {
            "total_users": len(user_service.users),
            "total_posts": len(blog_service.posts),
            "request_count": user_service.request_count
        }
        
        return create_success_response(
            data=stats,
            message="Admin statistics retrieved successfully",
            request_id=request_id
        )
    
    @admin_router.post("/rate-limit-test")
    @handle_http_exceptions
    async def test_rate_limit(
        request_id: str = Depends(get_request_id)
    ):
        """Test rate limiting"""
        try:
            user_service.check_rate_limit(request_id)
            return create_success_response(
                data={"status": "rate_limit_ok"},
                message="Rate limit check passed",
                request_id=request_id
            )
        except BusinessLogicError as e:
            raise_too_many_requests(
                message=str(e),
                error_code="RATE_LIMIT_EXCEEDED",
                retry_after=60
            )
    
    # Error simulation endpoints
    @user_router.get("/simulate/validation-error")
    @handle_http_exceptions
    async def simulate_validation_error():
        """Simulate validation error"""
        raise ValidationError(
            message="Invalid email format",
            field="email",
            value="invalid-email"
        )
    
    @user_router.get("/simulate/authentication-error")
    @handle_http_exceptions
    async def simulate_authentication_error():
        """Simulate authentication error"""
        raise AuthenticationError(
            message="Invalid authentication token",
            auth_method="bearer"
        )
    
    @user_router.get("/simulate/authorization-error")
    @handle_http_exceptions
    async def simulate_authorization_error():
        """Simulate authorization error"""
        raise AuthorizationError(
            message="Insufficient permissions",
            required_permission="admin"
        )
    
    @user_router.get("/simulate/not-found-error")
    @handle_http_exceptions
    async def simulate_not_found_error():
        """Simulate not found error"""
        raise ResourceNotFoundError(
            message="Resource not found",
            resource_type="user",
            resource_id="non-existent"
        )
    
    @user_router.get("/simulate/business-logic-error")
    @handle_http_exceptions
    async def simulate_business_logic_error():
        """Simulate business logic error"""
        raise BusinessLogicError(
            message="Business rule violation",
            business_rule="unique_constraint"
        )
    
    @user_router.get("/simulate/database-error")
    @handle_http_exceptions
    async def simulate_database_error():
        """Simulate database error"""
        raise DatabaseError(
            message="Database connection failed",
            operation="query",
            table="users"
        )
    
    @user_router.get("/simulate/unexpected-error")
    @handle_http_exceptions
    async def simulate_unexpected_error():
        """Simulate unexpected error"""
        # This will be caught by the general exception handler
        raise ValueError("Unexpected error occurred")
    
    # Include routers
    app.include_router(user_router)
    app.include_router(blog_router)
    app.include_router(admin_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "HTTPException Example App",
            "version": "1.0.0",
            "endpoints": {
                "users": "/users",
                "blog": "/blog",
                "admin": "/admin",
                "error_simulation": "/users/simulate/*"
            }
        }
    
    return app


# Example usage and testing
async def example_usage():
    """Example of how to use the HTTPException system"""
    
    # Create the app
    app = create_example_app()
    
    # Example test cases
    test_cases = [
        {
            "name": "Create User Success",
            "method": "POST",
            "path": "/users/",
            "data": {
                "email": "john@example.com",
                "name": "John Doe",
                "age": 30
            }
        },
        {
            "name": "Create User with Duplicate Email",
            "method": "POST",
            "path": "/users/",
            "data": {
                "email": "john@example.com",
                "name": "John Doe",
                "age": 30
            }
        },
        {
            "name": "Get Non-existent User",
            "method": "GET",
            "path": "/users/non-existent"
        },
        {
            "name": "Simulate Validation Error",
            "method": "GET",
            "path": "/users/simulate/validation-error"
        },
        {
            "name": "Simulate Authentication Error",
            "method": "GET",
            "path": "/users/simulate/authentication-error"
        },
        {
            "name": "Simulate Authorization Error",
            "method": "GET",
            "path": "/users/simulate/authorization-error"
        },
        {
            "name": "Simulate Not Found Error",
            "method": "GET",
            "path": "/users/simulate/not-found-error"
        },
        {
            "name": "Simulate Business Logic Error",
            "method": "GET",
            "path": "/users/simulate/business-logic-error"
        },
        {
            "name": "Simulate Database Error",
            "method": "GET",
            "path": "/users/simulate/database-error"
        },
        {
            "name": "Simulate Unexpected Error",
            "method": "GET",
            "path": "/users/simulate/unexpected-error"
        }
    ]
    
    print("HTTPException Example App")
    print("=" * 50)
    print("Test cases available:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
    print("\nRun the app and test these endpoints to see error handling in action!")


match __name__:
    case "__main__":
    asyncio.run(example_usage()) 