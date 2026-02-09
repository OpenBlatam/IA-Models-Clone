# Error Handling and Validation in FastAPI

## 1. Pydantic Validation

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from fastapi import HTTPException

class PostRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=3000)
    post_type: str = Field(..., regex="^(educational|promotional|personal|industry)$")
    tone: str = Field(default="professional", regex="^(professional|casual|enthusiastic|thoughtful)$")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class PostResponse(BaseModel):
    id: str
    content: str
    status: str
    created_at: str
```

## 2. Custom Exception Classes

```python
class LinkedInPostError(Exception):
    """Base exception for LinkedIn Posts application"""
    pass

class ValidationError(LinkedInPostError):
    """Raised when input validation fails"""
    pass

class ProcessingError(LinkedInPostError):
    """Raised when post processing fails"""
    pass

class RateLimitError(LinkedInPostError):
    """Raised when rate limit is exceeded"""
    pass
```

## 3. Global Exception Handlers

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "message": "Invalid input data"
        }
    )

@app.exception_handler(LinkedInPostError)
async def linkedin_post_exception_handler(request: Request, exc: LinkedInPostError):
    return JSONResponse(
        status_code=400,
        content={
            "error": type(exc).__name__,
            "message": str(exc)
        }
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail
        }
    )
```

## 4. Route-Level Error Handling

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

post_router = APIRouter()

@post_router.post("/posts", response_model=PostResponse)
async def create_post(post: PostRequest):
    try:
        # Validate business logic
        if len(post.content.split()) < 5:
            raise ValidationError("Post must have at least 5 words")
        
        # Process post
        result = await process_post(post)
        return PostResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail="Processing failed")
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 5. Dependency Injection for Validation

```python
from fastapi import Depends, HTTPException, status
from typing import Optional

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user_id
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

async def validate_rate_limit(user_id: str = Depends(get_current_user)):
    if await is_rate_limited(user_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return user_id

@post_router.post("/posts")
async def create_post(
    post: PostRequest,
    user_id: str = Depends(validate_rate_limit)
):
    # Post creation logic
    pass
```

## 6. Input Sanitization

```python
import re
from typing import Optional

def sanitize_text(text: str) -> str:
    """Remove potentially dangerous content"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def validate_content_safety(content: str) -> bool:
    """Check for inappropriate content"""
    inappropriate_words = ['spam', 'inappropriate', 'banned']
    content_lower = content.lower()
    return not any(word in content_lower for word in inappropriate_words)

class SafePostRequest(BaseModel):
    content: str
    
    @validator('content')
    def validate_and_sanitize_content(cls, v):
        # Sanitize input
        sanitized = sanitize_text(v)
        
        # Validate safety
        if not validate_content_safety(sanitized):
            raise ValueError("Content contains inappropriate material")
        
        return sanitized
```

## 7. Response Validation

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class PostResponse(BaseModel):
    id: str
    content: str
    hashtags: List[str]
    engagement_score: float
    
    @validator('engagement_score')
    def validate_score(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("Engagement score must be between 0 and 100")
        return round(v, 2)
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 hashtags allowed")
        return v

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: str
```

## 8. Middleware for Error Logging

```python
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error details
            logger.error(
                f"Error processing {request.method} {request.url}: {str(e)}",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host,
                    "user_agent": request.headers.get("user-agent"),
                    "processing_time": time.time() - start_time
                }
            )
            raise
```

## 9. Custom Validators

```python
from pydantic import validator
import re

class PostValidator:
    @staticmethod
    def validate_hashtags(hashtags: List[str]) -> List[str]:
        """Validate and normalize hashtags"""
        valid_hashtags = []
        for tag in hashtags:
            # Remove # if present and normalize
            clean_tag = tag.lstrip('#').lower()
            if re.match(r'^[a-zA-Z0-9_]+$', clean_tag):
                valid_hashtags.append(f"#{clean_tag}")
        return valid_hashtags
    
    @staticmethod
    def validate_content_length(content: str) -> str:
        """Validate content length and structure"""
        if len(content) < 10:
            raise ValueError("Content too short")
        if len(content) > 3000:
            raise ValueError("Content too long")
        return content

class ValidatedPostRequest(BaseModel):
    content: str
    hashtags: List[str] = []
    
    @validator('content')
    def validate_content(cls, v):
        return PostValidator.validate_content_length(v)
    
    @validator('hashtags')
    def validate_hashtags(cls, v):
        return PostValidator.validate_hashtags(v)
```

## 10. Error Response Examples

```python
# 400 Bad Request
{
    "error": "ValidationError",
    "message": "Invalid input data",
    "details": [
        {
            "loc": ["body", "content"],
            "msg": "ensure this value has at least 10 characters",
            "type": "value_error.any_str.min_length"
        }
    ],
    "timestamp": "2024-01-15T10:30:00Z"
}

# 429 Too Many Requests
{
    "error": "RateLimitError",
    "message": "Rate limit exceeded. Try again in 60 seconds.",
    "retry_after": 60,
    "timestamp": "2024-01-15T10:30:00Z"
}

# 500 Internal Server Error
{
    "error": "InternalServerError",
    "message": "An unexpected error occurred",
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 11. Testing Error Handling

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_validation_error():
    response = client.post("/posts", json={
        "content": "short",  # Too short
        "post_type": "invalid_type"  # Invalid type
    })
    assert response.status_code == 422
    assert "Validation Error" in response.json()["error"]

def test_rate_limit_error():
    # Make multiple requests quickly
    for _ in range(11):
        response = client.post("/posts", json={
            "content": "Valid content with enough words to pass validation",
            "post_type": "educational"
        })
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["message"]

def test_custom_exception():
    response = client.post("/posts", json={
        "content": "spam content",  # Contains banned word
        "post_type": "educational"
    })
    assert response.status_code == 400
    assert "inappropriate material" in response.json()["message"]
```

## 12. Best Practices

1. **Always validate input** using Pydantic models
2. **Use specific exception types** for different error scenarios
3. **Log errors with context** for debugging
4. **Return consistent error responses** across all endpoints
5. **Sanitize user input** to prevent security issues
6. **Use HTTP status codes correctly** (400 for client errors, 500 for server errors)
7. **Implement rate limiting** to prevent abuse
8. **Add request IDs** for tracking errors in logs
9. **Test error scenarios** thoroughly
10. **Document error responses** in API documentation 