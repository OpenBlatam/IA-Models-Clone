from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from functools import wraps
import httpx
import aiofiles
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, field_validator
import structlog
from typing import Any, List, Dict, Optional
import logging
"""
Synchronous vs Asynchronous Function Examples
============================================

This module demonstrates proper use of:
- `def` for synchronous operations (CPU-bound, simple transformations)
- `async def` for asynchronous operations (I/O-bound, external calls)

Key principles:
- Use `def` for CPU-bound computations and simple data processing
- Use `async def` for I/O operations (database, HTTP, file operations)
- Mix both appropriately for optimal performance
"""



logger = structlog.get_logger(__name__)

# ============================================================================
# Synchronous Functions (def) - CPU-bound and simple operations
# ============================================================================

def calculate_user_score(user_data: Dict[str, Any]) -> float:
    """
    Calculate user score based on various metrics (synchronous).
    
    Args:
        user_data: User data dictionary
        
    Returns:
        float: Calculated score (0-100)
    """
    # CPU-bound computation
    engagement_score = user_data.get('engagement', 0) * 0.3
    activity_score = user_data.get('activity', 0) * 0.4
    quality_score = user_data.get('quality', 0) * 0.3
    
    return min(engagement_score + activity_score + quality_score, 100.0)

def validate_email_format(email: str) -> bool:
    """
    Validate email format using regex (synchronous).
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[^@]+@[^@]+\.[^@]+$'
    return bool(re.match(pattern, email.strip()))

def format_user_display_name(first_name: str, last_name: str) -> str:
    """
    Format user display name (synchronous).
    
    Args:
        first_name: User's first name
        last_name: User's last name
        
    Returns:
        str: Formatted display name
    """
    first = first_name.strip().title() if first_name else ""
    last = last_name.strip().title() if last_name else ""
    
    if first and last:
        return f"{first} {last}"
    elif first:
        return first
    elif last:
        return last
    else:
        return "Unknown User"

def validate_user_input(data: Dict[str, Any]) -> List[str]:
    """
    Validate user input data (synchronous).
    
    Args:
        data: User input data
        
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    # Validate name
    name = data.get('name', '').strip()
    if not name:
        errors.append("Name is required")
    elif len(name) < 2:
        errors.append("Name must be at least 2 characters long")
    elif len(name) > 100:
        errors.append("Name must be less than 100 characters")
    
    # Validate email
    email = data.get('email', '').strip()
    if not email:
        errors.append("Email is required")
    elif not validate_email_format(email):
        errors.append("Invalid email format")
    
    # Validate age
    age = data.get('age')
    if age is not None:
        if not isinstance(age, int):
            errors.append("Age must be an integer")
        elif age < 0 or age > 150:
            errors.append("Age must be between 0 and 150")
    
    return errors

def calculate_password_strength(password: str) -> Dict[str, Any]:
    """
    Calculate password strength (synchronous).
    
    Args:
        password: Password to analyze
        
    Returns:
        Dict[str, Any]: Password strength analysis
    """
    if not password:
        return {
            "score": 0,
            "strength": "very_weak",
            "issues": ["Password is empty"]
        }
    
    score = 0
    issues = []
    
    # Length check
    if len(password) >= 8:
        score += 20
    else:
        issues.append("Password should be at least 8 characters long")
    
    if len(password) >= 12:
        score += 10
    
    # Character variety checks
    if re.search(r'[a-z]', password):
        score += 10
    else:
        issues.append("Password should contain lowercase letters")
    
    if re.search(r'[A-Z]', password):
        score += 10
    else:
        issues.append("Password should contain uppercase letters")
    
    if re.search(r'\d', password):
        score += 10
    else:
        issues.append("Password should contain numbers")
    
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 10
    else:
        issues.append("Password should contain special characters")
    
    # Determine strength level
    if score >= 60:
        strength = "strong"
    elif score >= 40:
        strength = "medium"
    elif score >= 20:
        strength = "weak"
    else:
        strength = "very_weak"
    
    return {
        "score": score,
        "strength": strength,
        "issues": issues
    }

def generate_cache_key(data: Dict[str, Any]) -> str:
    """
    Generate cache key from data (synchronous).
    
    Args:
        data: Data to generate key from
        
    Returns:
        str: Cache key
    """
    # Sort data for consistent keys
    sorted_data = json.dumps(data, sort_keys=True)
    return hashlib.md5(sorted_data.encode()).hexdigest()

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount (synchronous).
    
    Args:
        amount: Amount to format
        currency: Currency code
        
    Returns:
        str: Formatted currency string
    """
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    if currency == "JPY":
        return f"{symbol}{int(amount):,}"
    else:
        return f"{symbol}{amount:,.2f}"

def calculate_date_range(days: int) -> Tuple[datetime, datetime]:
    """
    Calculate date range (synchronous).
    
    Args:
        days: Number of days to look back
        
    Returns:
        Tuple[datetime, datetime]: Start and end dates
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    return start_date, end_date

# ============================================================================
# Asynchronous Functions (async def) - I/O-bound operations
# ============================================================================

async async def fetch_user_from_database(user_id: str, session: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Fetch user data from database (asynchronous).
    
    Args:
        user_id: User ID to fetch
        session: Database session
        
    Returns:
        Optional[Dict[str, Any]]: User data or None if not found
    """
    try:
        # Database operation (I/O-bound)
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user:
            return user.to_dict()
        return None
        
    except Exception as e:
        logger.error(f"Failed to fetch user {user_id}: {str(e)}")
        return None

async def create_user_in_database(user_data: Dict[str, Any], session: AsyncSession) -> Dict[str, Any]:
    """
    Create user in database (asynchronous).
    
    Args:
        user_data: User data to create
        session: Database session
        
    Returns:
        Dict[str, Any]: Created user data
    """
    try:
        user = User(**user_data)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user.to_dict()
        
    except Exception as e:
        await session.rollback()
        logger.error(f"Failed to create user: {str(e)}")
        raise

async def check_user_exists(email: str, session: AsyncSession) -> bool:
    """
    Check if user exists (asynchronous).
    
    Args:
        email: Email to check
        session: Database session
        
    Returns:
        bool: True if user exists, False otherwise
    """
    try:
        result = await session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none() is not None
        
    except Exception as e:
        logger.error(f"Failed to check user existence: {str(e)}")
        return False

async async def call_external_api(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call external API (asynchronous).
    
    Args:
        endpoint: API endpoint URL
        data: Request data
        
    Returns:
        Dict[str, Any]: API response
    """
    try:
        # HTTP request (I/O-bound)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPError as e:
        logger.error(f"API call failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in API call: {str(e)}")
        raise

async def send_email_notification(email: str, subject: str, content: str) -> bool:
    """
    Send email notification (asynchronous).
    
    Args:
        email: Recipient email
        subject: Email subject
        content: Email content
        
    Returns:
        bool: True if sent successfully, False otherwise
    """
    try:
        # Simulate email sending (I/O-bound)
        await asyncio.sleep(0.1)  # Simulate network delay
        
        logger.info(f"Email sent to {email}: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {email}: {str(e)}")
        return False

async async def process_file_upload(file_path: str) -> Dict[str, Any]:
    """
    Process uploaded file (asynchronous).
    
    Args:
        file_path: Path to uploaded file
        
    Returns:
        Dict[str, Any]: Processing results
    """
    try:
        # File I/O operation (I/O-bound)
        async with aiofiles.open(file_path, 'r') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            content = await file.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        # Process content (could be async if needed)
        file_size = len(content)
        line_count = content.count('\n') + 1
        word_count = len(content.split())
        
        return {
            "file_size": file_size,
            "line_count": line_count,
            "word_count": word_count,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}")
        raise

async async def fetch_user_analytics(user_id: str, days: int = 30) -> Dict[str, Any]:
    """
    Fetch user analytics data (asynchronous).
    
    Args:
        user_id: User ID
        days: Number of days to fetch
        
    Returns:
        Dict[str, Any]: Analytics data
    """
    try:
        # Calculate date range (synchronous)
        start_date, end_date = calculate_date_range(days)
        
        # Database query (I/O-bound)
        async with get_database_session() as session:
            result = await session.execute(
                select(UserAnalytics)
                .where(UserAnalytics.user_id == user_id)
                .where(UserAnalytics.date >= start_date)
                .where(UserAnalytics.date <= end_date)
            )
            
            analytics = result.scalars().all()
            
            return {
                "user_id": user_id,
                "period_days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "data": [item.to_dict() for item in analytics]
            }
            
    except Exception as e:
        logger.error(f"Failed to fetch analytics for user {user_id}: {str(e)}")
        return {}

# ============================================================================
# Hybrid Functions - Mixing sync and async operations
# ============================================================================

async def process_user_registration(user_request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user registration with mixed sync/async operations.
    
    Args:
        user_request: User registration data
        
    Returns:
        Dict[str, Any]: Registration results
    """
    # Synchronous validation
    validation_errors = validate_user_input(user_request)
    if validation_errors:
        return {
            "success": False,
            "errors": validation_errors
        }
    
    # Synchronous data preparation
    user_data = {
        "name": format_user_display_name(user_request.get('first_name', ''), user_request.get('last_name', '')),
        "email": user_request['email'].lower().strip(),
        "age": user_request.get('age'),
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    # Asynchronous database operations
    try:
        async with get_database_session() as session:
            # Check if user exists
            user_exists = await check_user_exists(user_data['email'], session)
            if user_exists:
                return {
                    "success": False,
                    "error": "User with this email already exists"
                }
            
            # Create user
            created_user = await create_user_in_database(user_data, session)
            
            # Parallel async operations
            await asyncio.gather(
                send_email_notification(
                    created_user['email'],
                    "Welcome!",
                    f"Welcome {created_user['name']} to our platform!"
                ),
                update_registration_analytics(),
                create_user_profile(created_user['id'])
            )
            
            return {
                "success": True,
                "user": created_user,
                "message": "User registered successfully"
            }
            
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return {
            "success": False,
            "error": "Registration failed"
        }

async def get_user_comprehensive_data(user_id: str) -> Dict[str, Any]:
    """
    Get comprehensive user data with parallel async operations.
    
    Args:
        user_id: User ID
        
    Returns:
        Dict[str, Any]: Comprehensive user data
    """
    try:
        # Parallel async operations
        user_data, analytics_data, preferences_data = await asyncio.gather(
            fetch_user_from_database(user_id, get_database_session()),
            fetch_user_analytics(user_id),
            fetch_user_preferences(user_id)
        )
        
        if not user_data:
            return {
                "success": False,
                "error": "User not found"
            }
        
        # Synchronous data processing
        user_score = calculate_user_score(user_data)
        display_name = format_user_display_name(user_data.get('first_name', ''), user_data.get('last_name', ''))
        
        return {
            "success": True,
            "user": {
                **user_data,
                "display_name": display_name,
                "score": user_score
            },
            "analytics": analytics_data,
            "preferences": preferences_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive data for user {user_id}: {str(e)}")
        return {
            "success": False,
            "error": "Failed to fetch user data"
        }

# ============================================================================
# Performance Monitoring Decorators
# ============================================================================

def measure_sync_performance(func) -> Any:
    """Decorator to measure synchronous function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        logger.info(
            f"Sync function {func.__name__} took {execution_time:.4f} seconds",
            function=func.__name__,
            execution_time=execution_time,
            type="sync"
        )
        return result
    return wrapper

def measure_async_performance(func) -> Any:
    """Decorator to measure asynchronous function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        logger.info(
            f"Async function {func.__name__} took {execution_time:.4f} seconds",
            function=func.__name__,
            execution_time=execution_time,
            type="async"
        )
        return result
    return wrapper

# ============================================================================
# FastAPI Route Examples
# ============================================================================

# Pydantic models
class UserCreateRequest(BaseModel):
    """User creation request model."""
    
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., description="User email")
    age: Optional[int] = Field(None, ge=0, le=150)
    password: str = Field(..., min_length=8)
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        if not validate_email_format(v):
            raise ValueError('Invalid email format')
        return v.lower().strip()

class UserResponse(BaseModel):
    """User response model."""
    
    success: bool
    user_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    score: Optional[float] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None

# FastAPI routes
app = FastAPI(title="Sync/Async Function Examples")

@app.post("/users/register", response_model=UserResponse)
@measure_async_performance
async def register_user(user_request: UserCreateRequest) -> UserResponse:
    """
    Register a new user (asynchronous route).
    
    Args:
        user_request: User registration data
        
    Returns:
        UserResponse: Registration result
    """
    # Convert to dict for processing
    user_data = user_request.model_dump()
    
    # Process registration (mixed sync/async)
    result = await process_user_registration(user_data)
    
    if result["success"]:
        return UserResponse(
            success=True,
            user_id=result["user"]["id"],
            name=result["user"]["name"],
            email=result["user"]["email"],
            message=result["message"]
        )
    else:
        return UserResponse(
            success=False,
            errors=result.get("errors", [result.get("error", "Unknown error")])
        )

@app.get("/users/{user_id}", response_model=UserResponse)
@measure_async_performance
async def get_user(user_id: str) -> UserResponse:
    """
    Get user by ID (asynchronous route).
    
    Args:
        user_id: User ID
        
    Returns:
        UserResponse: User data
    """
    # Get comprehensive user data (mixed sync/async)
    result = await get_user_comprehensive_data(user_id)
    
    if result["success"]:
        user = result["user"]
        return UserResponse(
            success=True,
            user_id=user["id"],
            name=user["display_name"],
            email=user["email"],
            score=user["score"]
        )
    else:
        return UserResponse(
            success=False,
            message=result["error"]
        )

@app.post("/users/validate")
@measure_sync_performance
def validate_user_data(user_request: UserCreateRequest) -> Dict[str, Any]:
    """
    Validate user data (synchronous route).
    
    Args:
        user_request: User data to validate
        
    Returns:
        Dict[str, Any]: Validation results
    """
    # Synchronous validation
    user_data = user_request.model_dump()
    validation_errors = validate_user_input(user_data)
    
    # Synchronous password strength check
    password_analysis = calculate_password_strength(user_data["password"])
    
    return {
        "valid": len(validation_errors) == 0,
        "validation_errors": validation_errors,
        "password_strength": password_analysis,
        "formatted_name": format_user_display_name(user_data["first_name"], user_data["last_name"])
    }

@app.get("/users/{user_id}/analytics")
@measure_async_performance
async def get_user_analytics(user_id: str, days: int = 30) -> Dict[str, Any]:
    """
    Get user analytics (asynchronous route).
    
    Args:
        user_id: User ID
        days: Number of days to fetch
        
    Returns:
        Dict[str, Any]: Analytics data
    """
    # Asynchronous analytics fetch
    analytics = await fetch_user_analytics(user_id, days)
    
    if not analytics:
        raise HTTPException(status_code=404, detail="User not found")
    
    return analytics

# ============================================================================
# Mock Database Models (for demonstration)
# ============================================================================

@dataclass
class User:
    """Mock user model."""
    id: str
    name: str
    email: str
    age: Optional[int]
    created_at: datetime
    status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "created_at": self.created_at.isoformat(),
            "status": self.status
        }

@dataclass
class UserAnalytics:
    """Mock user analytics model."""
    user_id: str
    date: datetime
    activity_score: float
    engagement_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "date": self.date.isoformat(),
            "activity_score": self.activity_score,
            "engagement_score": self.engagement_score
        }

# Mock database session
async def get_database_session():
    """Mock database session."""
    # In real implementation, this would return an actual database session
    return None

# Mock functions for demonstration
async def update_registration_analytics():
    """Mock function to update registration analytics."""
    await asyncio.sleep(0.05)

async def create_user_profile(user_id: str):
    """Mock function to create user profile."""
    await asyncio.sleep(0.1)

async def fetch_user_preferences(user_id: str):
    """Mock function to fetch user preferences."""
    await asyncio.sleep(0.05)
    return {"theme": "dark", "language": "en"}

# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of sync and async functions."""
    
    # Synchronous operations
    print("=== Synchronous Operations ===")
    
    # Calculate user score
    user_data = {"engagement": 80, "activity": 70, "quality": 90}
    score = calculate_user_score(user_data)
    print(f"User score: {score}")
    
    # Validate email
    is_valid = validate_email_format("user@example.com")
    print(f"Email valid: {is_valid}")
    
    # Format name
    display_name = format_user_display_name("john", "doe")
    print(f"Display name: {display_name}")
    
    # Asynchronous operations
    print("\n=== Asynchronous Operations ===")
    
    # Simulate user registration
    user_request = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "password": "SecurePass123!"
    }
    
    result = await process_user_registration(user_request)
    print(f"Registration result: {result}")
    
    # Simulate comprehensive data fetch
    user_data = await get_user_comprehensive_data("user123")
    print(f"User data: {user_data}")

match __name__:
    case "__main__":
    asyncio.run(main()) 