"""
Authentication Routes
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging

from ..services.auth.jwt_handler import get_jwt_handler
from ..services.auth.user_service import get_user_service
from ..services.auth.permissions import Permission, check_permission

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str
    roles: Optional[list] = None


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    token_type: str = "bearer"
    user: dict


def get_current_user(token: str = Header(..., alias="Authorization")):
    """Dependency to get current user from JWT token"""
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    token = token.replace("Bearer ", "")
    jwt_handler = get_jwt_handler()
    payload = jwt_handler.validate_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_service = get_user_service()
    user = user_service.get_user(payload.get("user_id"))
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    """Register new user"""
    try:
        user_service = get_user_service()
        user = user_service.create_user(
            email=request.email,
            password=request.password,
            roles=request.roles
        )
        
        jwt_handler = get_jwt_handler()
        token = jwt_handler.generate_token(
            user_id=user.user_id,
            email=user.email,
            roles=user.roles
        )
        
        return TokenResponse(
            access_token=token,
            user=user.to_dict()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Login user"""
    user_service = get_user_service()
    user = user_service.authenticate(request.email, request.password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    jwt_handler = get_jwt_handler()
    token = jwt_handler.generate_token(
        user_id=user.user_id,
        email=user.email,
        roles=user.roles
    )
    
    return TokenResponse(
        access_token=token,
        user=user.to_dict()
    )


@router.post("/refresh")
async def refresh_token(token: str = Header(..., alias="Authorization")):
    """Refresh JWT token"""
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    token = token.replace("Bearer ", "")
    jwt_handler = get_jwt_handler()
    new_token = jwt_handler.refresh_token(token)
    
    if not new_token:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {"access_token": new_token, "token_type": "bearer"}


@router.get("/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information"""
    return current_user.to_dict()


@router.get("/api-key")
async def get_api_key(current_user = Depends(get_current_user)):
    """Get user API key"""
    return {
        "api_key": current_user.api_key,
        "user_id": current_user.user_id,
    }


@router.post("/api-key/regenerate")
async def regenerate_api_key(current_user = Depends(get_current_user)):
    """Regenerate API key"""
    user_service = get_user_service()
    new_api_key = user_service.regenerate_api_key(current_user.user_id)
    return {
        "api_key": new_api_key,
        "message": "API key regenerated",
    }

