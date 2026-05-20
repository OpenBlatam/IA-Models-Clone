"""
Users and authentication routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.users import (
        CreateUserRequest,
        UserResponse,
        RegisterRequest,
        RegisterResponse,
        LoginRequest,
        LoginResponse
    )
    from schemas.common import ErrorResponse, SuccessResponse
    from dependencies import (
        AuthServiceDep,
    )
    from models.database import DatabaseManager
except ImportError:
    from ...schemas.users import (
        CreateUserRequest,
        UserResponse,
        RegisterRequest,
        RegisterResponse,
        LoginRequest,
        LoginResponse
    )
    from ...schemas.common import ErrorResponse, SuccessResponse
    from ...dependencies import (
        AuthServiceDep,
    )
    from ...models.database import DatabaseManager

router = APIRouter(prefix="/users", tags=["Users & Authentication"])

# Initialize database manager
try:
    db_manager = DatabaseManager()
except:
    db_manager = None


@router.post(
    "/create",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def create_user(
    request: CreateUserRequest
) -> UserResponse:
    """
    Crea un nuevo usuario
    
    - **user_id**: ID único del usuario
    - **email**: Email del usuario (opcional)
    - **name**: Nombre del usuario (opcional)
    """
    # Guard clause: Validate user_id
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager no disponible"
        )
    
    try:
        user = db_manager.create_user(request.user_id, request.email, request.name)
        
        return UserResponse(
            user_id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            updated_at=None
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creando usuario: {str(e)}"
        )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_user(user_id: str) -> UserResponse:
    """
    Obtiene información del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager no disponible"
        )
    
    try:
        user = db_manager.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        return UserResponse(
            user_id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.created_at,
            updated_at=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo usuario: {str(e)}"
        )


@router.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        409: {"model": ErrorResponse, "description": "User already exists"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def register(
    request: RegisterRequest,
    auth: AuthServiceDep
) -> RegisterResponse:
    """
    Registra un nuevo usuario
    
    - **user_id**: ID único del usuario
    - **email**: Email del usuario (opcional)
    - **password**: Contraseña (opcional, mínimo 8 caracteres)
    - **name**: Nombre del usuario (opcional)
    """
    # Guard clause: Validate user_id
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager no disponible"
        )
    
    try:
        # Check if user already exists
        existing_user = db_manager.get_user(request.user_id)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Usuario ya existe"
            )
        
        # Create user
        user = db_manager.create_user(request.user_id, request.email, request.name)
        
        # Hash password if provided
        hashed_password = None
        if request.password:
            hashed_password = auth.hash_password(request.password)
        
        # Create access token
        token_data = {"sub": request.user_id, "email": request.email}
        access_token = auth.create_access_token(data=token_data)
        
        return RegisterResponse(
            user_id=user.id,
            email=user.email,
            access_token=access_token,
            token_type="bearer",
            expires_in=3600
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registrando usuario: {str(e)}"
        )


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def login(
    request: LoginRequest,
    auth: AuthServiceDep
) -> LoginResponse:
    """
    Inicia sesión y obtiene token
    
    - **user_id**: ID del usuario
    - **password**: Contraseña (opcional)
    """
    # Guard clause: Validate user_id
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    if not db_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database manager no disponible"
        )
    
    try:
        user = db_manager.get_user(request.user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Usuario no encontrado"
            )
        
        # In real implementation, verify password here
        # if request.password:
        #     if not auth.verify_password(request.password, user.hashed_password):
        #         raise HTTPException(
        #             status_code=status.HTTP_401_UNAUTHORIZED,
        #             detail="Credenciales inválidas"
        #         )
        
        # Create access token
        token_data = {"sub": request.user_id, "email": user.email}
        access_token = auth.create_access_token(data=token_data)
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=request.user_id,
            expires_in=3600
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en login: {str(e)}"
        )

