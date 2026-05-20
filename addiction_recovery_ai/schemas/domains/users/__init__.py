"""
Users domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.users import (
        CreateUserRequest,
        UserResponse,
        RegisterRequest,
        RegisterResponse,
        LoginRequest,
        LoginResponse
    )
    
    def register_schemas():
        register_schema("users", "CreateUserRequest", CreateUserRequest)
        register_schema("users", "UserResponse", UserResponse)
        register_schema("users", "RegisterRequest", RegisterRequest)
        register_schema("users", "RegisterResponse", RegisterResponse)
        register_schema("users", "LoginRequest", LoginRequest)
        register_schema("users", "LoginResponse", LoginResponse)
except ImportError:
    pass



