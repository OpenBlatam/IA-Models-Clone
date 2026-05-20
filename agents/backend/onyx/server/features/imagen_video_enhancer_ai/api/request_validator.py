"""
Request Validator
=================

Request validation utilities for API endpoints.
"""

from typing import Any, Dict, List, Optional, Callable
from fastapi import HTTPException, Request
from pydantic import BaseModel, ValidationError


class RequestValidator:
    """Request validator for API endpoints."""
    
    @staticmethod
    def validate_model(model_class: type, data: Dict[str, Any]) -> BaseModel:
        """
        Validate data against Pydantic model.
        
        Args:
            model_class: Pydantic model class
            data: Data to validate
            
        Returns:
            Validated model instance
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            errors = [f"{err['loc'][-1]}: {err['msg']}" for err in e.errors()]
            raise HTTPException(
                status_code=422,
                detail={"message": "Validation error", "errors": errors}
            )
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required: List[str]) -> None:
        """
        Validate required fields are present.
        
        Args:
            data: Data dictionary
            required: List of required field names
            
        Raises:
            HTTPException: If required fields are missing
        """
        missing = [field for field in required if field not in data or data[field] is None]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing)}"
            )
    
    @staticmethod
    async def validate_file_upload(request: Request, field_name: str = "file") -> Any:
        """
        Validate file upload.
        
        Args:
            request: FastAPI request
            field_name: Form field name
            
        Returns:
            Uploaded file
            
        Raises:
            HTTPException: If file is missing or invalid
        """
        form = await request.form()
        if field_name not in form:
            raise HTTPException(
                status_code=400,
                detail=f"Missing file field: {field_name}"
            )
        
        # Additional file validation would go here
        return form[field_name]
    
    @staticmethod
    def validate_pagination(page: Optional[int] = None, page_size: Optional[int] = None) -> tuple[int, int]:
        """
        Validate and normalize pagination parameters.
        
        Args:
            page: Page number
            page_size: Items per page
            
        Returns:
            Tuple of (page, page_size)
            
        Raises:
            HTTPException: If pagination parameters are invalid
        """
        page = page or 1
        page_size = page_size or 10
        
        if page < 1:
            raise HTTPException(status_code=400, detail="Page must be >= 1")
        
        if page_size < 1 or page_size > 100:
            raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")
        
        return page, page_size
    
    @staticmethod
    def validate_id(id_value: str, id_name: str = "id") -> str:
        """
        Validate ID parameter.
        
        Args:
            id_value: ID value
            id_name: ID parameter name
            
        Returns:
            Validated ID
            
        Raises:
            HTTPException: If ID is invalid
        """
        if not id_value or not id_value.strip():
            raise HTTPException(status_code=400, detail=f"Invalid {id_name}")
        
        return id_value.strip()

