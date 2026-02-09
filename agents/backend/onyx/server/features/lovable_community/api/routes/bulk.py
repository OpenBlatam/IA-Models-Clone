"""
Rutas para operaciones en lote

Incluye: bulk_operation
"""

from fastapi import APIRouter, Depends

from ...dependencies import get_chat_service, get_user_id
from ...schemas import BulkOperationRequest, BulkOperationResponse
from ...services import ChatService
from ...validators import validate_chat_ids, validate_operation, validate_user_id
from ..cache import clear_response_cache
from ..decorators import handle_errors

router = APIRouter()


@router.post(
    "/bulk",
    response_model=BulkOperationResponse,
    summary="Operaciones en lote",
    description="Realiza una operación en lote sobre múltiples chats (máximo 100)"
)
@handle_errors
async def bulk_operation(
    request: BulkOperationRequest,
    user_id: str = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service)
) -> BulkOperationResponse:
    """
    Perform a bulk operation on multiple chats.
    
    Supported operations:
    - delete: Delete chats (requires user_id)
    - feature: Mark chats as featured
    - unfeature: Remove featured status
    - make_public: Make chats public
    - make_private: Make chats private
    
    Args:
        request: BulkOperationRequest with chat_ids and operation
        user_id: Current user ID (from dependency)
        service: ChatService instance
        
    Returns:
        BulkOperationResponse with operation results
        
    Raises:
        ValueError: If operation is invalid or chat_ids exceed limit
    """
    # Validate operation
    operation = validate_operation(request.operation)
    
    # Validate chat IDs (max 100)
    chat_ids = validate_chat_ids(request.chat_ids, max_count=100)
    
    # Validate user_id for delete operation
    validated_user_id = validate_user_id(user_id) if operation == "delete" else None
    
    # Clear cache to reflect changes
    clear_response_cache()
    
    # Perform bulk operation
    result = service.bulk_operation(
        chat_ids=chat_ids,
        operation=operation,
        user_id=validated_user_id
    )
    
    return BulkOperationResponse(**result)

