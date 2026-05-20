"""
Rutas para operaciones de votos

Incluye: vote_chat
"""

from fastapi import APIRouter, Depends, status

from ...dependencies import get_chat_service, get_user_id
from ...exceptions import InvalidChatError
from ...helpers import vote_to_response
from ...schemas import VoteRequest, VoteResponse
from ...services import ChatService
from ...validators import validate_chat_id, validate_user_id, validate_vote_type
from ..cache import clear_response_cache
from ..decorators import handle_errors

router = APIRouter()


@router.post(
    "/chats/{chat_id}/vote",
    response_model=VoteResponse,
    summary="Votar un chat",
    description="Vota (upvote o downvote) un chat"
)
@handle_errors
async def vote_chat(
    chat_id: str,
    request: VoteRequest,
    user_id: str = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service)
) -> VoteResponse:
    """
    Vote on a chat (upvote or downvote).
    
    Args:
        chat_id: Chat ID from URL path
        request: Vote request with chat_id and vote_type
        user_id: Current user ID (from dependency)
        service: ChatService instance
        
    Returns:
        VoteResponse with vote information
        
    Raises:
        InvalidChatError: If chat_id mismatch or validation fails
    """
    # Validate and normalize IDs
    chat_id = validate_chat_id(chat_id)
    user_id = validate_user_id(user_id)
    vote_type = validate_vote_type(request.vote_type)
    
    # Verify chat_id consistency
    if request.chat_id and request.chat_id != chat_id:
        raise InvalidChatError(
            f"Chat ID mismatch: path parameter '{chat_id}' != request body '{request.chat_id}'"
        )
    
    # Clear cache to reflect vote changes
    clear_response_cache()
    
    # Perform vote operation
    vote = service.vote_chat(chat_id, user_id, vote_type)
    return vote_to_response(vote)

