"""
Helpers para aserciones personalizadas de Lovable Community
"""

from typing import Dict, Any, List, Optional


def assert_chat_response_valid(response: Dict[str, Any]) -> None:
    """Verifica que una respuesta de chat sea válida"""
    assert "id" in response, "Missing id in response"
    assert "title" in response, "Missing title in response"
    assert "user_id" in response, "Missing user_id in response"
    
    assert isinstance(response["id"], str), "id must be string"
    assert isinstance(response["title"], str), "title must be string"
    assert isinstance(response["user_id"], str), "user_id must be string"
    
    assert len(response["title"]) > 0, "title cannot be empty"


def assert_chat_list_valid(
    chat_list: List[Dict[str, Any]],
    min_count: int = 0,
    max_count: Optional[int] = None
) -> None:
    """Verifica que una lista de chats sea válida"""
    assert isinstance(chat_list, list), "chat_list must be a list"
    assert len(chat_list) >= min_count, f"Too few chats: {len(chat_list)} < {min_count}"
    
    if max_count is not None:
        assert len(chat_list) <= max_count, f"Too many chats: {len(chat_list)} > {max_count}"
    
    for chat in chat_list:
        assert "id" in chat, "Missing id in chat"
        assert "title" in chat, "Missing title in chat"


def assert_pagination_valid(
    response: Dict[str, Any],
    expected_fields: List[str] = None
) -> None:
    """Verifica que una respuesta con paginación sea válida"""
    if expected_fields is None:
        expected_fields = ["chats", "total", "page", "page_size"]
    
    for field in expected_fields:
        assert field in response, f"Missing {field} in pagination response"
    
    assert isinstance(response["total"], int), "total must be int"
    assert response["total"] >= 0, "total cannot be negative"
    assert isinstance(response.get("page", 1), int), "page must be int"
    assert response.get("page", 1) >= 1, "page must be >= 1"


def assert_vote_response_valid(response: Dict[str, Any]) -> None:
    """Verifica que una respuesta de voto sea válida"""
    assert "id" in response, "Missing id in vote response"
    assert "chat_id" in response, "Missing chat_id in vote response"
    assert "user_id" in response, "Missing user_id in vote response"
    assert "vote_type" in response, "Missing vote_type in vote response"
    
    assert response["vote_type"] in ["upvote", "downvote"], \
        f"Invalid vote_type: {response['vote_type']}"


def assert_remix_response_valid(response: Dict[str, Any]) -> None:
    """Verifica que una respuesta de remix sea válida"""
    assert "id" in response, "Missing id in remix response"
    assert "original_chat_id" in response, "Missing original_chat_id"
    assert "remix_chat_id" in response, "Missing remix_chat_id"
    assert "user_id" in response, "Missing user_id in remix response"


def assert_stats_valid(stats: Dict[str, Any]) -> None:
    """Verifica que estadísticas sean válidas"""
    assert "chat_id" in stats, "Missing chat_id in stats"
    assert "vote_count" in stats, "Missing vote_count in stats"
    assert "remix_count" in stats, "Missing remix_count in stats"
    assert "view_count" in stats, "Missing view_count in stats"
    assert "score" in stats, "Missing score in stats"
    
    assert isinstance(stats["vote_count"], int), "vote_count must be int"
    assert isinstance(stats["remix_count"], int), "remix_count must be int"
    assert isinstance(stats["view_count"], int), "view_count must be int"
    assert isinstance(stats["score"], (int, float)), "score must be number"

