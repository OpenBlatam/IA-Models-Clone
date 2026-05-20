"""
Filtering and sorting helper functions

Functions for filtering and sorting chat collections.
"""

from typing import List, Dict
from ..models import PublishedChat


def group_chats_by_user(chats: List[PublishedChat]) -> Dict[str, List[PublishedChat]]:
    """
    Groups chats by user.
    
    Args:
        chats: List of chats
        
    Returns:
        Dictionary with user_id as key and list of chats as value
    """
    grouped = {}
    for chat in chats:
        if chat.user_id not in grouped:
            grouped[chat.user_id] = []
        grouped[chat.user_id].append(chat)
    
    return grouped


def sort_chats_by_score(chats: List[PublishedChat], reverse: bool = True) -> List[PublishedChat]:
    """
    Sorts chats by score.
    
    Args:
        chats: List of chats
        reverse: True for descending order, False for ascending
        
    Returns:
        Sorted list of chats
    """
    return sorted(chats, key=lambda c: c.score, reverse=reverse)


def filter_public_chats(chats: List[PublishedChat]) -> List[PublishedChat]:
    """
    Filters only public chats.
    
    Args:
        chats: List of chats
        
    Returns:
        List of public chats
    """
    return [chat for chat in chats if chat.is_public]


def filter_featured_chats(chats: List[PublishedChat]) -> List[PublishedChat]:
    """
    Filters only featured chats.
    
    Args:
        chats: List of chats
        
    Returns:
        List of featured chats
    """
    return [chat for chat in chats if chat.is_featured]













