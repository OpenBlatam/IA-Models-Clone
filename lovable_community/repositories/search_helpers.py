"""
Search Query Helpers

Helper functions for building search queries with multiple field matching.
"""

from typing import List
from sqlalchemy.orm import Query
from sqlalchemy import or_


def build_multi_field_search_filter(
    query: Query,
    search_term: str,
    fields: List,
    use_ilike: bool = True
) -> Query:
    """
    Build a search filter that matches search_term across multiple fields.
    
    This helper encapsulates the common pattern of searching across multiple
    fields (title, description, tags) that appears repeatedly in repositories.
    
    Args:
        query: SQLAlchemy query
        search_term: Search term to match (should already be formatted with wildcards)
        fields: List of model fields to search in
        use_ilike: Whether to use case-insensitive ILIKE (default: True)
        
    Returns:
        Query with search filters applied
        
    Example:
        >>> query = build_multi_field_search_filter(
        >>>     query,
        >>>     "%python%",
        >>>     [PublishedChat.title, PublishedChat.description, PublishedChat.tags]
        >>> )
    """
    if not search_term or not fields:
        return query
    
    filters = []
    for field in fields:
        if use_ilike:
            filters.append(field.ilike(search_term))
        else:
            filters.append(field.like(search_term))
    
    if filters:
        return query.filter(or_(*filters))
    
    return query


def build_tag_filters(
    query: Query,
    tags: List[str],
    tag_field,
    use_ilike: bool = True
) -> Query:
    """
    Build filters for tag-based search.
    
    This helper encapsulates the common pattern of filtering by tags
    that appears repeatedly in repository methods.
    
    Args:
        query: SQLAlchemy query
        tags: List of tags to filter by
        tag_field: Model field containing tags
        use_ilike: Whether to use case-insensitive ILIKE (default: True)
        
    Returns:
        Query with tag filters applied
        
    Example:
        >>> query = build_tag_filters(
        >>>     query,
        >>>     ["python", "ai"],
        >>>     PublishedChat.tags
        >>> )
    """
    if not tags:
        return query
    
    from ...helpers.string_normalization import build_search_term
    
    tag_filters = []
    for tag in tags:
        if tag:
            search_term = build_search_term(tag)
            if use_ilike:
                tag_filters.append(tag_field.ilike(search_term))
            else:
                tag_filters.append(tag_field.like(search_term))
    
    if tag_filters:
        return query.filter(or_(*tag_filters))
    
    return query

