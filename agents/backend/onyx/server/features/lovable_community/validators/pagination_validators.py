"""
Pagination validation functions

Functions for validating pagination parameters.
"""


def validate_page(page: int, max_page: int = 1000) -> int:
    """
    Validates a page number.
    
    Args:
        page: Page number to validate
        max_page: Maximum allowed page
        
    Returns:
        Validated page number
        
    Raises:
        ValueError: If the page is invalid
    """
    if not isinstance(page, int):
        raise ValueError("Page must be an integer")
    
    if page < 1:
        raise ValueError("Page must be greater than 0")
    
    if page > max_page:
        raise ValueError(f"Page cannot exceed {max_page}")
    
    return page


def validate_page_size(page_size: int, max_page_size: int = 100) -> int:
    """
    Validates a page size.
    
    Args:
        page_size: Page size to validate
        max_page_size: Maximum allowed page size
        
    Returns:
        Validated page size
        
    Raises:
        ValueError: If the page size is invalid
    """
    if not isinstance(page_size, int):
        raise ValueError("Page size must be an integer")
    
    if page_size < 1:
        raise ValueError("Page size must be greater than 0")
    
    if page_size > max_page_size:
        raise ValueError(f"Page size cannot exceed {max_page_size}")
    
    return page_size


def validate_limit(limit: int, max_limit: int = 100) -> int:
    """
    Validates a result limit.
    
    Args:
        limit: Limit to validate
        max_limit: Maximum allowed limit
        
    Returns:
        Validated limit
        
    Raises:
        ValueError: If the limit is invalid
    """
    if not isinstance(limit, int):
        raise ValueError("Limit must be an integer")
    
    if limit < 1:
        raise ValueError("Limit must be greater than 0")
    
    if limit > max_limit:
        raise ValueError(f"Limit cannot exceed {max_limit}")
    
    return limit













