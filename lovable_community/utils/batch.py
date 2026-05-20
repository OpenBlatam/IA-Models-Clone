"""
Batch Processing Utilities

Utilities for efficient batch operations.
"""

from typing import List, TypeVar, Callable, Iterator, Any, Optional

T = TypeVar('T')


def batch_process(
    items: List[T],
    batch_size: int = 100,
    processor: Optional[Callable[[List[T]], Any]] = None
) -> Iterator[List[T]]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        processor: Optional function to process each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        if processor:
            processor(batch)
        yield batch


def chunk_list(items: List[T], chunk_size: int = 100) -> Iterator[List[T]]:
    """
    Split list into chunks.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
        
    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def bulk_create(
    db_session,
    model_class,
    items: List[dict],
    batch_size: int = 100
) -> List[Any]:
    """
    Bulk create model instances efficiently.
    
    Args:
        db_session: Database session
        model_class: SQLAlchemy model class
        items: List of dictionaries with model attributes
        batch_size: Size of each batch
        
    Returns:
        List of created instances
    """
    created = []
    
    for batch in chunk_list(items, batch_size):
        instances = [model_class(**item) for item in batch]
        db_session.bulk_save_objects(instances)
        created.extend(instances)
    
    db_session.commit()
    return created


def bulk_update(
    db_session,
    model_class,
    updates: List[dict],
    batch_size: int = 100
) -> int:
    """
    Bulk update model instances efficiently.
    
    Args:
        db_session: Database session
        model_class: SQLAlchemy model class
        updates: List of dictionaries with id and fields to update
        batch_size: Size of each batch
        
    Returns:
        Number of updated rows
    """
    from sqlalchemy import update
    
    total_updated = 0
    
    for batch in chunk_list(updates, batch_size):
        for update_dict in batch:
            chat_id = update_dict.pop('id')
            stmt = (
                update(model_class)
                .where(model_class.id == chat_id)
                .values(**update_dict)
            )
            result = db_session.execute(stmt)
            total_updated += result.rowcount
        
        db_session.commit()
    
    return total_updated

