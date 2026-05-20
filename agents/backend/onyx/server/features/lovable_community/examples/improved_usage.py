"""
Ejemplos de uso mejorado

Demuestra cómo usar las nuevas mejoras: Unit of Work, Cache, Retry, etc.
"""

from datetime import datetime
from sqlalchemy.orm import Session

from ..core import unit_of_work, cached, get_cache
from ..utils import timer, measure_time, retry, validate_length
from ..repositories import ChatRepository
from ..exceptions import DatabaseError


# Ejemplo 1: Unit of Work para transacciones atómicas
def create_chat_with_transaction(db: Session, user_id: str, title: str, content: str):
    """
    Ejemplo de uso de Unit of Work para transacciones atómicas.
    """
    with unit_of_work(db) as uow:
        chat_repo = ChatRepository(db)
        
        # Crear chat
        chat = chat_repo.create(
            id="chat_123",
            user_id=user_id,
            title=title,
            chat_content=content,
            created_at=datetime.utcnow()
        )
        
        # Si todo va bien, se hace commit automático
        # Si hay error, se hace rollback automático
        uow.commit()
        
        return chat


# Ejemplo 2: Caché para mejorar performance
@cached(key_prefix="chat", ttl=300)  # Cache por 5 minutos
def get_chat_cached(chat_repo: ChatRepository, chat_id: str):
    """
    Ejemplo de función con caché.
    La primera llamada consulta la DB, las siguientes usan caché.
    """
    return chat_repo.get_by_id(chat_id)


# Ejemplo 3: Medición de performance
@measure_time
def process_chats(chats):
    """
    Ejemplo de función con medición automática de tiempo.
    """
    results = []
    for chat in chats:
        # Procesar chat
        results.append(process_single_chat(chat))
    return results


def process_single_chat(chat):
    """Procesar un solo chat."""
    # Simular procesamiento
    return {"id": chat.id, "processed": True}


# Ejemplo 4: Context manager para timing
def search_chats_with_timing(chat_repo: ChatRepository, query: str):
    """
    Ejemplo de uso de timer context manager.
    """
    with timer("Search operation"):
        results = chat_repo.search_by_query(query, skip=0, limit=20)
        # El tiempo se loggea automáticamente
        return results


# Ejemplo 5: Retry para operaciones que pueden fallar
@retry(max_attempts=3, delay=1.0, exceptions=(DatabaseError,))
def unreliable_database_operation(db: Session):
    """
    Ejemplo de operación con retry automático.
    Si falla, reintenta hasta 3 veces con backoff exponencial.
    """
    # Operación que puede fallar temporalmente
    result = db.execute("SELECT * FROM chats")
    return result


# Ejemplo 6: Validación mejorada
def validate_chat_data(title: str, description: str, content: str):
    """
    Ejemplo de uso de validaciones mejoradas.
    """
    from ..utils import validate_length, validate_not_empty
    
    # Validaciones con mensajes de error claros
    title = validate_length(
        validate_not_empty(title, "title"),
        min_length=1,
        max_length=200,
        field_name="title"
    )
    
    if description:
        description = validate_length(
            description,
            max_length=1000,
            field_name="description"
        )
    
    content = validate_length(
        validate_not_empty(content, "content"),
        min_length=1,
        max_length=50000,
        field_name="content"
    )
    
    return {
        "title": title,
        "description": description,
        "content": content
    }


# Ejemplo 7: Combinando múltiples mejoras
@cached(key_prefix="trending", ttl=600)  # Cache por 10 minutos
@measure_time
@retry(max_attempts=2, delay=0.5)
def get_trending_chats_improved(chat_repo: ChatRepository, period: str = "day"):
    """
    Ejemplo combinando caché, medición de tiempo y retry.
    """
    with timer("Get trending chats"):
        return chat_repo.get_trending(
            hours=24 if period == "day" else 168,
            limit=20
        )


# Ejemplo 8: Uso de caché manual
def get_chat_with_manual_cache(chat_repo: ChatRepository, chat_id: str):
    """
    Ejemplo de uso manual del caché.
    """
    cache = get_cache()
    
    # Intentar obtener del caché
    cached_chat = cache.get(f"chat:{chat_id}")
    if cached_chat:
        return cached_chat
    
    # Si no está en caché, obtener de DB
    chat = chat_repo.get_by_id(chat_id)
    
    if chat:
        # Guardar en caché
        cache.set(f"chat:{chat_id}", chat, ttl=300)
    
    return chat


# Ejemplo 9: Performance monitoring
def monitor_operation_performance():
    """
    Ejemplo de monitoreo de performance.
    """
    import time
    from ..utils import get_performance_monitor
    
    monitor = get_performance_monitor()
    
    # Ejecutar operaciones
    for i in range(10):
        start = datetime.utcnow()
        # Simular operación
        time.sleep(0.1)
        duration = (datetime.utcnow() - start).total_seconds()
        monitor.record("operation", duration)
    
    # Obtener estadísticas
    stats = monitor.get_stats("operation")
    print(f"Operation stats: {stats}")
    # Output: {'count': 10, 'min': 0.1, 'max': 0.1, 'avg': 0.1, 'total': 1.0}


if __name__ == "__main__":
    print("Ejemplos de uso mejorado de Lovable Community")
    print("Ver código fuente para detalles de implementación")

