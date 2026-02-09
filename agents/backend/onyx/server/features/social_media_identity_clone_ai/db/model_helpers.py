"""
Helper functions for database model operations (upsert, get_or_create).
Eliminates repetitive update-or-create patterns.
"""

from typing import TypeVar, Type, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


def upsert_model(
    db: Session,
    model_class: Type[T],
    identifier: Dict[str, Any],
    update_data: Dict[str, Any],
    create_data: Optional[Dict[str, Any]] = None,
    auto_timestamp: bool = True
) -> T:
    """
    Actualiza un modelo existente o crea uno nuevo (upsert).
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo SQLAlchemy
        identifier: Diccionario con campos para identificar el modelo (ej: {"id": "123"})
        update_data: Datos para actualizar (si existe) o crear (si no existe)
        create_data: Datos adicionales solo para creación (opcional)
        auto_timestamp: Si actualizar updated_at automáticamente (default: True)
        
    Returns:
        Instancia del modelo (actualizado o creado)
        
    Usage:
        identity = upsert_model(
            db,
            IdentityProfileModel,
            identifier={"id": identity.profile_id},
            update_data={
                "username": identity.username,
                "display_name": identity.display_name,
            }
        )
    """
    # Buscar modelo existente
    existing = db.query(model_class).filter_by(**identifier).first()
    
    if existing:
        # Actualizar campos
        for key, value in update_data.items():
            setattr(existing, key, value)
        
        if auto_timestamp and hasattr(existing, 'updated_at'):
            existing.updated_at = datetime.utcnow()
        
        logger.debug(f"Updated {model_class.__name__} with {identifier}")
        return existing
    else:
        # Crear nuevo
        create_dict = {**update_data}
        if create_data:
            create_dict.update(create_data)
        
        # Agregar identifier si no está en update_data
        for key, value in identifier.items():
            if key not in create_dict:
                create_dict[key] = value
        
        if auto_timestamp:
            now = datetime.utcnow()
            if 'created_at' not in create_dict and hasattr(model_class, 'created_at'):
                create_dict['created_at'] = now
            if 'updated_at' not in create_dict and hasattr(model_class, 'updated_at'):
                create_dict['updated_at'] = now
        
        new_model = model_class(**create_dict)
        db.add(new_model)
        logger.debug(f"Created new {model_class.__name__} with {identifier}")
        return new_model


def get_or_create(
    db: Session,
    model_class: Type[T],
    identifier: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None
) -> Tuple[T, bool]:
    """
    Obtiene un modelo o lo crea si no existe.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo SQLAlchemy
        identifier: Campos para identificar el modelo
        defaults: Valores por defecto si se crea nuevo
        
    Returns:
        Tupla (modelo, created) donde created es True si se creó nuevo
        
    Usage:
        identity, created = get_or_create(
            db,
            IdentityProfileModel,
            identifier={"id": identity_id},
            defaults={"username": "default"}
        )
    """
    existing = db.query(model_class).filter_by(**identifier).first()
    
    if existing:
        return existing, False
    
    create_data = {**identifier}
    if defaults:
        create_data.update(defaults)
    
    if hasattr(model_class, 'created_at'):
        create_data['created_at'] = datetime.utcnow()
    if hasattr(model_class, 'updated_at'):
        create_data['updated_at'] = datetime.utcnow()
    
    new_model = model_class(**create_data)
    db.add(new_model)
    logger.debug(f"Created new {model_class.__name__} with {identifier}")
    return new_model, True








