from datetime import datetime
from typing import Dict, Any

from ...domain.entities import User, SkinType


class UserMapper:
    
    @staticmethod
    def to_dict(user: User) -> Dict[str, Any]:
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "skin_type": user.skin_type.value if user.skin_type else None,
            "preferences": user.preferences,
            "created_at": user.created_at.isoformat(),
            "updated_at": user.updated_at.isoformat(),
        }
    
    @staticmethod
    def to_entity(data: Dict[str, Any]) -> User:
        return User(
            id=data["id"],
            email=data["email"],
            name=data.get("name"),
            skin_type=SkinType(data["skin_type"]) if data.get("skin_type") else None,
            preferences=data.get("preferences", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
    
    @staticmethod
    def to_update_dict(user: User) -> Dict[str, Any]:
        return {
            "name": user.name,
            "skin_type": user.skin_type.value if user.skin_type else None,
            "preferences": user.preferences,
            "updated_at": user.updated_at.isoformat(),
        }















