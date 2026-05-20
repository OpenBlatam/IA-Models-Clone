"""
Preset Manager for Color Grading AI
====================================

Manages user-created presets and favorites.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ColorPreset:
    """Color grading preset."""
    id: str
    name: str
    description: str
    color_params: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    is_favorite: bool = False
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorPreset":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)


class PresetManager:
    """
    Manages color grading presets.
    
    Features:
    - Create, update, delete presets
    - Favorites management
    - Search and filter
    - Usage tracking
    """
    
    def __init__(self, presets_dir: str = "presets"):
        """
        Initialize preset manager.
        
        Args:
            presets_dir: Directory for presets storage
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)
        self._presets: Dict[str, ColorPreset] = {}
        self._load_presets()
    
    def _load_presets(self):
        """Load presets from disk."""
        presets_file = self.presets_dir / "presets.json"
        if presets_file.exists():
            try:
                with open(presets_file, "r") as f:
                    data = json.load(f)
                
                for preset_data in data.get("presets", []):
                    preset = ColorPreset.from_dict(preset_data)
                    self._presets[preset.id] = preset
                
                logger.info(f"Loaded {len(self._presets)} presets")
            except Exception as e:
                logger.error(f"Error loading presets: {e}")
    
    def _save_presets(self):
        """Save presets to disk."""
        presets_file = self.presets_dir / "presets.json"
        data = {
            "presets": [preset.to_dict() for preset in self._presets.values()]
        }
        with open(presets_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_preset(
        self,
        name: str,
        description: str,
        color_params: Dict[str, Any],
        category: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new preset.
        
        Args:
            name: Preset name
            description: Preset description
            color_params: Color parameters
            category: Optional category
            tags: Optional tags
            
        Returns:
            Preset ID
        """
        import uuid
        preset_id = str(uuid.uuid4())
        now = datetime.now()
        
        preset = ColorPreset(
            id=preset_id,
            name=name,
            description=description,
            color_params=color_params,
            created_at=now,
            updated_at=now,
            category=category,
            tags=tags or []
        )
        
        self._presets[preset_id] = preset
        self._save_presets()
        
        logger.info(f"Created preset: {name} ({preset_id})")
        return preset_id
    
    def update_preset(
        self,
        preset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> bool:
        """
        Update a preset.
        
        Args:
            preset_id: Preset ID
            name: New name
            description: New description
            color_params: New color parameters
            **kwargs: Other fields to update
            
        Returns:
            True if updated
        """
        preset = self._presets.get(preset_id)
        if not preset:
            return False
        
        if name:
            preset.name = name
        if description:
            preset.description = description
        if color_params:
            preset.color_params = color_params
        if "category" in kwargs:
            preset.category = kwargs["category"]
        if "tags" in kwargs:
            preset.tags = kwargs["tags"]
        if "is_favorite" in kwargs:
            preset.is_favorite = kwargs["is_favorite"]
        
        preset.updated_at = datetime.now()
        self._save_presets()
        
        return True
    
    def delete_preset(self, preset_id: str) -> bool:
        """
        Delete a preset.
        
        Args:
            preset_id: Preset ID
            
        Returns:
            True if deleted
        """
        if preset_id in self._presets:
            del self._presets[preset_id]
            self._save_presets()
            return True
        return False
    
    def get_preset(self, preset_id: str) -> Optional[ColorPreset]:
        """Get preset by ID."""
        return self._presets.get(preset_id)
    
    def list_presets(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        favorites_only: bool = False
    ) -> List[ColorPreset]:
        """
        List presets with optional filtering.
        
        Args:
            category: Filter by category
            tags: Filter by tags
            favorites_only: Only return favorites
            
        Returns:
            List of presets
        """
        presets = list(self._presets.values())
        
        if category:
            presets = [p for p in presets if p.category == category]
        
        if tags:
            presets = [
                p for p in presets
                if any(tag in p.tags for tag in tags)
            ]
        
        if favorites_only:
            presets = [p for p in presets if p.is_favorite]
        
        # Sort by usage count (most used first)
        presets.sort(key=lambda p: p.usage_count, reverse=True)
        
        return presets
    
    def increment_usage(self, preset_id: str):
        """Increment usage count for preset."""
        preset = self._presets.get(preset_id)
        if preset:
            preset.usage_count += 1
            self._save_presets()
    
    def toggle_favorite(self, preset_id: str) -> bool:
        """
        Toggle favorite status.
        
        Args:
            preset_id: Preset ID
            
        Returns:
            New favorite status
        """
        preset = self._presets.get(preset_id)
        if preset:
            preset.is_favorite = not preset.is_favorite
            self._save_presets()
            return preset.is_favorite
        return False




