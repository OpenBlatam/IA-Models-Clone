from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Any, Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import colorsys
import re
from ..base import ModelField, ValidationMixin, CacheMixin, EventMixin, IndexMixin, PermissionMixin, StatusMixin
        from prometheus_client import Counter
        import structlog
        from prometheus_client import Counter
        import structlog
        import numpy as np
        import pandas as pd
        from prometheus_client import Counter
        import structlog
        import orjson
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Brand Kit Color Component - Onyx Integration
Component for managing brand colors with advanced features.
"""

@dataclass
class BrandKitColor:
    """Brand Kit Color Component with advanced color management"""
    name: str
    hex: str
    category: Literal['primary', 'secondary', 'accent', 'neutral', 'gradient', 'semantic'] = 'primary'
    description: Optional[str] = None
    rgb: Optional[Dict[str, int]] = None
    hsl: Optional[Dict[str, float]] = None
    opacity: float = 1.0
    is_dark: bool = False
    contrast_ratio: Optional[float] = None
    usage_guidelines: Optional[str] = None
    semantic_meaning: Optional[str] = None
    accessibility_level: Literal['AAA', 'AA', 'A', 'B'] = 'AA'
    color_blind_safe: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = '1.0.0'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> Any:
        """Initialize color field with validation and caching"""
        self.color_field = ModelField(
            name=self.name,
            value=self.hex,
            required=True,
            validation={
                'type': 'string',
                'pattern': '^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$',
                'min_length': 7,
                'max_length': 9,
                'timeout': 0.5
            },
            cache={
                'enabled': True,
                'ttl': 3600,
                'prefix': 'brand_kit:color'
            },
            events={
                'enabled': True,
                'types': ['color_created', 'color_updated', 'color_deleted'],
                'notify': True
            },
            index={
                'enabled': True,
                'fields': ['name', 'category', 'is_dark', 'accessibility_level'],
                'type': 'hash'
            },
            permissions={
                'roles': ['admin', 'designer'],
                'actions': ['create', 'read', 'update', 'delete']
            },
            status={
                'active': True,
                'archived': False
            }
        )
        self._calculate_color_properties()

    def _calculate_color_properties(self) -> Any:
        """Calculate advanced color properties"""
        if self.hex:
            # Convert hex to RGB
            hex_value = self.hex.lstrip('#')
            self.rgb = {
                'r': int(hex_value[0:2], 16),
                'g': int(hex_value[2:4], 16),
                'b': int(hex_value[4:6], 16)
            }
            
            # Calculate HSL
            r, g, b = self.rgb['r'] / 255, self.rgb['g'] / 255, self.rgb['b'] / 255
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            
            self.hsl = {
                'h': round(h * 360, 2),
                's': round(s * 100, 2),
                'l': round(l * 100, 2)
            }
            
            # Calculate if color is dark
            self.is_dark = l < 0.5
            
            # Calculate contrast ratio
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            self.contrast_ratio = round((luminance + 0.05) / 0.05, 2)
            
            # Calculate accessibility level
            self._calculate_accessibility_level()

    def _calculate_accessibility_level(self) -> Any:
        """Calculate WCAG accessibility level"""
        if self.contrast_ratio >= 7.0:
            self.accessibility_level = 'AAA'
        elif self.contrast_ratio >= 4.5:
            self.accessibility_level = 'AA'
        elif self.contrast_ratio >= 3.0:
            self.accessibility_level = 'A'
        else:
            self.accessibility_level = 'B'

    def get_data(self) -> Dict[str, Any]:
        """Get color data with caching"""
        cache_key = f"brand_kit:color:{self.name}"
        cached_data = self.color_field.get_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        data = {
            'name': self.name,
            'hex': self.hex,
            'category': self.category,
            'description': self.description,
            'rgb': self.rgb,
            'hsl': self.hsl,
            'opacity': self.opacity,
            'is_dark': self.is_dark,
            'contrast_ratio': self.contrast_ratio,
            'usage_guidelines': self.usage_guidelines,
            'semantic_meaning': self.semantic_meaning,
            'accessibility_level': self.accessibility_level,
            'color_blind_safe': self.color_blind_safe,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'metadata': self.metadata
        }
        
        self.color_field.set_cache(cache_key, data)
        return data

    def update(self, **kwargs) -> None:
        """Update color properties"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.utcnow()
        self._calculate_color_properties()
        self.color_field.clear_cache(f"brand_kit:color:{self.name}")

    def get_contrast_color(self) -> str:
        """Get contrasting color (black or white)"""
        return '#000000' if not self.is_dark else '#FFFFFF'

    def get_accessible_combinations(self) -> List[Dict[str, Any]]:
        """Get accessible color combinations"""
        if not self.contrast_ratio:
            return []
            
        combinations = []
        base_contrast = self.contrast_ratio
        
        # Add white and black combinations
        combinations.append({
            'background': self.hex,
            'text': '#FFFFFF' if self.is_dark else '#000000',
            'contrast_ratio': base_contrast,
            'accessibility_level': self.accessibility_level
        })
        
        # Add complementary color
        if self.hsl:
            complementary_hue = (self.hsl['h'] + 180) % 360
            complementary_hex = self._hsl_to_hex(complementary_hue, self.hsl['s'], self.hsl['l'])
            combinations.append({
                'background': self.hex,
                'text': complementary_hex,
                'contrast_ratio': base_contrast,
                'accessibility_level': self.accessibility_level
            })
        
        # Add triadic colors
        triadic_hue1 = (self.hsl['h'] + 120) % 360
        triadic_hue2 = (self.hsl['h'] + 240) % 360
        triadic_hex1 = self._hsl_to_hex(triadic_hue1, self.hsl['s'], self.hsl['l'])
        triadic_hex2 = self._hsl_to_hex(triadic_hue2, self.hsl['s'], self.hsl['l'])
        
        combinations.extend([
            {
                'background': self.hex,
                'text': triadic_hex1,
                'contrast_ratio': base_contrast,
                'accessibility_level': self.accessibility_level
            },
            {
                'background': self.hex,
                'text': triadic_hex2,
                'contrast_ratio': base_contrast,
                'accessibility_level': self.accessibility_level
            }
        ])
        
        return combinations

    def get_color_variations(self) -> Dict[str, str]:
        """Get color variations (lighter, darker, etc.)"""
        if not self.hsl:
            return {}
            
        variations = {}
        h, s, l = self.hsl['h'], self.hsl['s'], self.hsl['l']
        
        # Lighter variations
        variations['lighter'] = self._hsl_to_hex(h, s, min(l + 20, 100))
        variations['light'] = self._hsl_to_hex(h, s, min(l + 10, 100))
        
        # Darker variations
        variations['darker'] = self._hsl_to_hex(h, s, max(l - 20, 0))
        variations['dark'] = self._hsl_to_hex(h, s, max(l - 10, 0))
        
        # Saturated variations
        variations['saturated'] = self._hsl_to_hex(h, min(s + 20, 100), l)
        variations['desaturated'] = self._hsl_to_hex(h, max(s - 20, 0), l)
        
        return variations

    def get_css_variables(self) -> Dict[str, str]:
        """Get CSS variables for the color"""
        variables = {
            f'--color-{self.name}': self.hex,
            f'--color-{self.name}-rgb': f"{self.rgb['r']}, {self.rgb['g']}, {self.rgb['b']}",
            f'--color-{self.name}-hsl': f"{self.hsl['h']}, {self.hsl['s']}%, {self.hsl['l']}%"
        }
        
        # Add variations
        variations = self.get_color_variations()
        for name, hex_value in variations.items():
            variables[f'--color-{self.name}-{name}'] = hex_value
        
        return variables

    def _hsl_to_hex(self, h: float, s: float, l: float) -> str:
        """Convert HSL to hex color"""
        h /= 360
        s /= 100
        l /= 100
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t) -> Any:
                if t < 0:
                    t += 1
                if t > 1:
                    t -= 1
                if t < 1/6:
                    return p + (q - p) * 6 * t
                if t < 1/2:
                    return q
                if t < 2/3:
                    return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'

    def is_color_blind_safe(self) -> bool:
        """Check if color is safe for color blind users"""
        if not self.rgb:
            return False
            
        # Simplified color blind safety check
        r, g, b = self.rgb['r'], self.rgb['g'], self.rgb['b']
        
        # Check for red-green color blindness
        red_green_ratio = abs(r - g) / 255
        if red_green_ratio < 0.3:
            return False
            
        # Check for blue-yellow color blindness
        blue_yellow_ratio = abs((r + g) / 2 - b) / 255
        if blue_yellow_ratio < 0.3:
            return False
            
        return True

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> 'BrandKitColor':
        """Create color from data"""
        return cls(**data)

    # --- Batch Methods & ML/LLM Ready Mejorados ---
    @classmethod
    def batch_to_dicts(cls, objs: List["BrandKitColor"]) -> List[dict]:
        """Convierte una lista de BrandKitColor a lista de dicts, con métricas y logging."""
        logger = structlog.get_logger()
        metric = Counter('brandkitcolor_batch_to_dicts_total', 'Total batch_to_dicts calls')
        metric.inc()
        try:
            dicts = [obj.get_data() for obj in objs]
            return dicts
        except Exception as e:
            logger.error("batch_to_dicts_error", error=str(e))
            raise

    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List["BrandKitColor"]:
        """Convierte una lista de dicts a BrandKitColor, con métricas y logging."""
        logger = structlog.get_logger()
        metric = Counter('brandkitcolor_batch_from_dicts_total', 'Total batch_from_dicts calls')
        metric.inc()
        try:
            return [cls.from_data(d) for d in dicts]
        except Exception as e:
            logger.error("batch_from_dicts_error", error=str(e))
            raise

    @classmethod
    def batch_to_numpy(cls, objs: List["BrandKitColor"]):
        """Convierte una lista de BrandKitColor a un array numpy."""
        dicts = cls.batch_to_dicts(objs)
        return np.array(dicts)

    @classmethod
    def batch_to_pandas(cls, objs: List["BrandKitColor"]):
        """Convierte una lista de BrandKitColor a un DataFrame pandas."""
        dicts = cls.batch_to_dicts(objs)
        return pd.DataFrame(dicts)

    @classmethod
    def batch_deduplicate(cls, objs: List["BrandKitColor"], key="name") -> List["BrandKitColor"]:
        """Elimina duplicados por key, validando unicidad y tipos."""
        logger = structlog.get_logger()
        metric = Counter('brandkitcolor_batch_deduplicate_total', 'Total batch_deduplicate calls')
        metric.inc()
        seen = set()
        result = []
        for obj in objs:
            k = getattr(obj, key, None)
            if not isinstance(k, str):
                logger.warning("batch_deduplicate_nonstring_key", key=key, value=k)
                continue
            if k not in seen:
                seen.add(k)
                result.append(obj)
        return result

    @classmethod
    def to_training_example(cls, obj: "BrandKitColor") -> dict:
        """Convierte un BrandKitColor a ejemplo de entrenamiento ML/LLM."""
        return orjson.loads(orjson.dumps(obj.get_data()))

    @classmethod
    def from_training_example(cls, data: dict) -> "BrandKitColor":
        """Crea BrandKitColor desde ejemplo de entrenamiento."""
        return cls.from_data(data)

    @classmethod
    def batch_to_training_examples(cls, objs: List["BrandKitColor"]) -> List[dict]:
        """Batch a ejemplos de entrenamiento."""
        return [cls.to_training_example(obj) for obj in objs]

    @classmethod
    def batch_from_training_examples(cls, dicts: List[dict]) -> List["BrandKitColor"]:
        """Batch desde ejemplos de entrenamiento."""
        return [cls.from_training_example(d) for d in dicts] 