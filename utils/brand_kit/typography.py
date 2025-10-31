from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from ..base_types import (
from ..model_field import ModelField, FieldConfig
from ..base import ValidationMixin, CacheMixin, EventMixin, IndexMixin, PermissionMixin, StatusMixin
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
Brand Kit Typography Component - Onyx Integration
Component for managing brand typography with advanced features.
"""
    CACHE_TTL, VALIDATION_TIMEOUT,
    ModelId, ModelKey, ModelValue,
    ValidationType, CacheType, EventType,
    StatusType, CategoryType, PermissionType
)

T = TypeVar('T')

@dataclass
class BrandKitTypography:
    """Brand Kit Typography Component with advanced features"""
    name: str
    font_family: str
    style: Literal['heading', 'body', 'display', 'monospace', 'serif', 'sans-serif'] = 'body'
    category: Literal['primary', 'secondary', 'accent'] = 'primary'
    weights: List[int] = field(default_factory=lambda: [400, 700])
    sizes: Dict[str, float] = field(default_factory=lambda: {
        'xs': 0.75,
        'sm': 0.875,
        'base': 1.0,
        'lg': 1.125,
        'xl': 1.25,
        '2xl': 1.5,
        '3xl': 1.875,
        '4xl': 2.25,
        '5xl': 3.0,
        '6xl': 3.75,
        '7xl': 4.5,
        '8xl': 6.0,
        '9xl': 8.0
    })
    line_heights: Dict[str, float] = field(default_factory=lambda: {
        'tight': 1.25,
        'normal': 1.5,
        'relaxed': 1.75,
        'loose': 2.0
    })
    letter_spacing: Dict[str, float] = field(default_factory=lambda: {
        'tighter': -0.05,
        'tight': -0.025,
        'normal': 0,
        'wide': 0.025,
        'wider': 0.05,
        'widest': 0.1
    })
    font_features: Dict[str, bool] = field(default_factory=lambda: {
        'liga': True,
        'calt': True,
        'kern': True,
        'frac': False,
        'ordn': False,
        'tnum': False,
        'pnum': False,
        'onum': False,
        'lnum': False
    })
    font_variants: Dict[str, str] = field(default_factory=lambda: {
        'normal': 'normal',
        'small-caps': 'small-caps',
        'all-small-caps': 'all-small-caps',
        'petite-caps': 'petite-caps',
        'all-petite-caps': 'all-petite-caps',
        'unicase': 'unicase',
        'titling-caps': 'titling-caps'
    })
    description: Optional[str] = None
    usage_guidelines: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = '1.0.0'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> Any:
        """Initialize typography field with validation and caching"""
        self.typography_field = ModelField(
            name=self.name,
            value=self.font_family,
            required=True,
            validation={
                'type': 'string',
                'min_length': 1,
                'timeout': 0.5,
                'rules': {
                    'font_family': '^[a-zA-Z0-9\\s\\-\\,\\.]+$',
                    'weights': '^[0-9\\,]+$',
                    'sizes': '^[0-9\\.\\,]+$'
                }
            },
            cache={
                'enabled': True,
                'ttl': 3600,
                'prefix': 'brand_kit:typography'
            },
            events={
                'enabled': True,
                'types': ['typography_created', 'typography_updated', 'typography_deleted'],
                'notify': True
            },
            index={
                'enabled': True,
                'fields': ['name', 'style', 'category'],
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

    def get_data(self) -> Dict[str, Any]:
        """Get typography data with caching"""
        cache_key = f"brand_kit:typography:{self.name}"
        cached_data = self.typography_field.get_cache(cache_key)
        
        if cached_data:
            return cached_data
        
        data = {
            'name': self.name,
            'font_family': self.font_family,
            'style': self.style,
            'category': self.category,
            'weights': self.weights,
            'sizes': self.sizes,
            'line_heights': self.line_heights,
            'letter_spacing': self.letter_spacing,
            'font_features': self.font_features,
            'font_variants': self.font_variants,
            'description': self.description,
            'usage_guidelines': self.usage_guidelines,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'metadata': self.metadata
        }
        
        self.typography_field.set_cache(cache_key, data)
        return data

    def update(self, **kwargs) -> None:
        """Update typography properties"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.updated_at = datetime.utcnow()
        self.typography_field.clear_cache(f"brand_kit:typography:{self.name}")

    def get_font_stack(self) -> str:
        """Get CSS font stack with fallbacks"""
        fallbacks = {
            'serif': 'Georgia, "Times New Roman", serif',
            'sans-serif': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
            'monospace': 'SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            'display': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'heading': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            'body': 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
        }
        
        return f'"{self.font_family}", {fallbacks.get(self.style, fallbacks["sans-serif"])}'

    def get_css_variables(self) -> Dict[str, str]:
        """Get CSS variables for typography"""
        variables = {
            f'--font-{self.name}': self.get_font_stack(),
            f'--font-{self.name}-weights': ','.join(map(str, self.weights))
        }
        
        # Add size variables
        for size, value in self.sizes.items():
            variables[f'--font-{self.name}-size-{size}'] = f'{value}rem'
        
        # Add line height variables
        for name, value in self.line_heights.items():
            variables[f'--font-{self.name}-line-height-{name}'] = str(value)
        
        # Add letter spacing variables
        for name, value in self.letter_spacing.items():
            variables[f'--font-{self.name}-letter-spacing-{name}'] = f'{value}em'
        
        # Add font feature variables
        for feature, enabled in self.font_features.items():
            variables[f'--font-{self.name}-feature-{feature}'] = '1' if enabled else '0'
        
        # Add font variant variables
        for variant, value in self.font_variants.items():
            variables[f'--font-{self.name}-variant-{variant}'] = value
        
        return variables

    def get_scale_ratio(self) -> float:
        """Calculate typographic scale ratio"""
        sizes = list(self.sizes.values())
        if len(sizes) < 2:
            return 1.25  # Default ratio
        
        ratios = []
        for i in range(1, len(sizes)):
            ratios.append(sizes[i] / sizes[i-1])
        
        return sum(ratios) / len(ratios)

    def get_optimal_line_height(self, font_size: float) -> float:
        """Calculate optimal line height for given font size"""
        base_size = self.sizes.get('base', 1.0)
        base_line_height = self.line_heights.get('normal', 1.5)
        
        # Adjust line height based on font size
        if font_size < base_size:
            return base_line_height + 0.2
        elif font_size > base_size:
            return base_line_height - 0.1
        return base_line_height

    def get_optimal_letter_spacing(self, font_size: float) -> float:
        """Calculate optimal letter spacing for given font size"""
        base_size = self.sizes.get('base', 1.0)
        base_spacing = self.letter_spacing.get('normal', 0)
        
        # Adjust letter spacing based on font size
        if font_size < base_size:
            return base_spacing + 0.01
        elif font_size > base_size:
            return base_spacing - 0.01
        return base_spacing

    def get_font_feature_settings(self) -> str:
        """Get CSS font-feature-settings string"""
        features = []
        for feature, enabled in self.font_features.items():
            if enabled:
                features.append(f'"{feature}" 1')
            else:
                features.append(f'"{feature}" 0')
        return ', '.join(features)

    def get_font_variant_settings(self) -> str:
        """Get CSS font-variant settings string"""
        variants = []
        for variant, value in self.font_variants.items():
            if value != 'normal':
                variants.append(value)
        return ' '.join(variants)

    def get_fluid_typography(self, min_size: float, max_size: float, min_width: float = 320, max_width: float = 1200) -> str:
        """Generate fluid typography CSS"""
        scale = self.get_scale_ratio()
        min_scale = min_size / self.sizes['base']
        max_scale = max_size / self.sizes['base']
        
        return f"""
            font-size: clamp({min_size}rem, {min_scale}rem + {((max_scale - min_scale) * 100) / (max_width - min_width)}vw, {max_size}rem);
            line-height: clamp({self.get_optimal_line_height(min_size)}, {self.get_optimal_line_height(min_size)} + {((self.get_optimal_line_height(max_size) - self.get_optimal_line_height(min_size)) * 100) / (max_width - min_width)}vw, {self.get_optimal_line_height(max_size)});
            letter-spacing: clamp({self.get_optimal_letter_spacing(min_size)}em, {self.get_optimal_letter_spacing(min_size)}em + {((self.get_optimal_letter_spacing(max_size) - self.get_optimal_letter_spacing(min_size)) * 100) / (max_width - min_width)}vw, {self.get_optimal_letter_spacing(max_size)}em);
        """

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> 'BrandKitTypography':
        """Create typography from data"""
        return cls(**data)

    # --- Batch Methods & ML/LLM Ready Mejorados ---
    @classmethod
    def batch_to_dicts(cls, objs: List["BrandKitTypography"]) -> List[dict]:
        """Convierte una lista de BrandKitTypography a lista de dicts, con métricas y logging."""
        logger = structlog.get_logger()
        metric = Counter('brandkittypography_batch_to_dicts_total', 'Total batch_to_dicts calls')
        metric.inc()
        try:
            dicts = [obj.get_data() for obj in objs]
            return dicts
        except Exception as e:
            logger.error("batch_to_dicts_error", error=str(e))
            raise

    @classmethod
    def batch_from_dicts(cls, dicts: List[dict]) -> List["BrandKitTypography"]:
        """Convierte una lista de dicts a BrandKitTypography, con métricas y logging."""
        logger = structlog.get_logger()
        metric = Counter('brandkittypography_batch_from_dicts_total', 'Total batch_from_dicts calls')
        metric.inc()
        try:
            return [cls.from_data(d) for d in dicts]
        except Exception as e:
            logger.error("batch_from_dicts_error", error=str(e))
            raise

    @classmethod
    def batch_to_numpy(cls, objs: List["BrandKitTypography"]):
        """Convierte una lista de BrandKitTypography a un array numpy."""
        dicts = cls.batch_to_dicts(objs)
        return np.array(dicts)

    @classmethod
    def batch_to_pandas(cls, objs: List["BrandKitTypography"]):
        """Convierte una lista de BrandKitTypography a un DataFrame pandas."""
        dicts = cls.batch_to_dicts(objs)
        return pd.DataFrame(dicts)

    @classmethod
    def batch_deduplicate(cls, objs: List["BrandKitTypography"], key="name") -> List["BrandKitTypography"]:
        """Elimina duplicados por key, validando unicidad y tipos."""
        logger = structlog.get_logger()
        metric = Counter('brandkittypography_batch_deduplicate_total', 'Total batch_deduplicate calls')
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
    def to_training_example(cls, obj: "BrandKitTypography") -> dict:
        """Convierte un BrandKitTypography a ejemplo de entrenamiento ML/LLM."""
        return orjson.loads(orjson.dumps(obj.get_data()))

    @classmethod
    def from_training_example(cls, data: dict) -> "BrandKitTypography":
        """Crea BrandKitTypography desde ejemplo de entrenamiento."""
        return cls.from_data(data)

    @classmethod
    def batch_to_training_examples(cls, objs: List["BrandKitTypography"]) -> List[dict]:
        """Batch a ejemplos de entrenamiento."""
        return [cls.to_training_example(obj) for obj in objs]

    @classmethod
    def batch_from_training_examples(cls, dicts: List[dict]) -> List["BrandKitTypography"]:
        """Batch desde ejemplos de entrenamiento."""
        return [cls.from_training_example(d) for d in dicts] 