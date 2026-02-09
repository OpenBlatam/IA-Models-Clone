"""
Integrations Submodule
Aggregates various external library integrations.
"""

# Transformers integration
try:
    from .transformers import EnhancedTransformerWrapper
    try:
        from .transformers import LoRATransformerWrapper
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        LoRATransformerWrapper = None
        TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    EnhancedTransformerWrapper = None
    LoRATransformerWrapper = None

# Diffusion integration
try:
    from .diffusion import DiffusionSchedulerFactory, DiffusionPipelineWrapper
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    DiffusionSchedulerFactory = None
    DiffusionPipelineWrapper = None

# Backward compatibility - import from old locations
try:
    from .transformers_integration import HuggingFaceModelWrapper, TransformerMusicEncoder
except ImportError:
    HuggingFaceModelWrapper = None
    TransformerMusicEncoder = None

__all__ = []

if TRANSFORMERS_AVAILABLE:
    __all__.extend(["EnhancedTransformerWrapper"])
    if LoRATransformerWrapper is not None:
        __all__.append("LoRATransformerWrapper")

if DIFFUSERS_AVAILABLE:
    __all__.extend(["DiffusionSchedulerFactory", "DiffusionPipelineWrapper"])

if HuggingFaceModelWrapper is not None:
    __all__.extend(["HuggingFaceModelWrapper", "TransformerMusicEncoder"])



