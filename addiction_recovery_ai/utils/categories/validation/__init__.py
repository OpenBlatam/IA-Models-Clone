"""
Validation utilities
"""

from utils.categories import register_utility

try:
    from utils.validators import Validators
    from utils.advanced_validation import AdvancedValidation
    from utils.validation_combinators import ValidationCombinators
    from utils.sanitizers import Sanitizers
    from utils.pydantic_helpers import PydanticHelpers
    
    def register_utilities():
        register_utility("validation", "validators", Validators)
        register_utility("validation", "advanced", AdvancedValidation)
        register_utility("validation", "combinators", ValidationCombinators)
        register_utility("validation", "sanitizers", Sanitizers)
        register_utility("validation", "pydantic", PydanticHelpers)
except ImportError:
    pass



