"""
General helper utilities
"""

from utils.categories import register_utility

try:
    from utils.helpers import Helpers
    from utils.string_helpers import StringHelpers
    from utils.date_helpers import DateHelpers
    from utils.time_utils import TimeUtils
    from utils.math_helpers import MathHelpers
    from utils.file_utils import FileUtils
    from utils.decorators import Decorators
    
    def register_utilities():
        register_utility("helpers", "helpers", Helpers)
        register_utility("helpers", "string", StringHelpers)
        register_utility("helpers", "date", DateHelpers)
        register_utility("helpers", "time", TimeUtils)
        register_utility("helpers", "math", MathHelpers)
        register_utility("helpers", "file", FileUtils)
        register_utility("helpers", "decorators", Decorators)
except ImportError:
    pass



