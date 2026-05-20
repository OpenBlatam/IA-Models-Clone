"""
Functional programming utilities
"""

from utils.categories import register_utility

try:
    from utils.functional_helpers import FunctionalHelpers
    from utils.functors import Functors
    from utils.composers import Composers
    from utils.monads import Monads
    from utils.lenses import Lenses
    from utils.predicates import Predicates
    from utils.guards import Guards
    from utils.trampolines import Trampolines
    from utils.iterators import Iterators
    from utils.generators import Generators
    from utils.streams import Streams
    from utils.transformers import Transformers
    
    def register_utilities():
        register_utility("functional", "helpers", FunctionalHelpers)
        register_utility("functional", "functors", Functors)
        register_utility("functional", "composers", Composers)
        register_utility("functional", "monads", Monads)
        register_utility("functional", "lenses", Lenses)
        register_utility("functional", "predicates", Predicates)
        register_utility("functional", "guards", Guards)
        register_utility("functional", "trampolines", Trampolines)
        register_utility("functional", "iterators", Iterators)
        register_utility("functional", "generators", Generators)
        register_utility("functional", "streams", Streams)
        register_utility("functional", "transformers", Transformers)
except ImportError:
    pass



