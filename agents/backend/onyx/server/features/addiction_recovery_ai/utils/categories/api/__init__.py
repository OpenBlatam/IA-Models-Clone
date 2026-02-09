"""
API utilities
"""

from utils.categories import register_utility

try:
    from utils.api_docs import APIDocs
    from utils.api_versioning import APIVersioning
    from utils.response import Response
    from utils.response_builders import ResponseBuilders
    from utils.query_params import QueryParams
    from utils.pagination import Pagination
    
    def register_utilities():
        register_utility("api", "docs", APIDocs)
        register_utility("api", "versioning", APIVersioning)
        register_utility("api", "response", Response)
        register_utility("api", "response_builders", ResponseBuilders)
        register_utility("api", "query_params", QueryParams)
        register_utility("api", "pagination", Pagination)
except ImportError:
    pass



