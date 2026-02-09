"""
Search Testing Helpers
Specialized helpers for search functionality testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock


class SearchTestHelpers:
    """Helpers for search testing"""
    
    @staticmethod
    def create_mock_search_engine(
        results: Optional[Dict[str, List[Any]]] = None
    ) -> Mock:
        """Create mock search engine"""
        search_results = results or {}
        engine = Mock()
        
        async def search_side_effect(query: str, filters: Optional[Dict[str, Any]] = None):
            key = query.lower()
            return search_results.get(key, [])
        
        engine.search = AsyncMock(side_effect=search_side_effect)
        engine.index = AsyncMock()
        engine.delete = AsyncMock()
        engine.results = search_results
        return engine
    
    @staticmethod
    def assert_search_performed(
        engine: Mock,
        query: str,
        expected_results_count: Optional[int] = None
    ):
        """Assert search was performed"""
        assert engine.search.called, f"Search for '{query}' was not performed"
        
        if expected_results_count is not None and hasattr(engine, "results"):
            key = query.lower()
            results = engine.results.get(key, [])
            assert len(results) == expected_results_count, \
                f"Expected {expected_results_count} results, got {len(results)}"


class FilterHelpers:
    """Helpers for filter testing"""
    
    @staticmethod
    def create_mock_filter(
        filter_func: Optional[Callable] = None
    ) -> Mock:
        """Create mock filter"""
        filter_obj = Mock()
        
        if filter_func:
            filter_obj.apply = Mock(side_effect=filter_func)
        else:
            filter_obj.apply = Mock(return_value=True)
        
        return filter_obj
    
    @staticmethod
    def assert_filter_applied(filter_obj: Mock, data: Any):
        """Assert filter was applied"""
        assert filter_obj.apply.called, "Filter was not applied"


class SortingHelpers:
    """Helpers for sorting testing"""
    
    @staticmethod
    def assert_sorted(
        items: List[Any],
        key: Callable,
        reverse: bool = False
    ):
        """Assert items are sorted"""
        sorted_items = sorted(items, key=key, reverse=reverse)
        assert items == sorted_items, "Items are not sorted correctly"
    
    @staticmethod
    def assert_sorted_by_field(
        items: List[Dict[str, Any]],
        field: str,
        reverse: bool = False
    ):
        """Assert items are sorted by field"""
        sorted_items = sorted(
            items,
            key=lambda x: x.get(field),
            reverse=reverse
        )
        assert items == sorted_items, \
            f"Items are not sorted by field {field}"


# Convenience exports
create_mock_search_engine = SearchTestHelpers.create_mock_search_engine
assert_search_performed = SearchTestHelpers.assert_search_performed

create_mock_filter = FilterHelpers.create_mock_filter
assert_filter_applied = FilterHelpers.assert_filter_applied

assert_sorted = SortingHelpers.assert_sorted
assert_sorted_by_field = SortingHelpers.assert_sorted_by_field



