"""
Aggregation Testing Helpers
Specialized helpers for data aggregation testing
"""

from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock
from collections import defaultdict


class AggregationTestHelpers:
    """Helpers for aggregation testing"""
    
    @staticmethod
    def create_mock_aggregator(
        aggregation_func: Optional[Callable] = None
    ) -> Mock:
        """Create mock aggregator"""
        aggregator = Mock()
        
        if aggregation_func:
            aggregator.aggregate = Mock(side_effect=aggregation_func)
        else:
            aggregator.aggregate = Mock(return_value={})
        
        return aggregator
    
    @staticmethod
    def assert_aggregation_performed(
        aggregator: Mock,
        expected_result: Optional[Any] = None
    ):
        """Assert aggregation was performed"""
        assert aggregator.aggregate.called, "Aggregation was not performed"
        
        if expected_result is not None:
            call_args = aggregator.aggregate.call_args
            if call_args:
                result = aggregator.aggregate(*call_args[0], **call_args[1])
                assert result == expected_result, \
                    f"Aggregation result {result} does not match expected {expected_result}"


class GroupingHelpers:
    """Helpers for grouping testing"""
    
    @staticmethod
    def group_by_key(
        items: List[Dict[str, Any]],
        key: str
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """Group items by key"""
        grouped = defaultdict(list)
        for item in items:
            grouped[item.get(key)].append(item)
        return dict(grouped)
    
    @staticmethod
    def assert_grouped_correctly(
        grouped: Dict[Any, List[Dict[str, Any]]],
        expected_groups: int,
        key: str
    ):
        """Assert items were grouped correctly"""
        assert len(grouped) == expected_groups, \
            f"Number of groups {len(grouped)} does not match expected {expected_groups}"
        
        for group_key, items in grouped.items():
            for item in items:
                assert item.get(key) == group_key, \
                    f"Item {item} does not belong to group {group_key}"


class StatisticsHelpers:
    """Helpers for statistics testing"""
    
    @staticmethod
    def calculate_statistics(
        values: List[float]
    ) -> Dict[str, float]:
        """Calculate basic statistics"""
        if not values:
            return {}
        
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }
    
    @staticmethod
    def assert_statistics_valid(
        stats: Dict[str, float],
        expected_count: Optional[int] = None
    ):
        """Assert statistics are valid"""
        assert "count" in stats, "Statistics missing count"
        assert "mean" in stats, "Statistics missing mean"
        
        if expected_count:
            assert stats["count"] == expected_count, \
                f"Count {stats['count']} does not match expected {expected_count}"


# Convenience exports
create_mock_aggregator = AggregationTestHelpers.create_mock_aggregator
assert_aggregation_performed = AggregationTestHelpers.assert_aggregation_performed

group_by_key = GroupingHelpers.group_by_key
assert_grouped_correctly = GroupingHelpers.assert_grouped_correctly

calculate_statistics = StatisticsHelpers.calculate_statistics
assert_statistics_valid = StatisticsHelpers.assert_statistics_valid



