"""
Advanced query engine for KV cache.

This module provides sophisticated querying capabilities including
filtering, sorting, aggregation, and complex queries.
"""

import re
from typing import Dict, Any, List, Optional, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import operator
from functools import reduce


class QueryOperator(Enum):
    """Query operators."""
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    CONTAINS = "contains"  # Contains substring
    STARTS_WITH = "starts_with"  # Starts with
    ENDS_WITH = "ends_with"  # Ends with
    REGEX = "regex"  # Regular expression
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field doesn't exist


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class QueryCondition:
    """A single query condition."""
    field: str
    operator: QueryOperator
    value: Any
    logical_op: LogicalOperator = LogicalOperator.AND


@dataclass
class SortSpec:
    """Sort specification."""
    field: str
    order: str = "asc"  # "asc" or "desc"


@dataclass
class QueryResult:
    """Query execution result."""
    items: List[Any]
    total_count: int
    filtered_count: int
    execution_time: float
    query: 'CacheQuery'


@dataclass
class AggregationResult:
    """Aggregation result."""
    function: str
    field: str
    value: Any


class CacheQuery:
    """Query builder for cache operations."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.conditions: List[QueryCondition] = []
        self.sort_specs: List[SortSpec] = []
        self.limit: Optional[int] = None
        self.offset: int = 0
        self.fields: Optional[List[str]] = None
        
    def where(
        self,
        field: str,
        operator: Union[QueryOperator, str],
        value: Any,
        logical_op: LogicalOperator = LogicalOperator.AND
    ) -> 'CacheQuery':
        """Add a condition to the query."""
        if isinstance(operator, str):
            operator = QueryOperator(operator.lower())
            
        condition = QueryCondition(
            field=field,
            operator=operator,
            value=value,
            logical_op=logical_op
        )
        self.conditions.append(condition)
        return self
        
    def order_by(self, field: str, order: str = "asc") -> 'CacheQuery':
        """Add sorting to the query."""
        self.sort_specs.append(SortSpec(field=field, order=order))
        return self
        
    def limit_count(self, limit: int) -> 'CacheQuery':
        """Set the maximum number of results."""
        self.limit = limit
        return self
        
    def offset_count(self, offset: int) -> 'CacheQuery':
        """Set the offset for pagination."""
        self.offset = offset
        return self
        
    def select_fields(self, fields: List[str]) -> 'CacheQuery':
        """Select specific fields to return."""
        self.fields = fields
        return self
        
    def execute(self) -> QueryResult:
        """Execute the query."""
        import time
        start_time = time.time()
        
        # Get all items from cache
        all_items = list(self.cache._cache.values())
        total_count = len(all_items)
        
        # Apply filters
        filtered_items = self._apply_filters(all_items)
        filtered_count = len(filtered_items)
        
        # Apply sorting
        if self.sort_specs:
            filtered_items = self._apply_sorting(filtered_items)
            
        # Apply pagination
        if self.offset > 0:
            filtered_items = filtered_items[self.offset:]
        if self.limit:
            filtered_items = filtered_items[:self.limit]
            
        # Select fields
        if self.fields:
            filtered_items = self._select_fields(filtered_items)
            
        execution_time = time.time() - start_time
        
        return QueryResult(
            items=filtered_items,
            total_count=total_count,
            filtered_count=filtered_count,
            execution_time=execution_time,
            query=self
        )
        
    def _apply_filters(self, items: List[Any]) -> List[Any]:
        """Apply all filter conditions."""
        if not self.conditions:
            return items
            
        filtered = items
        
        for condition in self.conditions:
            filtered = [
                item for item in filtered
                if self._evaluate_condition(item, condition)
            ]
            
        return filtered
        
    def _evaluate_condition(self, item: Any, condition: QueryCondition) -> bool:
        """Evaluate a single condition against an item."""
        field_value = self._get_field_value(item, condition.field)
        
        if condition.operator == QueryOperator.EXISTS:
            return field_value is not None
        elif condition.operator == QueryOperator.NOT_EXISTS:
            return field_value is None
        elif field_value is None:
            return False
            
        op = condition.operator
        value = condition.value
        
        if op == QueryOperator.EQ:
            return field_value == value
        elif op == QueryOperator.NE:
            return field_value != value
        elif op == QueryOperator.GT:
            return field_value > value
        elif op == QueryOperator.GTE:
            return field_value >= value
        elif op == QueryOperator.LT:
            return field_value < value
        elif op == QueryOperator.LTE:
            return field_value <= value
        elif op == QueryOperator.IN:
            return field_value in value
        elif op == QueryOperator.NOT_IN:
            return field_value not in value
        elif op == QueryOperator.CONTAINS:
            return str(value) in str(field_value)
        elif op == QueryOperator.STARTS_WITH:
            return str(field_value).startswith(str(value))
        elif op == QueryOperator.ENDS_WITH:
            return str(field_value).endswith(str(value))
        elif op == QueryOperator.REGEX:
            return bool(re.search(str(value), str(field_value)))
        else:
            return False
            
    def _get_field_value(self, item: Any, field: str) -> Any:
        """Get field value from item (supports nested fields)."""
        if isinstance(item, dict):
            return item.get(field)
        elif hasattr(item, field):
            return getattr(item, field)
        elif '.' in field:
            # Nested field access
            parts = field.split('.')
            value = item
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return None
                if value is None:
                    return None
            return value
        return None
        
    def _apply_sorting(self, items: List[Any]) -> List[Any]:
        """Apply sorting to items."""
        def sort_key(item: Any) -> tuple:
            keys = []
            for spec in self.sort_specs:
                value = self._get_field_value(item, spec.field)
                # Handle None values
                if value is None:
                    value = float('-inf') if spec.order == "asc" else float('inf')
                keys.append(value)
            return tuple(keys)
            
        reverse = any(spec.order == "desc" for spec in self.sort_specs)
        return sorted(items, key=sort_key, reverse=reverse)
        
    def _select_fields(self, items: List[Any]) -> List[Any]:
        """Select specific fields from items."""
        result = []
        for item in items:
            if isinstance(item, dict):
                selected = {field: item.get(field) for field in self.fields}
            else:
                selected = {
                    field: getattr(item, field, None)
                    for field in self.fields
                }
            result.append(selected)
        return result


class CacheQueryEngine:
    """Advanced query engine for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        
    def query(self) -> CacheQuery:
        """Create a new query."""
        return CacheQuery(self.cache)
        
    def find(
        self,
        field: str,
        operator: Union[QueryOperator, str],
        value: Any
    ) -> List[Any]:
        """Find items matching a condition."""
        return self.query().where(field, operator, value).execute().items
        
    def find_one(
        self,
        field: str,
        operator: Union[QueryOperator, str],
        value: Any
    ) -> Optional[Any]:
        """Find a single item matching a condition."""
        results = self.find(field, operator, value)
        return results[0] if results else None
        
    def count(
        self,
        field: Optional[str] = None,
        operator: Optional[Union[QueryOperator, str]] = None,
        value: Optional[Any] = None
    ) -> int:
        """Count items matching optional conditions."""
        query = self.query()
        if field and operator and value:
            query.where(field, operator, value)
        return query.execute().filtered_count
        
    def aggregate(
        self,
        function: str,
        field: str,
        conditions: Optional[List[QueryCondition]] = None
    ) -> AggregationResult:
        """Perform aggregation on a field."""
        query = self.query()
        if conditions:
            for condition in conditions:
                query.conditions.append(condition)
                
        results = query.execute().items
        
        values = [
            query._get_field_value(item, field)
            for item in results
            if query._get_field_value(item, field) is not None
        ]
        
        if not values:
            return AggregationResult(function=function, field=field, value=None)
            
        if function == "sum":
            result_value = sum(values)
        elif function == "avg" or function == "mean":
            result_value = sum(values) / len(values)
        elif function == "min":
            result_value = min(values)
        elif function == "max":
            result_value = max(values)
        elif function == "count":
            result_value = len(values)
        else:
            raise ValueError(f"Unknown aggregation function: {function}")
            
        return AggregationResult(function=function, field=field, value=result_value)
        
    def group_by(
        self,
        group_field: str,
        aggregate_field: str,
        aggregate_function: str = "count"
    ) -> Dict[Any, Any]:
        """Group items by a field and aggregate."""
        query = self.query()
        results = query.execute().items
        
        groups = defaultdict(list)
        for item in results:
            group_key = query._get_field_value(item, group_field)
            groups[group_key].append(item)
            
        aggregated = {}
        for group_key, group_items in groups.items():
            values = [
                query._get_field_value(item, aggregate_field)
                for item in group_items
                if query._get_field_value(item, aggregate_field) is not None
            ]
            
            if aggregate_function == "count":
                aggregated[group_key] = len(values)
            elif aggregate_function == "sum":
                aggregated[group_key] = sum(values)
            elif aggregate_function == "avg":
                aggregated[group_key] = sum(values) / len(values) if values else 0
            elif aggregate_function == "min":
                aggregated[group_key] = min(values) if values else None
            elif aggregate_function == "max":
                aggregated[group_key] = max(values) if values else None
            else:
                aggregated[group_key] = values
                
        return aggregated
        
    def batch_query(self, queries: List[CacheQuery]) -> List[QueryResult]:
        """Execute multiple queries in batch."""
        return [query.execute() for query in queries]
















