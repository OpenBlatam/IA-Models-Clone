"""
Query Optimization Utilities
Provides hints and optimizations for database queries
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """Utilities for optimizing database queries"""
    
    @staticmethod
    def optimize_filter_conditions(
        filter_conditions: Optional[Dict[str, Any]],
        indexed_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize filter conditions by prioritizing indexed fields
        
        Args:
            filter_conditions: Original filter conditions
            indexed_fields: List of indexed field names
            
        Returns:
            Optimized filter conditions
        """
        if not filter_conditions:
            return {}
        
        if not indexed_fields:
            # Default indexed fields for common tables
            indexed_fields = ["id", "user_id", "created_at"]
        
        # Separate indexed and non-indexed conditions
        optimized = {}
        non_indexed = {}
        
        for key, value in filter_conditions.items():
            if key in indexed_fields:
                optimized[key] = value
            else:
                non_indexed[key] = value
        
        # Add non-indexed conditions after indexed ones
        optimized.update(non_indexed)
        
        return optimized
    
    @staticmethod
    def suggest_indexes(table_name: str, common_queries: List[Dict[str, Any]]) -> List[str]:
        """
        Suggest indexes based on common query patterns
        
        Args:
            table_name: Table name
            common_queries: List of common query patterns
            
        Returns:
            List of suggested index definitions
        """
        field_usage = {}
        
        for query in common_queries:
            filters = query.get("filter_conditions", {})
            for field in filters.keys():
                field_usage[field] = field_usage.get(field, 0) + 1
        
        # Sort by usage frequency
        sorted_fields = sorted(field_usage.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        for field, count in sorted_fields[:5]:  # Top 5 fields
            suggestions.append(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{field} ON {table_name}({field});")
        
        return suggestions
    
    @staticmethod
    def optimize_limit_offset(limit: int, offset: int, max_limit: int = 100) -> tuple:
        """
        Optimize limit and offset values
        
        Args:
            limit: Requested limit
            offset: Requested offset
            max_limit: Maximum allowed limit
            
        Returns:
            Optimized (limit, offset) tuple
        """
        # Clamp limit
        if limit > max_limit:
            logger.warning(f"Limit {limit} exceeds maximum {max_limit}, clamping")
            limit = max_limit
        
        if limit < 1:
            limit = 1
        
        # Ensure non-negative offset
        if offset < 0:
            offset = 0
        
        return limit, offset
    
    @staticmethod
    def build_query_hints(
        table_name: str,
        use_index: Optional[str] = None,
        force_index: bool = False
    ) -> Dict[str, Any]:
        """
        Build query hints for database optimization
        
        Args:
            table_name: Table name
            use_index: Index name to use
            force_index: Whether to force index usage
            
        Returns:
            Query hints dictionary
        """
        hints = {
            "table": table_name,
        }
        
        if use_index:
            hints["index"] = use_index
            hints["force_index"] = force_index
        
        return hints


class IndexManager:
    """Manage database indexes"""
    
    # Common indexes for dermatology_ai tables
    COMMON_INDEXES = {
        "analyses": [
            "CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);",
            "CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_analyses_user_status ON analyses(user_id, status);",
        ],
        "users": [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);",
        ],
        "products": [
            "CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);",
            "CREATE INDEX IF NOT EXISTS idx_products_skin_type ON products(skin_type);",
        ],
    }
    
    @staticmethod
    def get_indexes_for_table(table_name: str) -> List[str]:
        """Get recommended indexes for a table"""
        return IndexManager.COMMON_INDEXES.get(table_name, [])
    
    @staticmethod
    def get_all_indexes() -> Dict[str, List[str]]:
        """Get all recommended indexes"""
        return IndexManager.COMMON_INDEXES.copy()















