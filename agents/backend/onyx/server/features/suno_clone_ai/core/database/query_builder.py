"""
Query Builder

Utilities for building SQL queries.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Build SQL queries."""
    
    def __init__(self, table: str):
        """
        Initialize query builder.
        
        Args:
            table: Table name
        """
        self.table = table
        self.query_type = None
        self.columns = []
        self.where_conditions = []
        self.order_by = []
        self.limit_value = None
        self.values = {}
    
    def select(self, *columns: str) -> 'QueryBuilder':
        """
        Build SELECT query.
        
        Args:
            *columns: Column names
            
        Returns:
            Self for chaining
        """
        self.query_type = "SELECT"
        self.columns = list(columns) if columns else ["*"]
        return self
    
    def insert(self, **values: Any) -> 'QueryBuilder':
        """
        Build INSERT query.
        
        Args:
            **values: Column values
            
        Returns:
            Self for chaining
        """
        self.query_type = "INSERT"
        self.values = values
        return self
    
    def update(self, **values: Any) -> 'QueryBuilder':
        """
        Build UPDATE query.
        
        Args:
            **values: Column values
            
        Returns:
            Self for chaining
        """
        self.query_type = "UPDATE"
        self.values = values
        return self
    
    def where(self, condition: str, value: Any = None) -> 'QueryBuilder':
        """
        Add WHERE condition.
        
        Args:
            condition: Condition string
            value: Condition value
            
        Returns:
            Self for chaining
        """
        self.where_conditions.append((condition, value))
        return self
    
    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        """
        Add ORDER BY clause.
        
        Args:
            column: Column name
            desc: Descending order
            
        Returns:
            Self for chaining
        """
        self.order_by.append((column, desc))
        return self
    
    def limit(self, count: int) -> 'QueryBuilder':
        """
        Add LIMIT clause.
        
        Args:
            count: Limit count
            
        Returns:
            Self for chaining
        """
        self.limit_value = count
        return self
    
    def build(self) -> tuple:
        """
        Build query string and parameters.
        
        Returns:
            (query_string, parameters)
        """
        if self.query_type == "SELECT":
            return self._build_select()
        elif self.query_type == "INSERT":
            return self._build_insert()
        elif self.query_type == "UPDATE":
            return self._build_update()
        else:
            raise ValueError("No query type specified")
    
    def _build_select(self) -> tuple:
        """Build SELECT query."""
        columns = ", ".join(self.columns)
        query = f"SELECT {columns} FROM {self.table}"
        params = []
        
        if self.where_conditions:
            conditions = []
            for condition, value in self.where_conditions:
                if value is not None:
                    conditions.append(f"{condition} = ?")
                    params.append(value)
                else:
                    conditions.append(condition)
            query += " WHERE " + " AND ".join(conditions)
        
        if self.order_by:
            order_clauses = [
                f"{col} {'DESC' if desc else 'ASC'}"
                for col, desc in self.order_by
            ]
            query += " ORDER BY " + ", ".join(order_clauses)
        
        if self.limit_value:
            query += f" LIMIT {self.limit_value}"
        
        return query, tuple(params)
    
    def _build_insert(self) -> tuple:
        """Build INSERT query."""
        columns = ", ".join(self.values.keys())
        placeholders = ", ".join(["?"] * len(self.values))
        query = f"INSERT INTO {self.table} ({columns}) VALUES ({placeholders})"
        params = tuple(self.values.values())
        
        return query, params
    
    def _build_update(self) -> tuple:
        """Build UPDATE query."""
        set_clauses = [f"{col} = ?" for col in self.values.keys()]
        query = f"UPDATE {self.table} SET {", ".join(set_clauses)}"
        params = list(self.values.values())
        
        if self.where_conditions:
            conditions = []
            for condition, value in self.where_conditions:
                if value is not None:
                    conditions.append(f"{condition} = ?")
                    params.append(value)
                else:
                    conditions.append(condition)
            query += " WHERE " + " AND ".join(conditions)
        
        return query, tuple(params)


def create_query(table: str) -> QueryBuilder:
    """Create query builder."""
    return QueryBuilder(table)


def build_select(
    table: str,
    *columns: str,
    **kwargs
) -> tuple:
    """Build SELECT query."""
    builder = QueryBuilder(table).select(*columns)
    
    if 'where' in kwargs:
        builder.where(kwargs['where'], kwargs.get('where_value'))
    if 'order_by' in kwargs:
        builder.order_by(kwargs['order_by'], kwargs.get('desc', False))
    if 'limit' in kwargs:
        builder.limit(kwargs['limit'])
    
    return builder.build()


def build_insert(
    table: str,
    **values: Any
) -> tuple:
    """Build INSERT query."""
    builder = QueryBuilder(table).insert(**values)
    return builder.build()


def build_update(
    table: str,
    where: str,
    where_value: Any,
    **values: Any
) -> tuple:
    """Build UPDATE query."""
    builder = QueryBuilder(table).update(**values).where(where, where_value)
    return builder.build()



