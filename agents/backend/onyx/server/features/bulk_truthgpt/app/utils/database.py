"""
Database utilities for Ultimate Enhanced Supreme Production system
Following Flask best practices with functional programming patterns
"""

import time
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from flask import current_app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, func, desc, asc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker, scoped_session

logger = logging.getLogger(__name__)

# Global database instance
db = SQLAlchemy()

def init_database(app) -> None:
    """Initialize database with connection pooling."""
    try:
        db.init_app(app)
        
        # Configure connection pooling
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
            'pool_size': 20,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'max_overflow': 30
        }
        
        logger.info("🗄️ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

@contextmanager
def get_db_session():
    """Get database session with automatic cleanup."""
    session = db.session
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Database session error: {e}")
        raise
    finally:
        session.close()

def execute_query(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Execute raw SQL query with error handling."""
    try:
        with get_db_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]
    except SQLAlchemyError as e:
        logger.error(f"❌ Query execution failed: {e}")
        return []

def execute_single_query(query: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Execute single row query with early returns."""
    try:
        with get_db_session() as session:
            result = session.execute(text(query), params or {})
            row = result.fetchone()
            return dict(row._mapping) if row else None
    except SQLAlchemyError as e:
        logger.error(f"❌ Single query execution failed: {e}")
        return None

def insert_record(table_name: str, data: Dict[str, Any]) -> Optional[int]:
    """Insert record with early returns."""
    if not data:
        return None
    
    try:
        with get_db_session() as session:
            query = f"INSERT INTO {table_name} ({', '.join(data.keys())}) VALUES ({', '.join([f':{k}' for k in data.keys()])})"
            result = session.execute(text(query), data)
            return result.lastrowid
    except IntegrityError as e:
        logger.error(f"❌ Insert failed - integrity error: {e}")
        return None
    except SQLAlchemyError as e:
        logger.error(f"❌ Insert failed: {e}")
        return None

def update_record(table_name: str, data: Dict[str, Any], where_clause: str, where_params: Dict[str, Any] = None) -> bool:
    """Update record with early returns."""
    if not data:
        return False
    
    try:
        with get_db_session() as session:
            set_clause = ', '.join([f"{k} = :{k}" for k in data.keys()])
            query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"
            
            # Merge data and where_params
            params = {**data, **(where_params or {})}
            result = session.execute(text(query), params)
            
            return result.rowcount > 0
    except SQLAlchemyError as e:
        logger.error(f"❌ Update failed: {e}")
        return False

def delete_record(table_name: str, where_clause: str, where_params: Dict[str, Any] = None) -> bool:
    """Delete record with early returns."""
    try:
        with get_db_session() as session:
            query = f"DELETE FROM {table_name} WHERE {where_clause}"
            result = session.execute(text(query), where_params or {})
            
            return result.rowcount > 0
    except SQLAlchemyError as e:
        logger.error(f"❌ Delete failed: {e}")
        return False

def get_record_by_id(table_name: str, record_id: int, id_column: str = 'id') -> Optional[Dict[str, Any]]:
    """Get record by ID with early returns."""
    try:
        with get_db_session() as session:
            query = f"SELECT * FROM {table_name} WHERE {id_column} = :id"
            result = session.execute(text(query), {'id': record_id})
            row = result.fetchone()
            return dict(row._mapping) if row else None
    except SQLAlchemyError as e:
        logger.error(f"❌ Get record by ID failed: {e}")
        return None

def get_records_paginated(table_name: str, page: int = 1, per_page: int = 20, 
                         order_by: str = 'id', order_direction: str = 'ASC',
                         where_clause: str = None, where_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get paginated records with early returns."""
    try:
        with get_db_session() as session:
            # Build base query
            base_query = f"SELECT * FROM {table_name}"
            count_query = f"SELECT COUNT(*) as total FROM {table_name}"
            
            # Add WHERE clause if provided
            if where_clause:
                base_query += f" WHERE {where_clause}"
                count_query += f" WHERE {where_clause}"
            
            # Get total count
            count_result = session.execute(text(count_query), where_params or {})
            total = count_result.fetchone().total
            
            # Add pagination and ordering
            offset = (page - 1) * per_page
            order_direction = order_direction.upper()
            if order_direction not in ['ASC', 'DESC']:
                order_direction = 'ASC'
            
            query = f"{base_query} ORDER BY {order_by} {order_direction} LIMIT {per_page} OFFSET {offset}"
            result = session.execute(text(query), where_params or {})
            
            records = [dict(row._mapping) for row in result]
            
            return {
                'records': records,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
    except SQLAlchemyError as e:
        logger.error(f"❌ Get paginated records failed: {e}")
        return {'records': [], 'pagination': {'page': 1, 'per_page': 20, 'total': 0, 'pages': 0}}

def search_records(table_name: str, search_columns: List[str], search_term: str,
                  page: int = 1, per_page: int = 20) -> Dict[str, Any]:
    """Search records with early returns."""
    if not search_columns or not search_term:
        return {'records': [], 'pagination': {'page': 1, 'per_page': 20, 'total': 0, 'pages': 0}}
    
    try:
        with get_db_session() as session:
            # Build search conditions
            search_conditions = []
            search_params = {'search_term': f'%{search_term}%'}
            
            for i, column in enumerate(search_columns):
                param_name = f'search_{i}'
                search_conditions.append(f"{column} ILIKE :{param_name}")
                search_params[param_name] = f'%{search_term}%'
            
            where_clause = ' OR '.join(search_conditions)
            
            # Get total count
            count_query = f"SELECT COUNT(*) as total FROM {table_name} WHERE {where_clause}"
            count_result = session.execute(text(count_query), search_params)
            total = count_result.fetchone().total
            
            # Get paginated results
            offset = (page - 1) * per_page
            query = f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT {per_page} OFFSET {offset}"
            result = session.execute(text(query), search_params)
            
            records = [dict(row._mapping) for row in result]
            
            return {
                'records': records,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
    except SQLAlchemyError as e:
        logger.error(f"❌ Search records failed: {e}")
        return {'records': [], 'pagination': {'page': 1, 'per_page': 20, 'total': 0, 'pages': 0}}

def get_aggregated_data(table_name: str, group_by_column: str, 
                       aggregate_functions: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """Get aggregated data with early returns."""
    if not group_by_column:
        return []
    
    try:
        with get_db_session() as session:
            # Build aggregate functions
            if not aggregate_functions:
                aggregate_functions = {'count': 'COUNT(*)'}
            
            select_clause = f"{group_by_column}, " + ", ".join([f"{func}({column}) as {alias}" 
                                                              for alias, func in aggregate_functions.items() 
                                                              for column in [group_by_column]])
            
            query = f"SELECT {select_clause} FROM {table_name} GROUP BY {group_by_column}"
            result = session.execute(text(query))
            
            return [dict(row._mapping) for row in result]
    except SQLAlchemyError as e:
        logger.error(f"❌ Get aggregated data failed: {e}")
        return []

def execute_transaction(operations: List[Callable]) -> bool:
    """Execute multiple operations in a transaction with early returns."""
    if not operations:
        return True
    
    try:
        with get_db_session() as session:
            for operation in operations:
                operation(session)
            return True
    except Exception as e:
        logger.error(f"❌ Transaction failed: {e}")
        return False

def backup_table(table_name: str, backup_suffix: str = None) -> bool:
    """Backup table with early returns."""
    if not table_name:
        return False
    
    try:
        with get_db_session() as session:
            suffix = backup_suffix or str(int(time.time()))
            backup_table_name = f"{table_name}_backup_{suffix}"
            
            # Create backup table
            query = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
            session.execute(text(query))
            
            logger.info(f"✅ Table {table_name} backed up as {backup_table_name}")
            return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Backup failed: {e}")
        return False

def restore_table(table_name: str, backup_table_name: str) -> bool:
    """Restore table from backup with early returns."""
    if not table_name or not backup_table_name:
        return False
    
    try:
        with get_db_session() as session:
            # Clear existing table
            session.execute(text(f"DELETE FROM {table_name}"))
            
            # Restore from backup
            query = f"INSERT INTO {table_name} SELECT * FROM {backup_table_name}"
            session.execute(text(query))
            
            logger.info(f"✅ Table {table_name} restored from {backup_table_name}")
            return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Restore failed: {e}")
        return False

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics with early returns."""
    try:
        with get_db_session() as session:
            # Get table sizes
            size_query = """
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """
            size_result = session.execute(text(size_query))
            table_sizes = [dict(row._mapping) for row in size_result]
            
            # Get connection info
            conn_query = """
                SELECT 
                    count(*) as active_connections,
                    max_connections
                FROM pg_stat_activity, 
                     (SELECT setting::int as max_connections FROM pg_settings WHERE name = 'max_connections') mc
            """
            conn_result = session.execute(text(conn_query))
            conn_info = dict(conn_result.fetchone()._mapping)
            
            return {
                'table_sizes': table_sizes,
                'connection_info': conn_info,
                'timestamp': time.time()
            }
    except SQLAlchemyError as e:
        logger.error(f"❌ Get database stats failed: {e}")
        return {}

def optimize_database() -> bool:
    """Optimize database with early returns."""
    try:
        with get_db_session() as session:
            # Analyze tables
            session.execute(text("ANALYZE"))
            
            # Vacuum tables
            session.execute(text("VACUUM ANALYZE"))
            
            logger.info("✅ Database optimization completed")
            return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Database optimization failed: {e}")
        return False

# Database decorators
def with_database_session(func: Callable) -> Callable:
    """Decorator for database session management."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with get_db_session() as session:
            kwargs['db_session'] = session
            return func(*args, **kwargs)
    return wrapper

def transactional(func: Callable) -> Callable:
    """Decorator for transactional operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with get_db_session() as session:
            try:
                result = func(*args, **kwargs)
                session.commit()
                return result
            except Exception as e:
                session.rollback()
                logger.error(f"❌ Transaction failed in {func.__name__}: {e}")
                raise
    return wrapper









