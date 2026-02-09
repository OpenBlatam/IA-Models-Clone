from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import aiohttp
import asyncpg
import aiosqlite
import aioredis
from pathlib import Path
from async_io_manager import (
        import shutil
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Async I/O Demo
Product Descriptions Feature - Comprehensive Asynchronous I/O Operations Demonstration
"""


# Import async I/O manager
    AsyncIOManager,
    ConnectionConfig,
    ConnectionType,
    OperationType,
    IOMetrics,
    async_io_timed,
    async_io_retry,
    initialize_database_connections,
    initialize_api_sessions,
    cleanup_io_connections,
    get_user_by_id,
    fetch_external_data,
    create_user_with_profile,
    fetch_multiple_external_data
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncIODemo:
    """Comprehensive async I/O operations demonstration"""
    
    def __init__(self) -> Any:
        self.results: List[Dict[str, Any]] = []
        self.io_manager = AsyncIOManager()
        
        # Test data
        self.test_users = [
            {"name": "John Doe", "email": "john@example.com"},
            {"name": "Jane Smith", "email": "jane@example.com"},
            {"name": "Bob Johnson", "email": "bob@example.com"}
        ]
        
        self.test_profiles = [
            {"bio": "Software engineer", "avatar": "avatar1.jpg"},
            {"bio": "Data scientist", "avatar": "avatar2.jpg"},
            {"bio": "Product manager", "avatar": "avatar3.jpg"}
        ]
    
    def log_result(self, test_name: str, success: bool, data: Dict[str, Any], duration: float):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "data": data,
            "duration": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(result)
        logger.info(f"Test: {test_name} - {'PASS' if success else 'FAIL'} ({duration:.3f}s)")
    
    async def setup(self) -> Any:
        """Setup demo environment"""
        # Create test directories
        Path("test_data").mkdir(exist_ok=True)
        Path("test_cache").mkdir(exist_ok=True)
        
        # Initialize database connections
        await initialize_database_connections()
        
        # Initialize API sessions
        await initialize_api_sessions()
        
        # Create test tables
        await self._create_test_tables()
        
        logger.info("Async I/O demo environment setup completed")
    
    async def cleanup(self) -> Any:
        """Cleanup demo environment"""
        await cleanup_io_connections()
        
        # Cleanup test files
        if Path("test_data").exists():
            shutil.rmtree("test_data")
        if Path("test_cache").exists():
            shutil.rmtree("test_cache")
        
        logger.info("Async I/O demo environment cleanup completed")
    
    async def _create_test_tables(self) -> Any:
        """Create test database tables"""
        try:
            # Create users table
            create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            # Create profiles table
            create_profiles_table = """
            CREATE TABLE IF NOT EXISTS profiles (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                bio TEXT,
                avatar VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            await self.io_manager.execute_query("postgres", create_users_table, operation_type=OperationType.EXECUTE)
            await self.io_manager.execute_query("postgres", create_profiles_table, operation_type=OperationType.EXECUTE)
            
            logger.info("Test tables created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create test tables (this is expected if PostgreSQL is not running): {e}")
    
    async def test_database_connections(self) -> Dict[str, Any]:
        """Test database connection initialization"""
        start_time = time.time()
        
        try:
            # Test PostgreSQL connection
            postgres_config = ConnectionConfig(
                connection_type=ConnectionType.POSTGRESQL,
                host="localhost",
                port=5432,
                database="product_descriptions",
                username="postgres",
                password="password",
                pool_size=5,
                timeout=10.0
            )
            
            await self.io_manager.initialize_database("test_postgres", postgres_config)
            
            # Test SQLite connection
            sqlite_config = ConnectionConfig(
                connection_type=ConnectionType.SQLITE,
                host="",
                port=0,
                database="test_data/test.db",
                pool_size=1,
                timeout=10.0
            )
            
            await self.io_manager.initialize_database("test_sqlite", sqlite_config)
            
            # Test Redis connection
            redis_config = ConnectionConfig(
                connection_type=ConnectionType.REDIS,
                host="localhost",
                port=6379,
                database=0,
                pool_size=5,
                timeout=10.0
            )
            
            await self.io_manager.initialize_database("test_redis", redis_config)
            
            duration = time.time() - start_time
            
            success = True
            data = {
                "postgres_initialized": True,
                "sqlite_initialized": True,
                "redis_initialized": True,
                "database_connections_working": success
            }
            
            self.log_result("Database Connections", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Database Connections", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async async def test_api_sessions(self) -> Dict[str, Any]:
        """Test API session initialization"""
        start_time = time.time()
        
        try:
            # Test external API session
            await self.io_manager.initialize_api(
                "test_external_api",
                "https://httpbin.org",
                headers={"User-Agent": "AsyncIODemo/1.0"},
                timeout=10.0,
                ssl_verify=True
            )
            
            # Test internal API session
            await self.io_manager.initialize_api(
                "test_internal_api",
                "http://localhost:8000",
                headers={"Content-Type": "application/json"},
                timeout=5.0,
                ssl_verify=False
            )
            
            duration = time.time() - start_time
            
            success = True
            data = {
                "external_api_initialized": True,
                "internal_api_initialized": True,
                "api_sessions_working": success
            }
            
            self.log_result("API Sessions", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("API Sessions", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_postgresql_operations(self) -> Dict[str, Any]:
        """Test PostgreSQL async operations"""
        start_time = time.time()
        
        try:
            # Test INSERT operation
            insert_query = """
            INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id
            """
            insert_params = {"name": "Test User", "email": "test@example.com"}
            
            insert_result = await self.io_manager.execute_query(
                "postgres", insert_query, insert_params, OperationType.WRITE
            )
            
            user_id = insert_result[0]["id"] if insert_result else None
            
            # Test SELECT operation
            select_query = "SELECT * FROM users WHERE id = $1"
            select_params = {"id": user_id}
            
            select_result = await self.io_manager.execute_query(
                "postgres", select_query, select_params, OperationType.QUERY
            )
            
            # Test UPDATE operation
            update_query = "UPDATE users SET name = $1 WHERE id = $2"
            update_params = {"name": "Updated User", "id": user_id}
            
            update_result = await self.io_manager.execute_query(
                "postgres", update_query, update_params, OperationType.UPDATE
            )
            
            duration = time.time() - start_time
            
            success = (
                user_id is not None and
                len(select_result) == 1 and
                select_result[0]["email"] == "test@example.com"
            )
            
            data = {
                "insert_operation": "successful",
                "select_operation": "successful",
                "update_operation": "successful",
                "user_id": user_id,
                "postgresql_operations_working": success
            }
            
            self.log_result("PostgreSQL Operations", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("PostgreSQL Operations", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_sqlite_operations(self) -> Dict[str, Any]:
        """Test SQLite async operations"""
        start_time = time.time()
        
        try:
            # Create test table
            create_table = """
            CREATE TABLE IF NOT EXISTS test_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value INTEGER
            )
            """
            
            await self.io_manager.execute_query(
                "sqlite", create_table, operation_type=OperationType.EXECUTE
            )
            
            # Test INSERT operation
            insert_query = "INSERT INTO test_items (name, value) VALUES (?, ?)"
            insert_params = {"name": "Test Item", "value": 42}
            
            insert_result = await self.io_manager.execute_query(
                "sqlite", insert_query, insert_params, OperationType.WRITE
            )
            
            # Test SELECT operation
            select_query = "SELECT * FROM test_items WHERE name = ?"
            select_params = {"name": "Test Item"}
            
            select_result = await self.io_manager.execute_query(
                "sqlite", select_query, select_params, OperationType.QUERY
            )
            
            duration = time.time() - start_time
            
            success = len(select_result) == 1 and select_result[0]["value"] == 42
            
            data = {
                "insert_operation": "successful",
                "select_operation": "successful",
                "result_count": len(select_result),
                "sqlite_operations_working": success
            }
            
            self.log_result("SQLite Operations", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("SQLite Operations", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_redis_operations(self) -> Dict[str, Any]:
        """Test Redis async operations"""
        start_time = time.time()
        
        try:
            # Test SET operation
            set_key = "test_key"
            set_value = "test_value"
            
            set_result = await self.io_manager.execute_query(
                "redis", set_key, {"value": set_value}, OperationType.WRITE
            )
            
            # Test GET operation
            get_result = await self.io_manager.execute_query(
                "redis", set_key, operation_type=OperationType.READ
            )
            
            # Test DELETE operation
            delete_result = await self.io_manager.execute_query(
                "redis", set_key, operation_type=OperationType.DELETE
            )
            
            duration = time.time() - start_time
            
            success = (
                set_result and
                get_result and
                get_result[0]["value"] == set_value
            )
            
            data = {
                "set_operation": "successful",
                "get_operation": "successful",
                "delete_operation": "successful",
                "retrieved_value": get_result[0]["value"] if get_result else None,
                "redis_operations_working": success
            }
            
            self.log_result("Redis Operations", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Redis Operations", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_database_transactions(self) -> Dict[str, Any]:
        """Test database transactions"""
        start_time = time.time()
        
        try:
            # Test transaction with multiple operations
            transaction_queries = [
                {
                    "query": "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
                    "params": {"name": "Transaction User", "email": "transaction@example.com"},
                    "operation": "query"
                },
                {
                    "query": "INSERT INTO profiles (user_id, bio, avatar) VALUES ($1, $2, $3)",
                    "params": {"user_id": "$1", "bio": "Transaction test", "avatar": "transaction.jpg"},
                    "operation": "execute"
                }
            ]
            
            transaction_result = await self.io_manager.execute_transaction("postgres", transaction_queries)
            
            duration = time.time() - start_time
            
            success = (
                transaction_result and
                len(transaction_result) == 2 and
                transaction_result[0] and
                len(transaction_result[0]) == 1
            )
            
            data = {
                "transaction_executed": "successful",
                "operations_count": len(transaction_result),
                "user_id": transaction_result[0][0]["id"] if transaction_result and transaction_result[0] else None,
                "database_transactions_working": success
            }
            
            self.log_result("Database Transactions", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Database Transactions", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async async def test_api_requests(self) -> Dict[str, Any]:
        """Test async API requests"""
        start_time = time.time()
        
        try:
            # Test GET request
            get_response = await self.io_manager.make_api_request(
                "test_external_api",
                "GET",
                "/get",
                params={"test": "value"}
            )
            
            # Test POST request
            post_data = {"name": "Test User", "email": "test@example.com"}
            post_response = await self.io_manager.make_api_request(
                "test_external_api",
                "POST",
                "/post",
                data=post_data
            )
            
            duration = time.time() - start_time
            
            success = (
                get_response["status_code"] == 200 and
                post_response["status_code"] == 200 and
                "args" in get_response["data"] and
                "json" in post_response["data"]
            )
            
            data = {
                "get_request": "successful",
                "post_request": "successful",
                "get_status": get_response["status_code"],
                "post_status": post_response["status_code"],
                "api_requests_working": success
            }
            
            self.log_result("API Requests", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("API Requests", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async async def test_batch_api_requests(self) -> Dict[str, Any]:
        """Test batch API requests"""
        start_time = time.time()
        
        try:
            # Create multiple requests
            requests = [
                {
                    "method": "GET",
                    "url": "/get",
                    "params": {"id": str(i)}
                }
                for i in range(5)
            ]
            
            # Execute batch requests
            batch_results = await self.io_manager.make_batch_api_requests(
                "test_external_api", requests
            )
            
            duration = time.time() - start_time
            
            success = (
                len(batch_results) == 5 and
                all(r["status_code"] == 200 for r in batch_results)
            )
            
            data = {
                "batch_requests_executed": "successful",
                "requests_count": len(batch_results),
                "successful_requests": sum(1 for r in batch_results if r["status_code"] == 200),
                "batch_api_requests_working": success
            }
            
            self.log_result("Batch API Requests", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Batch API Requests", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent database and API operations"""
        start_time = time.time()
        
        try:
            # Create concurrent tasks
            tasks = []
            
            # Database operations
            for i in range(3):
                task = self.io_manager.execute_query(
                    "postgres",
                    "SELECT COUNT(*) as count FROM users",
                    operation_type=OperationType.QUERY
                )
                tasks.append(task)
            
            # API operations
            for i in range(3):
                task = self.io_manager.make_api_request(
                    "test_external_api",
                    "GET",
                    "/delay/1"  # 1 second delay
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Check results
            db_results = results[:3]
            api_results = results[3:]
            
            db_success = all(not isinstance(r, Exception) for r in db_results)
            api_success = all(not isinstance(r, Exception) for r in api_results)
            
            success = db_success and api_success
            
            data = {
                "concurrent_operations": "successful",
                "database_operations": len(db_results),
                "api_operations": len(api_results),
                "db_success": db_success,
                "api_success": api_success,
                "concurrent_operations_working": success
            }
            
            self.log_result("Concurrent Operations", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Concurrent Operations", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_decorators(self) -> Dict[str, Any]:
        """Test async I/O decorators"""
        start_time = time.time()
        
        try:
            # Test timed decorator
            @async_io_timed("test_operation")
            async def test_db_operation():
                
    """test_db_operation function."""
await asyncio.sleep(0.1)
                return await self.io_manager.execute_query(
                    "postgres",
                    "SELECT 1 as test",
                    operation_type=OperationType.QUERY
                )
            
            # Test retry decorator
            @async_io_retry(max_attempts=2, delay=0.1)
            async def test_api_operation():
                
    """test_api_operation function."""
await asyncio.sleep(0.05)
                return await self.io_manager.make_api_request(
                    "test_external_api",
                    "GET",
                    "/get"
                )
            
            # Execute decorated functions
            db_result = await test_db_operation()
            api_result = await test_api_operation()
            
            duration = time.time() - start_time
            
            success = (
                db_result and
                len(db_result) == 1 and
                db_result[0]["test"] == 1 and
                api_result and
                api_result["status_code"] == 200
            )
            
            data = {
                "timed_decorator": "working",
                "retry_decorator": "working",
                "db_result": db_result[0] if db_result else None,
                "api_result_status": api_result["status_code"] if api_result else None,
                "decorators_working": success
            }
            
            self.log_result("Async I/O Decorators", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Async I/O Decorators", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_metrics_collection(self) -> Dict[str, Any]:
        """Test metrics collection"""
        start_time = time.time()
        
        try:
            # Perform some operations to generate metrics
            await self.io_manager.execute_query(
                "postgres",
                "SELECT COUNT(*) as count FROM users",
                operation_type=OperationType.QUERY
            )
            
            await self.io_manager.make_api_request(
                "test_external_api",
                "GET",
                "/get"
            )
            
            # Collect metrics
            postgres_metrics = await self.io_manager.get_database_metrics("postgres")
            api_metrics = await self.io_manager.get_api_metrics()
            
            duration = time.time() - start_time
            
            success = (
                postgres_metrics and
                "total_operations" in postgres_metrics and
                api_metrics and
                "total_requests" in api_metrics
            )
            
            data = {
                "postgres_metrics": postgres_metrics,
                "api_metrics": api_metrics,
                "metrics_collection_working": success
            }
            
            self.log_result("Metrics Collection", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Metrics Collection", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling in async I/O operations"""
        start_time = time.time()
        
        try:
            # Test invalid database query
            try:
                await self.io_manager.execute_query(
                    "postgres",
                    "SELECT * FROM nonexistent_table",
                    operation_type=OperationType.QUERY
                )
                db_error_handled = False
            except Exception as e:
                db_error_handled = True
                db_error = str(e)
            
            # Test invalid API request
            try:
                await self.io_manager.make_api_request(
                    "test_external_api",
                    "GET",
                    "/nonexistent-endpoint"
                )
                api_error_handled = False
            except Exception as e:
                api_error_handled = True
                api_error = str(e)
            
            duration = time.time() - start_time
            
            success = db_error_handled and api_error_handled
            
            data = {
                "database_error_handled": db_error_handled,
                "api_error_handled": api_error_handled,
                "db_error": db_error if db_error_handled else None,
                "api_error": api_error if api_error_handled else None,
                "error_handling_working": success
            }
            
            self.log_result("Error Handling", success, data, duration)
            return data
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Error Handling", False, {"error": str(e)}, duration)
            return {"error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all async I/O tests"""
        logger.info("Starting Async I/O Demo Tests...")
        
        # Setup
        await self.setup()
        
        tests = [
            self.test_database_connections,
            self.test_api_sessions,
            self.test_postgresql_operations,
            self.test_sqlite_operations,
            self.test_redis_operations,
            self.test_database_transactions,
            self.test_api_requests,
            self.test_batch_api_requests,
            self.test_concurrent_operations,
            self.test_decorators,
            self.test_metrics_collection,
            self.test_error_handling
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(0.5)  # Small delay between tests
            except Exception as e:
                logger.error(f"Test failed: {test.__name__} - {e}")
        
        # Cleanup
        await self.cleanup()
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - passed_tests
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "results": self.results
        }
        
        logger.info(f"Async I/O Demo completed: {passed_tests}/{total_tests} tests passed")
        return summary
    
    def save_results(self, filename: str = "async_io_demo_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

async def main():
    """Main demo execution"""
    print("=" * 70)
    print("ASYNC I/O DEMO - PRODUCT DESCRIPTIONS FEATURE")
    print("=" * 70)
    
    # Create demo instance
    demo = AsyncIODemo()
    
    # Run all tests
    summary = await demo.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for result in summary['results']:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status}: {result['test']} ({result['duration']:.3f}s)")
        
        if not result['success'] and 'error' in result['data']:
            print(f"  Error: {result['data']['error']}")
    
    # Save results
    demo.save_results()
    
    print("\n" + "=" * 70)
    print("Demo completed! Check async_io_demo_results.json for detailed results.")
    print("=" * 70)

match __name__:
    case "__main__":
    asyncio.run(main()) 