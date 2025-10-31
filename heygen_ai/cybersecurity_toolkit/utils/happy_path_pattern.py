from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from ..exceptions.custom_exceptions import (
from typing import Any, List, Dict, Optional
import asyncio
"""
Happy Path Pattern Implementation
================================

Demonstrates the happy path pattern with:
- Early returns for error conditions
- Avoiding nested conditionals
- Keeping main success logic at the end
- Guard clauses for validation
- Clean, readable code structure
"""


# Import custom exceptions
    ValidationError,
    MissingRequiredFieldError,
    InvalidFieldTypeError,
    FieldValueOutOfRangeError,
    NetworkError,
    ConnectionTimeoutError
)

# Get logger
logger = logging.getLogger(__name__)


class HappyPathProcessor:
    """
    Demonstrates happy path pattern with early returns and clean structure.
    """
    
    def __init__(self, max_retries: int = 3, timeout: float = 5.0):
        """
        Initialize processor with validation using early returns.
        
        Args:
            max_retries: Maximum retry attempts
            timeout: Operation timeout
        """
        # Early return pattern: Validate parameters first
        if not isinstance(max_retries, int) or max_retries <= 0:
            raise FieldValueOutOfRangeError(
                "max_retries",
                max_retries,
                min_value=1,
                context={"operation": "processor_initialization"}
            )
        
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise FieldValueOutOfRangeError(
                "timeout",
                timeout,
                min_value=0.1,
                context={"operation": "processor_initialization"}
            )
        
        # Happy path: Set valid parameters
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_count = 0
    
    def process_data_happy_path(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data using happy path pattern - main logic at the end.
        
        Args:
            data: Data to process
            
        Returns:
            Processing result
            
        Raises:
            ValidationError: When data validation fails
        """
        # Guard clause 1: Check if data is provided
        if not data:
            raise MissingRequiredFieldError(
                "data",
                context={"operation": "data_processing"}
            )
        
        # Guard clause 2: Check if data is a dictionary
        if not isinstance(data, dict):
            raise InvalidFieldTypeError(
                "data",
                data,
                "dict",
                context={"operation": "data_processing"}
            )
        
        # Guard clause 3: Check if required fields exist
        required_fields = ["id", "content", "type"]
        for field in required_fields:
            if field not in data:
                raise MissingRequiredFieldError(
                    field,
                    context={"operation": "data_processing"}
                )
        
        # Guard clause 4: Validate field types
        if not isinstance(data["id"], str):
            raise InvalidFieldTypeError(
                "id",
                data["id"],
                "str",
                context={"operation": "data_processing"}
            )
        
        if not isinstance(data["content"], str):
            raise InvalidFieldTypeError(
                "content",
                data["content"],
                "str",
                context={"operation": "data_processing"}
            )
        
        if not isinstance(data["type"], str):
            raise InvalidFieldTypeError(
                "type",
                data["type"],
                "str",
                context={"operation": "data_processing"}
            )
        
        # Guard clause 5: Validate content length
        if len(data["content"]) > 1000:
            raise FieldValueOutOfRangeError(
                "content",
                len(data["content"]),
                max_value=1000,
                context={"operation": "data_processing"}
            )
        
        # Guard clause 6: Validate data type
        valid_types = ["text", "json", "xml", "binary"]
        if data["type"] not in valid_types:
            raise ValidationError(
                "type",
                data["type"],
                "invalid_type",
                f"Type must be one of: {valid_types}",
                context={"operation": "data_processing"}
            )
        
        # Happy path: All validation passed, process the data
        processed_data = {
            "id": data["id"],
            "content": data["content"].upper() if data["type"] == "text" else data["content"],
            "type": data["type"],
            "processed_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        return processed_data
    
    def validate_network_config_happy_path(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate network configuration using happy path pattern.
        
        Args:
            config: Network configuration
            
        Returns:
            Validation result
            
        Raises:
            ValidationError: When validation fails
        """
        # Guard clause 1: Check if config is provided
        if not config:
            raise MissingRequiredFieldError(
                "config",
                context={"operation": "network_validation"}
            )
        
        # Guard clause 2: Check if config is a dictionary
        if not isinstance(config, dict):
            raise InvalidFieldTypeError(
                "config",
                config,
                "dict",
                context={"operation": "network_validation"}
            )
        
        # Guard clause 3: Check required network fields
        network_fields = ["host", "port", "protocol"]
        for field in network_fields:
            if field not in config:
                raise MissingRequiredFieldError(
                    field,
                    context={"operation": "network_validation"}
                )
        
        # Guard clause 4: Validate host
        host = config["host"]
        if not isinstance(host, str) or not host.strip():
            raise InvalidFieldTypeError(
                "host",
                host,
                "non_empty_string",
                context={"operation": "network_validation"}
            )
        
        # Guard clause 5: Validate port
        port = config["port"]
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise FieldValueOutOfRangeError(
                "port",
                port,
                min_value=1,
                max_value=65535,
                context={"operation": "network_validation"}
            )
        
        # Guard clause 6: Validate protocol
        protocol = config["protocol"]
        valid_protocols = ["http", "https", "ftp", "ssh", "tcp", "udp"]
        if not isinstance(protocol, str) or protocol.lower() not in valid_protocols:
            raise ValidationError(
                "protocol",
                protocol,
                "invalid_protocol",
                f"Protocol must be one of: {valid_protocols}",
                context={"operation": "network_validation"}
            )
        
        # Guard clause 7: Validate optional timeout
        if "timeout" in config:
            timeout = config["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise FieldValueOutOfRangeError(
                    "timeout",
                    timeout,
                    min_value=0.1,
                    context={"operation": "network_validation"}
                )
        
        # Happy path: All validation passed, return validated config
        validated_config = {
            "host": host.strip(),
            "port": port,
            "protocol": protocol.lower(),
            "timeout": config.get("timeout", 30.0),
            "validated_at": datetime.utcnow().isoformat(),
            "status": "valid"
        }
        
        return validated_config
    
    async def process_user_request_happy_path(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user request using happy path pattern with multiple validation layers.
        
        Args:
            request: User request
            
        Returns:
            Processing result
            
        Raises:
            ValidationError: When validation fails
        """
        # Guard clause 1: Basic request validation
        if not request:
            raise MissingRequiredFieldError(
                "request",
                context={"operation": "user_request_processing"}
            )
        
        if not isinstance(request, dict):
            raise InvalidFieldTypeError(
                "request",
                request,
                "dict",
                context={"operation": "user_request_processing"}
            )
        
        # Guard clause 2: Authentication validation
        if "auth_token" not in request:
            raise MissingRequiredFieldError(
                "auth_token",
                context={"operation": "user_request_processing"}
            )
        
        auth_token = request["auth_token"]
        if not isinstance(auth_token, str) or len(auth_token) < 10:
            raise ValidationError(
                "auth_token",
                auth_token,
                "invalid_token",
                "Auth token must be a string with at least 10 characters",
                context={"operation": "user_request_processing"}
            )
        
        # Guard clause 3: Request type validation
        if "request_type" not in request:
            raise MissingRequiredFieldError(
                "request_type",
                context={"operation": "user_request_processing"}
            )
        
        request_type = request["request_type"]
        valid_types = ["read", "write", "delete", "update"]
        if not isinstance(request_type, str) or request_type not in valid_types:
            raise ValidationError(
                "request_type",
                request_type,
                "invalid_request_type",
                f"Request type must be one of: {valid_types}",
                context={"operation": "user_request_processing"}
            )
        
        # Guard clause 4: Resource validation
        if "resource_id" not in request:
            raise MissingRequiredFieldError(
                "resource_id",
                context={"operation": "user_request_processing"}
            )
        
        resource_id = request["resource_id"]
        if not isinstance(resource_id, str) or not resource_id.strip():
            raise InvalidFieldTypeError(
                "resource_id",
                resource_id,
                "non_empty_string",
                context={"operation": "user_request_processing"}
            )
        
        # Guard clause 5: Permission validation (simulated)
        if request_type in ["write", "delete"] and not self._has_write_permission(auth_token):
            raise ValidationError(
                "permissions",
                request_type,
                "insufficient_permissions",
                f"Insufficient permissions for {request_type} operation",
                context={"operation": "user_request_processing"}
            )
        
        # Guard clause 6: Rate limiting validation (simulated)
        if not self._check_rate_limit(auth_token):
            raise ValidationError(
                "rate_limit",
                auth_token,
                "rate_limit_exceeded",
                "Rate limit exceeded, please try again later",
                context={"operation": "user_request_processing"}
            )
        
        # Happy path: All validation passed, process the request
        result = {
            "request_id": f"req_{datetime.utcnow().timestamp()}",
            "user_id": self._extract_user_id(auth_token),
            "resource_id": resource_id.strip(),
            "request_type": request_type,
            "status": "processed",
            "processed_at": datetime.utcnow().isoformat(),
            "data": request.get("data", {})
        }
        
        return result
    
    def _has_write_permission(self, auth_token: str) -> bool:
        """Simulate permission check."""
        # Simulate permission validation
        return len(auth_token) > 15  # Simple simulation
    
    def _check_rate_limit(self, auth_token: str) -> bool:
        """Simulate rate limiting check."""
        # Simulate rate limiting
        return True  # Always allow for demo
    
    def _extract_user_id(self, auth_token: str) -> str:
        """Extract user ID from auth token."""
        # Simulate user ID extraction
        return f"user_{hash(auth_token) % 10000}"


def demonstrate_happy_path_pattern():
    """
    Demonstrate the happy path pattern with various examples.
    """
    processor = HappyPathProcessor()
    
    print("=" * 80)
    print("HAPPY PATH PATTERN DEMONSTRATION")
    print("=" * 80)
    
    print("✓ Happy Path Pattern Structure:")
    print("""
def function_with_happy_path(param) -> Any:
    # Guard clause 1: Check if param is provided
    if not param:
        raise ValidationError("Parameter required")
    
    # Guard clause 2: Check param type
    if not isinstance(param, expected_type):
        raise ValidationError("Invalid parameter type")
    
    # Guard clause 3: Check param value
    if param < min_value or param > max_value:
        raise ValidationError("Parameter out of range")
    
    # Happy path: All validation passed, main logic here
    result = process_parameter(param)
    return result
    """)
    
    print("\n✓ Benefits of Happy Path Pattern:")
    print("  - Early returns prevent deep nesting")
    print("  - Main success logic is clearly visible")
    print("  - Easier to read and understand")
    print("  - Reduces cognitive complexity")
    print("  - Makes debugging easier")
    print("  - Improves code maintainability")
    
    # Test data processing
    print("\n✓ Data Processing Example:")
    try:
        valid_data = {
            "id": "user123",
            "content": "Hello, World!",
            "type": "text"
        }
        result = processor.process_data_happy_path(valid_data)
        print(f"  ✅ Success: {result['status']}")
        
        # Test invalid data
        invalid_data = {
            "id": "user123",
            "content": "A" * 1001,  # Too long
            "type": "text"
        }
        result = processor.process_data_happy_path(invalid_data)
    except ValidationError as e:
        print(f"  ❌ Validation error: {e.message}")
    
    # Test network config validation
    print("\n✓ Network Config Validation Example:")
    try:
        valid_config = {
            "host": "example.com",
            "port": 443,
            "protocol": "https",
            "timeout": 30.0
        }
        result = processor.validate_network_config_happy_path(valid_config)
        print(f"  ✅ Success: {result['status']}")
        
        # Test invalid config
        invalid_config = {
            "host": "example.com",
            "port": 99999,  # Invalid port
            "protocol": "https"
        }
        result = processor.validate_network_config_happy_path(invalid_config)
    except ValidationError as e:
        print(f"  ❌ Validation error: {e.message}")
    
    # Test user request processing
    print("\n✓ User Request Processing Example:")
    try:
        valid_request = {
            "auth_token": "valid_token_with_sufficient_length",
            "request_type": "read",
            "resource_id": "resource123",
            "data": {"key": "value"}
        }
        result = processor.process_user_request_happy_path(valid_request)
        print(f"  ✅ Success: {result['status']}")
        
        # Test invalid request
        invalid_request = {
            "auth_token": "short",  # Too short
            "request_type": "read",
            "resource_id": "resource123"
        }
        result = processor.process_user_request_happy_path(invalid_request)
    except ValidationError as e:
        print(f"  ❌ Validation error: {e.message}")


def demonstrate_nested_vs_happy_path():
    """
    Compare nested conditionals vs happy path pattern.
    """
    print("\n" + "=" * 80)
    print("NESTED CONDITIONALS VS HAPPY PATH PATTERN")
    print("=" * 80)
    
    print("✓ Nested Conditionals (Avoid This):")
    print("""
def process_data_nested(data) -> Any:
    if data is not None:
        if isinstance(data, dict):
            if "id" in data:
                if "content" in data:
                    if len(data["content"]) <= 1000:
                        if data["type"] in ["text", "json"]:
                            # Main logic here (deeply nested)
                            return {"status": "success", "data": data}
                        else:
                            return {"status": "error", "message": "Invalid type"}
                    else:
                        return {"status": "error", "message": "Content too long"}
                else:
                    return {"status": "error", "message": "Missing content"}
            else:
                return {"status": "error", "message": "Missing id"}
        else:
            return {"status": "error", "message": "Invalid data type"}
    else:
        return {"status": "error", "message": "No data provided"}
    """)
    
    print("\n✓ Happy Path Pattern (Use This):")
    print("""
def process_data_happy_path(data) -> Any:
    # Guard clause 1: Check if data is provided
    if not data:
        raise ValidationError("No data provided")
    
    # Guard clause 2: Check data type
    if not isinstance(data, dict):
        raise ValidationError("Invalid data type")
    
    # Guard clause 3: Check required fields
    if "id" not in data:
        raise ValidationError("Missing id")
    
    if "content" not in data:
        raise ValidationError("Missing content")
    
    # Guard clause 4: Validate content length
    if len(data["content"]) > 1000:
        raise ValidationError("Content too long")
    
    # Guard clause 5: Validate type
    if data["type"] not in ["text", "json"]:
        raise ValidationError("Invalid type")
    
    # Happy path: All validation passed, main logic here
    return {"status": "success", "data": data}
    """)
    
    print("\n✓ Comparison Benefits:")
    print("  Nested Conditionals:")
    print("    - Hard to read and understand")
    print("    - High cognitive complexity")
    print("    - Difficult to maintain")
    print("    - Error handling mixed with logic")
    print("    - Deep nesting levels")
    
    print("\n  Happy Path Pattern:")
    print("    - Clear and readable")
    print("    - Low cognitive complexity")
    print("    - Easy to maintain")
    print("    - Clear separation of concerns")
    print("    - Main logic is obvious")


def demonstrate_guard_clauses():
    """
    Demonstrate guard clauses as part of happy path pattern.
    """
    print("\n" + "=" * 80)
    print("GUARD CLAUSES DEMONSTRATION")
    print("=" * 80)
    
    print("✓ Guard Clause Pattern:")
    print("""
def function_with_guard_clauses(param1, param2, param3) -> Any:
    # Guard clause 1: Validate param1
    if not param1:
        raise ValidationError("param1 is required")
    
    # Guard clause 2: Validate param2
    if not isinstance(param2, str):
        raise ValidationError("param2 must be a string")
    
    # Guard clause 3: Validate param3
    if param3 < 0 or param3 > 100:
        raise ValidationError("param3 must be between 0 and 100")
    
    # Guard clause 4: Check business rules
    if param1 == "admin" and param3 < 50:
        raise ValidationError("Admin requires higher permission level")
    
    # Happy path: All validation passed
    result = process_parameters(param1, param2, param3)
    return result
    """)
    
    print("\n✓ Guard Clause Benefits:")
    print("  - Early validation prevents invalid processing")
    print("  - Clear error messages for each validation failure")
    print("  - Reduces complexity of main logic")
    print("  - Makes function behavior predictable")
    print("  - Improves testability")


def main():
    """Main demonstration function."""
    print("HAPPY PATH PATTERN IMPLEMENTATION DEMONSTRATION")
    print("=" * 100)
    
    try:
        demonstrate_happy_path_pattern()
        demonstrate_nested_vs_happy_path()
        demonstrate_guard_clauses()
        
        print("\n" + "=" * 100)
        print("✅ HAPPY PATH PATTERN DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 100)
        
        print("\n🎯 Key Patterns Demonstrated:")
        print("  ✅ Early returns for error conditions")
        print("  ✅ Avoiding nested conditionals")
        print("  ✅ Keeping main success logic at the end")
        print("  ✅ Guard clauses for validation")
        print("  ✅ Clean, readable code structure")
        print("  ✅ Reduced cognitive complexity")
        
        print("\n📋 Best Practices:")
        print("  1. Use guard clauses at the beginning of functions")
        print("  2. Return early on error conditions")
        print("  3. Keep the main success logic at the end")
        print("  4. Avoid deep nesting of conditionals")
        print("  5. Use descriptive error messages")
        print("  6. Separate validation from business logic")
        print("  7. Make the happy path obvious and clear")
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        return False
    
    return True


match __name__:
    case "__main__":
    main() 