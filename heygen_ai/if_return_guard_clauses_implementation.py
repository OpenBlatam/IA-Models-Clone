from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
    import time
from typing import Any, List, Dict, Optional
import asyncio
"""
If-Return Pattern and Guard Clauses Implementation
=================================================

This module demonstrates:
- If-return pattern to avoid unnecessary else statements
- Guard clauses to handle preconditions and invalid states early
- Early returns for better code readability
- Reduced nesting and improved maintainability
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class User:
    """User data model"""
    id: int
    username: str
    email: str
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self) -> Any:
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str
    layers: List[int]
    learning_rate: float
    batch_size: int
    epochs: int


@dataclass
class TrainingResult:
    """Training result"""
    model_id: str
    final_loss: float
    final_accuracy: float
    training_time: float
    epochs_completed: int


# ============================================================================
# If-Return Pattern Examples
# ============================================================================

def validate_user_data_bad(user_data: Dict[str, Any]) -> str:
    """
    BAD: Using else statements (avoid this)
    """
    if user_data.get("username"):
        if len(user_data["username"]) >= 3:
            if user_data.get("email"):
                if "@" in user_data["email"]:
                    return "Valid user data"
                else:
                    return "Invalid email format"
            else:
                return "Email is required"
        else:
            return "Username too short"
    else:
        return "Username is required"


def validate_user_data_good(user_data: Dict[str, Any]) -> str:
    """
    GOOD: Using if-return pattern (preferred)
    """
    # Guard clauses - handle invalid states early
    if not user_data.get("username"):
        return "Username is required"
    
    if len(user_data["username"]) < 3:
        return "Username too short"
    
    if not user_data.get("email"):
        return "Email is required"
    
    if "@" not in user_data["email"]:
        return "Invalid email format"
    
    # Happy path - all validations passed
    return "Valid user data"


def process_user_permissions_bad(user: User, action: str) -> bool:
    """
    BAD: Nested if-else statements
    """
    if user.is_active:
        if action == "read":
            return True
        elif action == "write":
            if user.username.startswith("admin"):
                return True
            else:
                return False
        elif action == "delete":
            if user.username.startswith("admin"):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def process_user_permissions_good(user: User, action: str) -> bool:
    """
    GOOD: Using guard clauses and if-return pattern
    """
    # Guard clause - handle inactive users early
    if not user.is_active:
        return False
    
    # Guard clause - handle invalid actions early
    if action not in ["read", "write", "delete"]:
        return False
    
    # Handle read permissions
    if action == "read":
        return True
    
    # Handle write/delete permissions (require admin)
    if action in ["write", "delete"]:
        return user.username.startswith("admin")
    
    # This should never be reached due to guard clause above
    return False


# ============================================================================
# Guard Clauses for Preconditions
# ============================================================================

def train_model_with_guards(config: ModelConfig, data_path: str) -> Optional[TrainingResult]:
    """
    Using guard clauses to handle preconditions early
    """
    # Guard clause: Check if config is provided
    if not config:
        logger.error("Model configuration is required")
        return None
    
    # Guard clause: Validate model type
    if config.model_type not in ["neural_network", "transformer", "cnn"]:
        logger.error(f"Invalid model type: {config.model_type}")
        return None
    
    # Guard clause: Check if data path exists
    if not Path(data_path).exists():
        logger.error(f"Data path does not exist: {data_path}")
        return None
    
    # Guard clause: Validate hyperparameters
    if config.learning_rate <= 0 or config.learning_rate > 1:
        logger.error(f"Invalid learning rate: {config.learning_rate}")
        return None
    
    if config.batch_size <= 0:
        logger.error(f"Invalid batch size: {config.batch_size}")
        return None
    
    if config.epochs <= 0:
        logger.error(f"Invalid number of epochs: {config.epochs}")
        return None
    
    # Guard clause: Validate layer configuration
    if not config.layers or len(config.layers) < 2:
        logger.error("Model must have at least 2 layers")
        return None
    
    # Happy path - all preconditions satisfied
    logger.info("Starting model training...")
    return TrainingResult(
        model_id=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        final_loss=0.15,
        final_accuracy=0.92,
        training_time=120.5,
        epochs_completed=config.epochs
    )


def process_payment_with_guards(
    user: User,
    amount: float,
    payment_method: str,
    currency: str = "USD"
) -> Dict[str, Any]:
    """
    Using guard clauses for payment processing
    """
    # Guard clause: Check if user exists
    if not user:
        return {"success": False, "error": "User not found"}
    
    # Guard clause: Check if user is active
    if not user.is_active:
        return {"success": False, "error": "Inactive user account"}
    
    # Guard clause: Validate amount
    if amount <= 0:
        return {"success": False, "error": "Invalid payment amount"}
    
    if amount > 10000:
        return {"success": False, "error": "Amount exceeds maximum limit"}
    
    # Guard clause: Validate payment method
    if payment_method not in ["credit_card", "debit_card", "bank_transfer"]:
        return {"success": False, "error": "Invalid payment method"}
    
    # Guard clause: Validate currency
    if currency not in ["USD", "EUR", "GBP"]:
        return {"success": False, "error": "Unsupported currency"}
    
    # Happy path - process payment
    logger.info(f"Processing payment of {amount} {currency} for user {user.username}")
    return {
        "success": True,
        "transaction_id": f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "amount": amount,
        "currency": currency,
        "user_id": user.id
    }


# ============================================================================
# Complex Business Logic with Guard Clauses
# ============================================================================

def validate_model_deployment_bad(
    model_path: str,
    user: User,
    environment: str,
    resources: Dict[str, Any]
) -> Dict[str, Any]:
    """
    BAD: Nested if-else statements (avoid this)
    """
    result = {"success": False, "message": ""}
    
    if Path(model_path).exists():
        if user.is_active:
            if environment in ["dev", "staging", "prod"]:
                if resources.get("memory", 0) >= 512:
                    if resources.get("cpu", 0) >= 1:
                        if environment == "prod" and not user.username.startswith("admin"):
                            result["message"] = "Admin required for production deployment"
                        else:
                            result["success"] = True
                            result["message"] = "Deployment validated"
                    else:
                        result["message"] = "Insufficient CPU resources"
                else:
                    result["message"] = "Insufficient memory resources"
            else:
                result["message"] = "Invalid environment"
        else:
            result["message"] = "Inactive user"
    else:
        result["message"] = "Model file not found"
    
    return result


def validate_model_deployment_good(
    model_path: str,
    user: User,
    environment: str,
    resources: Dict[str, Any]
) -> Dict[str, Any]:
    """
    GOOD: Using guard clauses and if-return pattern
    """
    # Guard clause: Check if model file exists
    if not Path(model_path).exists():
        return {"success": False, "message": "Model file not found"}
    
    # Guard clause: Check if user is active
    if not user.is_active:
        return {"success": False, "message": "Inactive user"}
    
    # Guard clause: Validate environment
    if environment not in ["dev", "staging", "prod"]:
        return {"success": False, "message": "Invalid environment"}
    
    # Guard clause: Check memory resources
    if resources.get("memory", 0) < 512:
        return {"success": False, "message": "Insufficient memory resources"}
    
    # Guard clause: Check CPU resources
    if resources.get("cpu", 0) < 1:
        return {"success": False, "message": "Insufficient CPU resources"}
    
    # Guard clause: Check production permissions
    if environment == "prod" and not user.username.startswith("admin"):
        return {"success": False, "message": "Admin required for production deployment"}
    
    # Happy path - all validations passed
    return {"success": True, "message": "Deployment validated"}


# ============================================================================
# Data Processing with Guard Clauses
# ============================================================================

def process_dataset_bad(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    BAD: Nested if-else statements
    """
    processed_data = []
    
    for item in data:
        if item.get("value") is not None:
            if isinstance(item["value"], (int, float)):
                if config.get("normalize"):
                    if item["value"] > 0:
                        normalized_value = item["value"] / 100
                        processed_data.append({**item, "value": normalized_value})
                    else:
                        processed_data.append(item)
                else:
                    processed_data.append(item)
            else:
                if config.get("skip_invalid"):
                    continue
                else:
                    processed_data.append(item)
        else:
            if config.get("fill_missing"):
                processed_data.append({**item, "value": 0})
            else:
                processed_data.append(item)
    
    return processed_data


def process_dataset_good(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    GOOD: Using guard clauses and if-return pattern
    """
    processed_data = []
    
    for item in data:
        # Guard clause: Handle missing values
        if item.get("value") is None:
            if config.get("fill_missing"):
                processed_data.append({**item, "value": 0})
            else:
                processed_data.append(item)
            continue
        
        # Guard clause: Handle non-numeric values
        if not isinstance(item["value"], (int, float)):
            if config.get("skip_invalid"):
                continue
            else:
                processed_data.append(item)
            continue
        
        # Guard clause: Handle normalization
        if config.get("normalize") and item["value"] > 0:
            normalized_value = item["value"] / 100
            processed_data.append({**item, "value": normalized_value})
            continue
        
        # Happy path - no special processing needed
        processed_data.append(item)
    
    return processed_data


# ============================================================================
# API Response Handling with Guard Clauses
# ============================================================================

async def handle_api_response_bad(response: Dict[str, Any]) -> str:
    """
    BAD: Nested if-else statements
    """
    if response.get("status") == "success":
        if response.get("data"):
            if isinstance(response["data"], dict):
                if response["data"].get("message"):
                    return response["data"]["message"]
                else:
                    return "Success with no message"
            else:
                return "Invalid data format"
        else:
            return "No data in response"
    else:
        if response.get("error"):
            return f"Error: {response['error']}"
        else:
            return "Unknown error"


async def handle_api_response_good(response: Dict[str, Any]) -> str:
    """
    GOOD: Using guard clauses and if-return pattern
    """
    # Guard clause: Handle error responses
    if response.get("status") != "success":
        error_msg = response.get("error", "Unknown error")
        return f"Error: {error_msg}"
    
    # Guard clause: Check if data exists
    if not response.get("data"):
        return "No data in response"
    
    # Guard clause: Validate data format
    if not isinstance(response["data"], dict):
        return "Invalid data format"
    
    # Guard clause: Check for message
    if response["data"].get("message"):
        return response["data"]["message"]
    
    # Happy path - success with no message
    return "Success with no message"


# ============================================================================
# Configuration Validation with Guard Clauses
# ============================================================================

def validate_config_bad(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    BAD: Nested if-else statements
    """
    errors = []
    
    if "database" in config:
        db_config = config["database"]
        if "host" in db_config:
            if not isinstance(db_config["host"], str):
                errors.append("Database host must be a string")
        else:
            errors.append("Database host is required")
        
        if "port" in db_config:
            if not isinstance(db_config["port"], int):
                errors.append("Database port must be an integer")
            else:
                if db_config["port"] < 1 or db_config["port"] > 65535:
                    errors.append("Database port must be between 1 and 65535")
        else:
            errors.append("Database port is required")
    else:
        errors.append("Database configuration is required")
    
    if "api" in config:
        api_config = config["api"]
        if "timeout" in api_config:
            if not isinstance(api_config["timeout"], (int, float)):
                errors.append("API timeout must be a number")
            else:
                if api_config["timeout"] <= 0:
                    errors.append("API timeout must be positive")
        else:
            errors.append("API timeout is required")
    else:
        errors.append("API configuration is required")
    
    return {"valid": len(errors) == 0, "errors": errors}


def validate_config_good(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    GOOD: Using guard clauses and if-return pattern
    """
    errors = []
    
    # Guard clause: Check if database config exists
    if "database" not in config:
        errors.append("Database configuration is required")
        return {"valid": False, "errors": errors}
    
    db_config = config["database"]
    
    # Guard clause: Validate database host
    if "host" not in db_config:
        errors.append("Database host is required")
    elif not isinstance(db_config["host"], str):
        errors.append("Database host must be a string")
    
    # Guard clause: Validate database port
    if "port" not in db_config:
        errors.append("Database port is required")
    elif not isinstance(db_config["port"], int):
        errors.append("Database port must be an integer")
    elif db_config["port"] < 1 or db_config["port"] > 65535:
        errors.append("Database port must be between 1 and 65535")
    
    # Guard clause: Check if API config exists
    if "api" not in config:
        errors.append("API configuration is required")
        return {"valid": False, "errors": errors}
    
    api_config = config["api"]
    
    # Guard clause: Validate API timeout
    if "timeout" not in api_config:
        errors.append("API timeout is required")
    elif not isinstance(api_config["timeout"], (int, float)):
        errors.append("API timeout must be a number")
    elif api_config["timeout"] <= 0:
        errors.append("API timeout must be positive")
    
    # Happy path - all validations passed
    return {"valid": len(errors) == 0, "errors": errors}


# ============================================================================
# Performance Comparison Functions
# ============================================================================

def compare_performance():
    """Compare performance of bad vs good patterns"""
    
    # Test data
    user_data = {"username": "testuser", "email": "test@example.com"}
    user = User(id=1, username="admin_user", email="admin@example.com")
    
    # Test bad pattern
    start_time = time.time()
    for _ in range(10000):
        validate_user_data_bad(user_data)
    bad_time = time.time() - start_time
    
    # Test good pattern
    start_time = time.time()
    for _ in range(10000):
        validate_user_data_good(user_data)
    good_time = time.time() - start_time
    
    print(f"Bad pattern time: {bad_time:.4f}s")
    print(f"Good pattern time: {good_time:.4f}s")
    print(f"Improvement: {((bad_time - good_time) / bad_time * 100):.1f}%")


# ============================================================================
# Example Usage and Testing
# ============================================================================

def demonstrate_patterns():
    """Demonstrate if-return pattern and guard clauses"""
    print("=" * 60)
    print("If-Return Pattern and Guard Clauses Demonstration")
    print("=" * 60)
    
    # Test user validation
    print("\n1. User Validation:")
    test_cases = [
        {},
        {"username": "ab"},
        {"username": "testuser"},
        {"username": "testuser", "email": "invalid"},
        {"username": "testuser", "email": "test@example.com"}
    ]
    
    for i, case in enumerate(test_cases, 1):
        result_bad = validate_user_data_bad(case)
        result_good = validate_user_data_good(case)
        print(f"  Test {i}: {result_good}")
        assert result_bad == result_good, f"Results differ for case {i}"
    
    # Test user permissions
    print("\n2. User Permissions:")
    user = User(id=1, username="admin_user", email="admin@example.com")
    actions = ["read", "write", "delete", "invalid"]
    
    for action in actions:
        result_bad = process_user_permissions_bad(user, action)
        result_good = process_user_permissions_good(user, action)
        print(f"  {action}: {result_good}")
        assert result_bad == result_good, f"Results differ for action {action}"
    
    # Test model deployment validation
    print("\n3. Model Deployment Validation:")
    config = ModelConfig(
        model_type="neural_network",
        layers=[784, 512, 10],
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )
    
    result = train_model_with_guards(config, "nonexistent/path.csv")
    print(f"  Invalid path result: {result}")
    
    # Test payment processing
    print("\n4. Payment Processing:")
    payment_result = process_payment_with_guards(
        user=user,
        amount=100.0,
        payment_method="credit_card",
        currency="USD"
    )
    print(f"  Payment result: {payment_result}")
    
    # Test API response handling
    print("\n5. API Response Handling:")
    api_responses = [
        {"status": "error", "error": "Something went wrong"},
        {"status": "success", "data": {"message": "Operation completed"}},
        {"status": "success", "data": {}},
        {"status": "success"}
    ]
    
    for response in api_responses:
        result = handle_api_response_good(response)
        print(f"  Response: {result}")


if __name__ == "__main__":
    demonstrate_patterns()
    compare_performance() 