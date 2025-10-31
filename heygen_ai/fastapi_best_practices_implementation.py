from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime, timedelta
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat
    import uvicorn
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices Implementation
====================================

This module demonstrates Python/FastAPI best practices including:
- Type hints for all function signatures
- Pydantic models over raw dictionaries
- Async/sync function separation
- Clean file structure with routers, utilities, types
- Error handling and validation patterns
- Early returns and happy path placement
"""



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models (Types)
# ============================================================================

class UserBase(BaseModel):
    """Base user model with validation"""
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    is_active: bool = Field(default=True, description="User active status")
    
    @validator('email')
    def validate_email(cls, v: str) -> str:
        """Validate email format"""
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username format"""
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8, description="User password")
    
    @validator('password')
    def validate_password(cls, v: str) -> str:
        """Validate password strength"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v


class UserResponse(UserBase):
    """User response model"""
    id: int = Field(..., description="User ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "email": "user@example.com",
                "username": "testuser",
                "is_active": True,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        }


class ModelTrainingRequest(BaseModel):
    """Model training request model"""
    model_type: str = Field(..., description="Type of model to train")
    training_data_path: str = Field(..., description="Path to training data")
    epochs: PositiveInt = Field(default=100, description="Number of training epochs")
    batch_size: PositiveInt = Field(default=32, description="Training batch size")
    learning_rate: PositiveFloat = Field(default=0.001, description="Learning rate")
    validation_split: float = Field(default=0.2, ge=0.0, le=1.0, description="Validation split")
    
    @root_validator
    def validate_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file paths exist"""
        training_path = values.get('training_data_path')
        if training_path and not Path(training_path).exists():
            raise ValueError(f'Training data path does not exist: {training_path}')
        return values


class ModelTrainingResponse(BaseModel):
    """Model training response model"""
    model_id: str = Field(..., description="Trained model ID")
    final_loss: float = Field(..., description="Final training loss")
    final_accuracy: float = Field(..., description="Final training accuracy")
    training_time: float = Field(..., description="Training duration in seconds")
    model_path: str = Field(..., description="Path to saved model")
    created_at: datetime = Field(default_factory=datetime.now, description="Training completion time")


class PredictionRequest(BaseModel):
    """Prediction request model"""
    model_id: str = Field(..., description="Model ID to use for prediction")
    input_data: List[List[float]] = Field(..., description="Input data for prediction")
    preprocessing_params: Optional[Dict[str, Any]] = Field(default=None, description="Preprocessing parameters")
    
    @validator('input_data')
    def validate_input_data(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate input data format"""
        if not v:
            raise ValueError('Input data cannot be empty')
        if not all(isinstance(row, list) for row in v):
            raise ValueError('Input data must be list of lists')
        return v


class PredictionResponse(BaseModel):
    """Prediction response model"""
    predictions: List[float] = Field(..., description="Model predictions")
    confidence_scores: List[float] = Field(..., description="Prediction confidence scores")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model ID used for prediction")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# ============================================================================
# Custom Exceptions
# ============================================================================

class ModelNotFoundError(Exception):
    """Raised when model is not found"""
    pass


class TrainingError(Exception):
    """Raised when model training fails"""
    pass


class ValidationError(Exception):
    """Raised when data validation fails"""
    pass


class DatabaseError(Exception):
    """Raised when database operation fails"""
    pass


# ============================================================================
# Utilities
# ============================================================================

def validate_file_path(file_path: str) -> bool:
    """Validate if file path exists and is accessible"""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


async def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


async def log_request_info(request_id: str, endpoint: str, method: str) -> None:
    """Log request information"""
    logger.info(f"Request {request_id}: {method} {endpoint}")


def log_error(request_id: str, error: Exception, context: str) -> None:
    """Log error information"""
    logger.error(f"Request {request_id} failed in {context}: {str(error)}")


# ============================================================================
# Database Utilities (Mock)
# ============================================================================

class DatabaseManager:
    """Mock database manager for demonstration"""
    
    def __init__(self) -> None:
        self.users: Dict[int, UserResponse] = {}
        self.models: Dict[str, ModelTrainingResponse] = {}
        self.user_counter = 0
        self.model_counter = 0
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user"""
        if not user_data.email or not user_data.username:
            raise ValidationError("Email and username are required")
        
        # Check for duplicate email
        for user in self.users.values():
            if user.email == user_data.email:
                raise ValidationError("Email already exists")
        
        self.user_counter += 1
        user = UserResponse(
            id=self.user_counter,
            email=user_data.email,
            username=user_data.username,
            is_active=user_data.is_active,
            created_at=datetime.now()
        )
        self.users[user.id] = user
        return user
    
    async def get_user(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    async def save_model(self, model_data: ModelTrainingResponse) -> str:
        """Save trained model"""
        model_id = f"model_{self.model_counter}"
        self.model_counter += 1
        self.models[model_id] = model_data
        return model_id
    
    async def get_model(self, model_id: str) -> Optional[ModelTrainingResponse]:
        """Get model by ID"""
        return self.models.get(model_id)


# ============================================================================
# Dependencies
# ============================================================================

async def get_database() -> DatabaseManager:
    """Dependency to get database manager"""
    return DatabaseManager()


async def get_current_user(user_id: int, db: DatabaseManager = Depends(get_database)) -> UserResponse:
    """Dependency to get current user"""
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return user


# ============================================================================
# Pure Functions (def)
# ============================================================================

def calculate_model_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
    """Calculate model performance metrics"""
    if not predictions or not targets:
        raise ValidationError("Predictions and targets cannot be empty")
    
    if len(predictions) != len(targets):
        raise ValidationError("Predictions and targets must have same length")
    
    # Calculate metrics
    mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)
    mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
    
    return {
        "mse": mse,
        "mae": mae,
        "rmse": mse ** 0.5
    }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """Validate model configuration"""
    required_fields = ["model_type", "layers", "activation"]
    
    for field in required_fields:
        if field not in config:
            raise ValidationError(f"Missing required field: {field}")
    
    if not isinstance(config["layers"], list):
        raise ValidationError("Layers must be a list")
    
    if config["activation"] not in ["relu", "tanh", "sigmoid"]:
        raise ValidationError("Invalid activation function")
    
    return True


def preprocess_input_data(data: List[List[float]], params: Optional[Dict[str, Any]] = None) -> List[List[float]]:
    """Preprocess input data"""
    if not data:
        raise ValidationError("Input data cannot be empty")
    
    processed_data = data.copy()
    
    if params and params.get("normalize"):
        # Simple normalization
        for i, row in enumerate(processed_data):
            if row:
                max_val = max(row)
                if max_val > 0:
                    processed_data[i] = [x / max_val for x in row]
    
    return processed_data


# ============================================================================
# Async Functions (async def)
# ============================================================================

async def train_model_async(request: ModelTrainingRequest) -> ModelTrainingResponse:
    """Async model training function"""
    # Early error handling
    if not validate_file_path(request.training_data_path):
        raise TrainingError(f"Training data path not found: {request.training_data_path}")
    
    if request.epochs <= 0:
        raise TrainingError("Epochs must be positive")
    
    if request.batch_size <= 0:
        raise TrainingError("Batch size must be positive")
    
    # Simulate async training
    start_time = datetime.now()
    await asyncio.sleep(2)  # Simulate training time
    
    # Happy path - training completed successfully
    training_time = (datetime.now() - start_time).total_seconds()
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return ModelTrainingResponse(
        model_id=model_id,
        final_loss=0.15,
        final_accuracy=0.92,
        training_time=training_time,
        model_path=f"models/{model_id}.pth"
    )


async def predict_async(request: PredictionRequest, db: DatabaseManager) -> PredictionResponse:
    """Async prediction function"""
    # Early error handling
    if not request.input_data:
        raise ValidationError("Input data cannot be empty")
    
    model = await db.get_model(request.model_id)
    if not model:
        raise ModelNotFoundError(f"Model not found: {request.model_id}")
    
    # Preprocess input data
    processed_data = preprocess_input_data(request.input_data, request.preprocessing_params)
    
    # Simulate async prediction
    start_time = datetime.now()
    await asyncio.sleep(0.5)  # Simulate prediction time
    
    # Happy path - prediction completed successfully
    processing_time = (datetime.now() - start_time).total_seconds()
    predictions = [0.8, 0.2, 0.9]  # Mock predictions
    confidence_scores = [0.85, 0.75, 0.92]  # Mock confidence scores
    
    return PredictionResponse(
        predictions=predictions,
        confidence_scores=confidence_scores,
        processing_time=processing_time,
        model_used=request.model_id
    )


async def create_user_async(user_data: UserCreate, db: DatabaseManager) -> UserResponse:
    """Async user creation function"""
    # Early error handling
    if not user_data.email or not user_data.username:
        raise ValidationError("Email and username are required")
    
    # Check for existing user
    for user in db.users.values():
        if user.email == user_data.email:
            raise ValidationError("Email already exists")
    
    # Happy path - user creation
    return await db.create_user(user_data)


# ============================================================================
# Router Definitions
# ============================================================================

# Main router
router = APIRouter(prefix="/api/v1", tags=["api"])

# User router
user_router = APIRouter(prefix="/users", tags=["users"])

# Model router
model_router = APIRouter(prefix="/models", tags=["models"])

# Prediction router
prediction_router = APIRouter(prefix="/predictions", tags=["predictions"])


# ============================================================================
# Route Handlers
# ============================================================================

@user_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: DatabaseManager = Depends(get_database)
) -> UserResponse:
    """Create new user endpoint"""
    request_id = generate_request_id()
    log_request_info(request_id, "/users", "POST")
    
    try:
        user = await create_user_async(user_data, db)
        logger.info(f"User created successfully: {user.id}")
        return user
    except ValidationError as e:
        log_error(request_id, e, "user_creation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log_error(request_id, e, "user_creation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """Get user by ID endpoint"""
    request_id = generate_request_id()
    log_request_info(request_id, f"/users/{user_id}", "GET")
    
    # Early error handling
    if user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Happy path
    return current_user


@model_router.post("/train", response_model=ModelTrainingResponse, status_code=status.HTTP_201_CREATED)
async def train_model(
    request: ModelTrainingRequest,
    current_user: UserResponse = Depends(get_current_user)
) -> ModelTrainingResponse:
    """Train model endpoint"""
    request_id = generate_request_id()
    log_request_info(request_id, "/models/train", "POST")
    
    try:
        model = await train_model_async(request)
        logger.info(f"Model training completed: {model.model_id}")
        return model
    except TrainingError as e:
        log_error(request_id, e, "model_training")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log_error(request_id, e, "model_training")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Training failed"
        )


@prediction_router.post("/", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest,
    current_user: UserResponse = Depends(get_current_user),
    db: DatabaseManager = Depends(get_database)
) -> PredictionResponse:
    """Make prediction endpoint"""
    request_id = generate_request_id()
    log_request_info(request_id, "/predictions", "POST")
    
    try:
        prediction = await predict_async(request, db)
        logger.info(f"Prediction completed for user: {current_user.id}")
        return prediction
    except ModelNotFoundError as e:
        log_error(request_id, e, "prediction")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValidationError as e:
        log_error(request_id, e, "prediction")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        log_error(request_id, e, "prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@router.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            request_id=generate_request_id()
        ).dict()
    )


@router.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request, exc: ModelNotFoundError) -> JSONResponse:
    """Handle model not found errors"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="Model Not Found",
            detail=str(exc),
            request_id=generate_request_id()
        ).dict()
    )


@router.exception_handler(TrainingError)
async def training_error_handler(request, exc: TrainingError) -> JSONResponse:
    """Handle training errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Training Error",
            detail=str(exc),
            request_id=generate_request_id()
        ).dict()
    )


# ============================================================================
# Include Routers
# ============================================================================

router.include_router(user_router)
router.include_router(model_router)
router.include_router(prediction_router)


# ============================================================================
# Main Application
# ============================================================================

app = FastAPI(
    title="FastAPI Best Practices Demo",
    description="Demonstration of FastAPI best practices",
    version="1.0.0"
)

app.include_router(router)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Example Usage Functions
# ============================================================================

def demonstrate_type_hints() -> None:
    """Demonstrate type hints usage"""
    # Pure function with type hints
    def calculate_sum(numbers: List[int]) -> int:
        return sum(numbers)
    
    # Async function with type hints
    async def process_data_async(data: List[str]) -> List[str]:
        return [item.upper() for item in data]
    
    # Function with complex types
    def process_user_data(user: UserResponse, config: Dict[str, Any]) -> bool:
        return user.is_active and config.get("enabled", False)


def demonstrate_conditional_statements() -> None:
    """Demonstrate clean conditional statements"""
    user_active = True
    user_verified = True
    
    # Single-line conditionals without braces
    if user_active: print("User is active")
    if user_verified: print("User is verified")
    
    # Early returns for error conditions
    def process_user(user_id: int) -> str:
        if user_id <= 0:
            return "Invalid user ID"
        
        if user_id > 1000:
            return "User ID too large"
        
        # Happy path last
        return f"Processing user {user_id}"


def demonstrate_error_handling() -> None:
    """Demonstrate error handling patterns"""
    def safe_divide(a: float, b: float) -> float:
        # Early error handling
        if b == 0:
            raise ValueError("Division by zero")
        
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError("Both arguments must be numbers")
        
        # Happy path last
        return a / b


match __name__:
    case "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 