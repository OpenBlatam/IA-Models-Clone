from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Generic
from datetime import datetime, date, timedelta
import uuid
import re
from enum import Enum
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
import structlog
from .pydantic_schemas import (
from .schema_validators import validation_registry, ValidationResult
from .schema_integration import (
    from fastapi import APIRouter, HTTPException, status
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Schema Examples - Comprehensive Usage Examples
Complete examples demonstrating Pydantic schema usage, patterns, and best practices.
"""



    BaseInputModel, BaseOutputModel, PaginatedOutputModel,
    SchemaFactory, schema_registry, validate_input, validate_output
)
    schema_endpoint, OnyxSchemaIntegration, SchemaResponseHandler
)

logger = structlog.get_logger(__name__)

# Example 1: E-commerce Product Schemas
class ProductCategory(str, Enum):
    """Product categories."""
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    HOME = "home"
    SPORTS = "sports"

class ProductCreateInput(BaseInputModel):
    """Input schema for product creation."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: str = Field(..., min_length=10, max_length=2000, description="Product description")
    price: float = Field(..., gt=0, le=1000000, description="Product price")
    category: ProductCategory = Field(..., description="Product category")
    sku: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Z0-9-]+$", description="Stock keeping unit")
    stock_quantity: int = Field(..., ge=0, le=100000, description="Available stock quantity")
    weight_kg: Optional[float] = Field(None, gt=0, le=1000, description="Product weight in kg")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Product dimensions (length, width, height)")
    tags: List[str] = Field(default_factory=list, max_length=20, description="Product tags")
    is_active: bool = Field(default=True, description="Product availability")
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and clean product name."""
        v = v.strip()
        if not v:
            raise ValueError("Product name cannot be empty")
        return v.title()
    
    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price with proper rounding."""
        return round(v, 2)
    
    @field_validator("sku")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        """Validate and normalize SKU."""
        v = v.upper().strip()
        if not re.match(r"^[A-Z0-9-]+$", v):
            raise ValueError("SKU must contain only uppercase letters, numbers, and hyphens")
        return v
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Validate and clean tags."""
        # Remove duplicates and empty tags
        cleaned_tags = list(set(tag.strip().lower() for tag in v if tag.strip()))
        
        # Limit number of tags
        if len(cleaned_tags) > 20:
            raise ValueError("Maximum 20 tags allowed")
        
        return cleaned_tags
    
    @model_validator(mode='after')
    def validate_product_data(self) -> 'ProductCreateInput':
        """Validate product data after all fields are set."""
        # Check if price is reasonable for category
        if self.category == ProductCategory.ELECTRONICS and self.price < 10:
            raise ValueError("Electronics products should cost at least $10")
        
        # Check if weight is provided for physical products
        if self.category != ProductCategory.BOOKS and self.weight_kg is None:
            raise ValueError("Weight is required for physical products")
        
        return self

class ProductOutput(BaseOutputModel):
    """Output schema for product data."""
    
    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    price: float = Field(..., description="Product price")
    category: ProductCategory = Field(..., description="Product category")
    sku: str = Field(..., description="Stock keeping unit")
    stock_quantity: int = Field(..., description="Available stock quantity")
    weight_kg: Optional[float] = Field(None, description="Product weight in kg")
    dimensions: Optional[Dict[str, float]] = Field(None, description="Product dimensions")
    tags: List[str] = Field(default_factory=list, description="Product tags")
    is_active: bool = Field(..., description="Product availability")
    created_at: datetime = Field(..., description="Creation date")
    updated_at: datetime = Field(..., description="Last update date")
    view_count: int = Field(default=0, description="Product view count")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Average rating")
    review_count: int = Field(default=0, description="Number of reviews")
    
    @computed_field
    @property
    def price_formatted(self) -> str:
        """Get formatted price."""
        return f"${self.price:.2f}"
    
    @computed_field
    @property
    def stock_status(self) -> str:
        """Get stock status."""
        if self.stock_quantity == 0:
            return "out_of_stock"
        elif self.stock_quantity < 10:
            return "low_stock"
        else:
            return "in_stock"
    
    @computed_field
    @property
    def is_popular(self) -> bool:
        """Check if product is popular."""
        return self.view_count > 1000 or (self.rating and self.rating >= 4.5)
    
    @computed_field
    @property
    def volume_cm3(self) -> Optional[float]:
        """Calculate product volume."""
        if self.dimensions:
            return (
                self.dimensions.get('length', 0) *
                self.dimensions.get('width', 0) *
                self.dimensions.get('height', 0)
            )
        return None

class ProductListOutput(PaginatedOutputModel[ProductOutput]):
    """Output schema for paginated product list."""
    
    items: List[ProductOutput] = Field(default_factory=list, description="List of products")
    total_count: int = Field(default=0, description="Total number of products")
    pagination: PaginationInfo = Field(default_factory=PaginationInfo, description="Pagination information")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")

# Example 2: User Management Schemas
class UserRole(str, Enum):
    """User roles."""
    ADMIN = "admin"
    MODERATOR = "moderator"
    USER = "user"
    GUEST = "guest"

class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class UserCreateInput(BaseInputModel):
    """Input schema for user creation."""
    
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$", description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    date_of_birth: Optional[date] = Field(None, description="Date of birth")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    is_active: bool = Field(default=True, description="Account status")
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate and normalize username."""
        v = v.lower().strip()
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username must contain only letters, numbers, and underscores")
        return v
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and normalize email."""
        v = v.lower().strip()
        if not re.match(r"^[^@]+@[^@]+\.[^@]+$", v):
            raise ValueError("Invalid email format")
        return v
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not re.search(r'[A-Z]', v):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not re.search(r'[a-z]', v):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not re.search(r'\d', v):
            raise ValueError("Password must contain at least one digit")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError("Password must contain at least one special character")
        
        return v
    
    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, v: Optional[date]) -> Optional[date]:
        """Validate date of birth."""
        if v is not None:
            age = (date.today() - v).days / 365.25
            if age < 13:
                raise ValueError("User must be at least 13 years old")
            if age > 120:
                raise ValueError("Invalid date of birth")
        return v
    
    @model_validator(mode='after')
    def validate_user_data(self) -> 'UserCreateInput':
        """Validate user data after all fields are set."""
        # Check if username is not too similar to email
        if self.username.lower() in self.email.lower():
            raise ValueError("Username should not be part of email address")
        
        return self

class UserOutput(BaseOutputModel):
    """Output schema for user data."""
    
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    date_of_birth: Optional[date] = Field(None, description="Date of birth")
    role: UserRole = Field(..., description="User role")
    status: UserStatus = Field(..., description="Account status")
    is_active: bool = Field(..., description="Account status")
    created_at: datetime = Field(..., description="Account creation date")
    updated_at: datetime = Field(..., description="Last update date")
    last_login_at: Optional[datetime] = Field(None, description="Last login date")
    login_count: int = Field(default=0, description="Number of logins")
    profile_completion: float = Field(default=0.0, ge=0, le=100, description="Profile completion percentage")
    
    @computed_field
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def age(self) -> Optional[int]:
        """Calculate user age."""
        if self.date_of_birth:
            return (date.today() - self.date_of_birth).days // 365
        return None
    
    @computed_field
    @property
    def is_adult(self) -> bool:
        """Check if user is adult."""
        return self.age is not None and self.age >= 18
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.username or self.full_name
    
    @computed_field
    @property
    def is_admin(self) -> bool:
        """Check if user is admin."""
        return self.role in [UserRole.ADMIN, UserRole.MODERATOR]

# Example 3: Order Management Schemas
class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class PaymentMethod(str, Enum):
    """Payment methods."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"

class OrderItemInput(BaseInputModel):
    """Input schema for order item."""
    
    product_id: str = Field(..., description="Product ID")
    quantity: int = Field(..., gt=0, le=100, description="Quantity")
    unit_price: float = Field(..., gt=0, description="Unit price")
    
    @field_validator("unit_price")
    @classmethod
    def validate_unit_price(cls, v: float) -> float:
        """Validate and round unit price."""
        return round(v, 2)

class OrderCreateInput(BaseInputModel):
    """Input schema for order creation."""
    
    customer_id: str = Field(..., description="Customer ID")
    items: List[OrderItemInput] = Field(..., min_length=1, max_length=50, description="Order items")
    shipping_address: Dict[str, Any] = Field(..., description="Shipping address")
    billing_address: Optional[Dict[str, Any]] = Field(None, description="Billing address")
    payment_method: PaymentMethod = Field(..., description="Payment method")
    notes: Optional[str] = Field(None, max_length=500, description="Order notes")
    
    @field_validator("items")
    @classmethod
    def validate_items(cls, v: List[OrderItemInput]) -> List[OrderItemInput]:
        """Validate order items."""
        if not v:
            raise ValueError("Order must contain at least one item")
        
        # Check for duplicate products
        product_ids = [item.product_id for item in v]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Order cannot contain duplicate products")
        
        return v
    
    @field_validator("shipping_address")
    @classmethod
    def validate_shipping_address(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate shipping address."""
        required_fields = ['street', 'city', 'state', 'postal_code', 'country']
        missing_fields = [field for field in required_fields if field not in v or not v[field]]
        
        if missing_fields:
            raise ValueError(f"Missing required shipping address fields: {', '.join(missing_fields)}")
        
        return v
    
    @model_validator(mode='after')
    def validate_order_data(self) -> 'OrderCreateInput':
        """Validate order data after all fields are set."""
        # Set billing address to shipping address if not provided
        if not self.billing_address:
            self.billing_address = self.shipping_address
        
        return self

class OrderItemOutput(BaseOutputModel):
    """Output schema for order item."""
    
    id: str = Field(..., description="Order item ID")
    product_id: str = Field(..., description="Product ID")
    product_name: str = Field(..., description="Product name")
    quantity: int = Field(..., description="Quantity")
    unit_price: float = Field(..., description="Unit price")
    total_price: float = Field(..., description="Total price")
    
    @computed_field
    @property
    def total_price_formatted(self) -> str:
        """Get formatted total price."""
        return f"${self.total_price:.2f}"

class OrderOutput(BaseOutputModel):
    """Output schema for order data."""
    
    id: str = Field(..., description="Order ID")
    customer_id: str = Field(..., description="Customer ID")
    customer_name: str = Field(..., description="Customer name")
    items: List[OrderItemOutput] = Field(default_factory=list, description="Order items")
    status: OrderStatus = Field(..., description="Order status")
    payment_method: PaymentMethod = Field(..., description="Payment method")
    shipping_address: Dict[str, Any] = Field(..., description="Shipping address")
    billing_address: Dict[str, Any] = Field(..., description="Billing address")
    subtotal: float = Field(..., description="Subtotal")
    tax_amount: float = Field(..., description="Tax amount")
    shipping_cost: float = Field(..., description="Shipping cost")
    total_amount: float = Field(..., description="Total amount")
    notes: Optional[str] = Field(None, description="Order notes")
    created_at: datetime = Field(..., description="Order creation date")
    updated_at: datetime = Field(..., description="Last update date")
    estimated_delivery: Optional[date] = Field(None, description="Estimated delivery date")
    
    @computed_field
    @property
    def item_count(self) -> int:
        """Get number of items in order."""
        return len(self.items)
    
    @computed_field
    @property
    def total_quantity(self) -> int:
        """Get total quantity of items."""
        return sum(item.quantity for item in self.items)
    
    @computed_field
    @property
    def is_high_value(self) -> bool:
        """Check if order is high value."""
        return self.total_amount > 1000
    
    @computed_field
    @property
    def can_cancel(self) -> bool:
        """Check if order can be cancelled."""
        return self.status in [OrderStatus.PENDING, OrderStatus.CONFIRMED]
    
    @computed_field
    @property
    def total_amount_formatted(self) -> str:
        """Get formatted total amount."""
        return f"${self.total_amount:.2f}"

# Example 4: Dynamic Schema Creation
def create_dynamic_schemas():
    """Create schemas dynamically using SchemaFactory."""
    
    # Create a dynamic product schema
    product_fields = {
        'name': (str, Field(..., min_length=1, max_length=100)),
        'price': (float, Field(..., gt=0)),
        'category': (str, Field(..., max_length=50)),
        'description': (str, Field(default="", max_length=500)),
    }
    
    DynamicProductInput = SchemaFactory.create_input_schema(
        "DynamicProductInput",
        product_fields,
        description="Dynamically created product input schema"
    )
    
    # Create a dynamic user schema
    user_fields = {
        'username': (str, Field(..., min_length=3, max_length=50)),
        'email': (str, Field(..., description="Email address")),
        'age': (int, Field(..., ge=0, le=120)),
    }
    
    DynamicUserInput = SchemaFactory.create_input_schema(
        "DynamicUserInput",
        user_fields,
        description="Dynamically created user input schema"
    )
    
    return DynamicProductInput, DynamicUserInput

# Example 5: Schema Validation Examples
def demonstrate_validation():
    """Demonstrate various validation scenarios."""
    
    # Valid product creation
    try:
        product_data = {
            "name": "iPhone 15 Pro",
            "description": "Latest iPhone with advanced features",
            "price": 999.99,
            "category": ProductCategory.ELECTRONICS,
            "sku": "IPHONE-15-PRO-256",
            "stock_quantity": 50,
            "weight_kg": 0.187,
            "dimensions": {"length": 14.7, "width": 7.1, "height": 0.8},
            "tags": ["smartphone", "apple", "5g"],
            "is_active": True
        }
        
        product_input = ProductCreateInput(**product_data)
        logger.info(f"Valid product created: {product_input.name}")
        
    except ValidationError as e:
        logger.error(f"Product validation failed: {e}")
    
    # Invalid product creation
    try:
        invalid_product_data = {
            "name": "",  # Empty name
            "description": "Short",  # Too short
            "price": -10,  # Negative price
            "category": "invalid_category",  # Invalid category
            "sku": "invalid sku",  # Invalid SKU format
            "stock_quantity": -5,  # Negative stock
        }
        
        invalid_product = ProductCreateInput(**invalid_product_data)
        
    except ValidationError as e:
        logger.info(f"Expected validation errors: {e}")
    
    # Valid user creation
    try:
        user_data = {
            "username": "john_doe",
            "email": "john.doe@example.com",
            "password": "SecurePass123!",
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+1234567890",
            "date_of_birth": date(1990, 1, 1),
            "role": UserRole.USER,
            "is_active": True
        }
        
        user_input = UserCreateInput(**user_data)
        logger.info(f"Valid user created: {user_input.username}")
        
    except ValidationError as e:
        logger.error(f"User validation failed: {e}")

# Example 6: Integration with Onyx Models
def demonstrate_onyx_integration():
    """Demonstrate integration with Onyx models."""
    
    # Mock Onyx model
    class MockOnyxProduct:
        def __init__(self, **data) -> Any:
            self.id = data.get('id', str(uuid.uuid4()))
            self.name = data.get('name', '')
            self.price = data.get('price', 0.0)
            self.category = data.get('category', '')
            self.created_at = data.get('created_at', datetime.utcnow())
        
        def to_dict(self) -> Any:
            return {
                'id': self.id,
                'name': self.name,
                'price': self.price,
                'category': self.category,
                'created_at': self.created_at
            }
    
    # Convert Onyx model to Pydantic
    onyx_product = MockOnyxProduct(
        name="Test Product",
        price=99.99,
        category="electronics"
    )
    
    try:
        pydantic_product = OnyxSchemaIntegration.onyx_to_pydantic(
            onyx_product, 
            ProductOutput
        )
        logger.info(f"Converted Onyx model to Pydantic: {pydantic_product.name}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
    
    # Validate Onyx model
    validation_result = OnyxSchemaIntegration.validate_onyx_model(
        onyx_product, 
        "ProductOutput"
    )
    
    if validation_result.is_valid:
        logger.info("Onyx model validation successful")
    else:
        logger.warning(f"Onyx model validation failed: {validation_result.errors}")

# Example 7: API Endpoint Examples
def create_api_examples():
    """Create example API endpoints with schema integration."""
    
    
    router = APIRouter()
    
    @router.post("/products/", response_model=SuccessResponse[ProductOutput])
    @schema_endpoint(
        input_model=ProductCreateInput,
        output_model=ProductOutput,
        validate_input=True,
        validate_output=True
    )
    async def create_product(product_data: ProductCreateInput):
        """Create a new product."""
        try:
            # Mock product creation
            product = ProductOutput(
                id=str(uuid.uuid4()),
                name=product_data.name,
                description=product_data.description,
                price=product_data.price,
                category=product_data.category,
                sku=product_data.sku,
                stock_quantity=product_data.stock_quantity,
                weight_kg=product_data.weight_kg,
                dimensions=product_data.dimensions,
                tags=product_data.tags,
                is_active=product_data.is_active,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            return SchemaResponseHandler.success_response(
                data=product.model_dump(),
                message="Product created successfully",
                status_code=201
            )
            
        except Exception as e:
            logger.error(f"Error creating product: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create product"
            )
    
    @router.get("/products/{product_id}", response_model=SuccessResponse[ProductOutput])
    async def get_product(product_id: str):
        """Get product by ID."""
        try:
            # Mock product retrieval
            product = ProductOutput(
                id=product_id,
                name="Sample Product",
                description="A sample product for demonstration",
                price=99.99,
                category=ProductCategory.ELECTRONICS,
                sku="SAMPLE-001",
                stock_quantity=100,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            return SchemaResponseHandler.success_response(
                data=product.model_dump(),
                message="Product retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Error retrieving product: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found"
            )
    
    return router

# Register example schemas
def register_example_schemas():
    """Register all example schemas."""
    
    # Register product schemas
    schema_registry.register_schema("ProductCreateInput", ProductCreateInput)
    schema_registry.register_schema("ProductOutput", ProductOutput)
    schema_registry.register_schema("ProductListOutput", ProductListOutput)
    
    # Register user schemas
    schema_registry.register_schema("UserCreateInput", UserCreateInput)
    schema_registry.register_schema("UserOutput", UserOutput)
    
    # Register order schemas
    schema_registry.register_schema("OrderCreateInput", OrderCreateInput)
    schema_registry.register_schema("OrderOutput", OrderOutput)
    schema_registry.register_schema("OrderItemInput", OrderItemInput)
    schema_registry.register_schema("OrderItemOutput", OrderItemOutput)
    
    logger.info("Example schemas registered successfully")

# Run examples
if __name__ == "__main__":
    # Register schemas
    register_example_schemas()
    
    # Demonstrate validation
    demonstrate_validation()
    
    # Demonstrate Onyx integration
    demonstrate_onyx_integration()
    
    # Create dynamic schemas
    DynamicProductInput, DynamicUserInput = create_dynamic_schemas()
    logger.info(f"Created dynamic schemas: {DynamicProductInput.__name__}, {DynamicUserInput.__name__}")
    
    logger.info("Schema examples completed successfully") 