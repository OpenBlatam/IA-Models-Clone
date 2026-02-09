# Pydantic Schema System - Comprehensive Documentation

## Overview

The Pydantic Schema System provides a comprehensive, production-ready solution for consistent input/output validation and response schemas in the Blatam Academy backend. This system builds on Pydantic v2 best practices and integrates seamlessly with the existing Onyx infrastructure.

## Architecture

### Core Components

1. **Base Models**
   - `BaseInputModel`: Foundation for all input validation
   - `BaseOutputModel`: Foundation for all output responses
   - `PaginatedOutputModel`: Specialized model for paginated responses

2. **Schema Registry**
   - Central registry for managing schemas and validation rules
   - Dynamic schema creation and registration
   - Schema validation and caching

3. **Validation System**
   - Custom validators for common data types
   - Business rule validation
   - Performance monitoring and metrics

4. **Integration Layer**
   - FastAPI integration with decorators
   - Onyx model integration
   - Middleware and dependency injection

## Key Features

### 1. Input Validation (`BaseInputModel`)

```python
class ProductCreateInput(BaseInputModel):
    name: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., gt=0, le=1000000)
    category: ProductCategory = Field(...)
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip().title()
    
    @model_validator(mode='after')
    def validate_product_data(self) -> 'ProductCreateInput':
        if self.category == ProductCategory.ELECTRONICS and self.price < 10:
            raise ValueError("Electronics products should cost at least $10")
        return self
```

**Features:**
- Strict validation with detailed error messages
- Performance monitoring and audit logging
- Input sanitization and normalization
- Custom validation rules
- Business logic validation

### 2. Output Responses (`BaseOutputModel`)

```python
class ProductOutput(BaseOutputModel):
    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    price: float = Field(..., description="Product price")
    
    @computed_field
    @property
    def price_formatted(self) -> str:
        return f"${self.price:.2f}"
    
    @computed_field
    @property
    def is_popular(self) -> bool:
        return self.view_count > 1000 or self.rating >= 4.5
```

**Features:**
- Consistent response formatting
- Computed fields and properties
- Performance monitoring
- Serialization optimization
- Response metadata

### 3. Pagination Support (`PaginatedOutputModel`)

```python
class ProductListOutput(PaginatedOutputModel[ProductOutput]):
    items: List[ProductOutput] = Field(default_factory=list)
    total_count: int = Field(default=0)
    pagination: PaginationInfo = Field(default_factory=PaginationInfo)
    
    @computed_field
    @property
    def has_more(self) -> bool:
        return self.pagination.page < self.pagination.total_pages
```

**Features:**
- Generic type support for any model
- Built-in pagination information
- Computed properties for pagination state
- Consistent list response format

### 4. Custom Validators

```python
class EmailValidator(BaseValidator):
    def __init__(self, allow_disposable: bool = False, check_mx: bool = False):
        super().__init__("EmailValidator")
        self.allow_disposable = allow_disposable
        self.check_mx = check_mx
    
    def _validate_impl(self, value: Any, **kwargs) -> ValidationResult:
        result = ValidationResult(True)
        
        # Basic email pattern
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            result.add_error("Invalid email format")
        
        # Check for disposable domains
        if not self.allow_disposable:
            domain = value.split('@')[1]
            if domain in self._disposable_domains:
                result.add_error("Disposable email addresses are not allowed")
        
        return result
```

**Available Validators:**
- `StringValidator`: String length, pattern, and value validation
- `EmailValidator`: Email format and domain validation
- `PhoneValidator`: Phone number format and country validation
- `URLValidator`: URL format and security validation
- `DateValidator`: Date range and business rule validation
- `FileValidator`: File type and size validation
- `BusinessRuleValidator`: Custom business logic validation

### 5. Schema Registry

```python
# Register schemas
schema_registry.register_schema("ProductCreateInput", ProductCreateInput)
schema_registry.register_schema("ProductOutput", ProductOutput)

# Register validation rules
validation_rules = [
    ValidationSchema(
        field_name="price",
        field_type="float",
        required=True,
        min_value=0,
        max_value=1000000
    )
]
schema_registry.register_validation_rules("ProductCreateInput", validation_rules)

# Validate data
validated_data = schema_registry.validate_data("ProductCreateInput", input_data)
```

**Features:**
- Centralized schema management
- Dynamic schema registration
- Validation rule management
- Performance monitoring
- Error tracking

### 6. FastAPI Integration

```python
@router.post("/products/", response_model=SuccessResponse[ProductOutput])
@schema_endpoint(
    input_model=ProductCreateInput,
    output_model=ProductOutput,
    validate_input=True,
    validate_output=True
)
async def create_product(product_data: ProductCreateInput):
    """Create a new product."""
    # Input is automatically validated
    # Output is automatically validated and formatted
    product = ProductOutput(
        id=str(uuid.uuid4()),
        name=product_data.name,
        price=product_data.price,
        # ... other fields
    )
    
    return SchemaResponseHandler.success_response(
        data=product.model_dump(),
        message="Product created successfully",
        status_code=201
    )
```

**Features:**
- Automatic input/output validation
- Standardized response formatting
- Performance monitoring
- Error handling
- Audit logging

### 7. Onyx Integration

```python
# Convert Onyx model to Pydantic
onyx_product = OnyxProduct.get_by_id(product_id)
pydantic_product = OnyxSchemaIntegration.onyx_to_pydantic(
    onyx_product, 
    ProductOutput
)

# Convert Pydantic model to Onyx
pydantic_input = ProductCreateInput(**input_data)
onyx_product = OnyxSchemaIntegration.pydantic_to_onyx(
    pydantic_input, 
    OnyxProduct
)

# Validate Onyx model
validation_result = OnyxSchemaIntegration.validate_onyx_model(
    onyx_product, 
    "ProductOutput"
)
```

**Features:**
- Seamless conversion between Onyx and Pydantic models
- Validation of Onyx models using Pydantic schemas
- Error handling and logging
- Performance optimization

## Usage Patterns

### 1. Basic Input/Output Models

```python
# Input model with validation
class UserCreateInput(BaseInputModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8)
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.lower().strip()

# Output model with computed fields
class UserOutput(BaseOutputModel):
    id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    
    @computed_field
    @property
    def display_name(self) -> str:
        return self.username or f"User {self.id}"
```

### 2. Paginated Responses

```python
class ProductListOutput(PaginatedOutputModel[ProductOutput]):
    items: List[ProductOutput] = Field(default_factory=list)
    total_count: int = Field(default=0)
    pagination: PaginationInfo = Field(default_factory=PaginationInfo)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    
    def to_list_response(self, message: str = "Products retrieved successfully") -> ListResponse:
        return ListResponse(
            success=True,
            data=self.items,
            pagination=self.pagination,
            total_count=self.total_count,
            message=message,
            timestamp=self.timestamp,
            response_id=self.response_id
        )
```

### 3. Dynamic Schema Creation

```python
# Create schema dynamically
product_fields = {
    'name': (str, Field(..., min_length=1, max_length=100)),
    'price': (float, Field(..., gt=0)),
    'category': (str, Field(..., max_length=50)),
}

DynamicProductInput = SchemaFactory.create_input_schema(
    "DynamicProductInput",
    product_fields,
    description="Dynamically created product input schema"
)

# Use the dynamic schema
product_data = DynamicProductInput(
    name="Test Product",
    price=99.99,
    category="electronics"
)
```

### 4. Custom Validation

```python
# Business rule validation
def validate_user_age(value: Dict[str, Any], **kwargs) -> Union[bool, str]:
    """Validate user age is at least 13 years old."""
    if 'date_of_birth' in value:
        birth_date = value['date_of_birth']
        if isinstance(birth_date, str):
            birth_date = datetime.fromisoformat(birth_date).date()
        
        age = (date.today() - birth_date).days / 365.25
        if age < 13:
            return "User must be at least 13 years old"
    
    return True

# Register business rule validator
user_business_validator = BusinessRuleValidator([
    validate_user_age,
    validate_password_strength,
    validate_unique_email
])
validation_registry.register_validator("user_business", user_business_validator)
```

### 5. API Endpoint with Full Integration

```python
@router.post("/users/", response_model=SuccessResponse[UserOutput])
@schema_endpoint(
    input_model=UserCreateInput,
    output_model=UserOutput,
    validate_input=True,
    validate_output=True
)
@validate_business_rules("user_business")
async def create_user(user_data: UserCreateInput):
    """Create a new user with full validation."""
    try:
        # Input is automatically validated
        # Business rules are automatically validated
        
        # Create user (mock implementation)
        user = UserOutput(
            id=str(uuid.uuid4()),
            username=user_data.username,
            email=user_data.email,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Output is automatically validated and formatted
        return SchemaResponseHandler.success_response(
            data=user.model_dump(),
            message="User created successfully",
            status_code=201
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )
```

## Configuration

### Global Schema Configuration

```python
class SchemaConfig:
    """Global schema configuration."""
    validation_level: ValidationLevel = ValidationLevel.NORMAL
    enable_performance_monitoring: bool = True
    enable_audit_logging: bool = True
    max_field_length: int = 10000
    max_nested_depth: int = 10
    allowed_file_types: List[str] = ["jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"]
    max_file_size_mb: int = 50
```

### Validation Levels

```python
class ValidationLevel(str, Enum):
    """Validation levels for different environments."""
    STRICT = "strict"    # Maximum validation
    NORMAL = "normal"    # Standard validation
    LENIENT = "lenient"  # Minimal validation
```

## Performance Features

### 1. Performance Monitoring

```python
# Automatic performance tracking
class OptimizedBaseModel(BaseModel):
    _metrics: ClassVar[PydanticMetrics] = _metrics
    
    def __init__(self, **data: Any) -> None:
        start_time = time.perf_counter()
        try:
            super().__init__(**data)
            self._metrics.record_instance(self.__class__.__name__)
        except Exception as e:
            self._metrics.record_error(self.__class__.__name__, type(e).__name__)
            raise
```

### 2. Caching

```python
class CachedOptimizedModel(OptimizedBaseModel):
    """Optimized model with built-in caching."""
    
    @classmethod
    @lru_cache(maxsize=128)
    def _cached_validation(cls, value: str) -> str:
        """Cache expensive validation operations."""
        return expensive_validation_function(value)
```

### 3. ORJSON Integration

```python
model_config = ConfigDict(
    json_loads=orjson.loads,
    json_dumps=lambda v, *, default: orjson.dumps(v, default=default).decode(),
)
```

## Error Handling

### 1. Validation Errors

```python
try:
    user_input = UserCreateInput(**input_data)
except ValidationError as e:
    # Detailed error information
    for error in e.errors():
        field = error['loc'][0]
        message = error['msg']
        logger.warning(f"Validation error in field '{field}': {message}")
```

### 2. Business Rule Errors

```python
# Business rule validation
business_errors = input_model.validate_business_rules()
if business_errors:
    raise ValidationError(f"Business validation failed: {', '.join(business_errors)}")
```

### 3. Standardized Error Responses

```python
def error_response(error: str, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    return ErrorResponse(
        success=False,
        error=error,
        status_code=status_code,
        timestamp=datetime.utcnow(),
        details=details or {}
    )
```

## Monitoring and Observability

### 1. Performance Metrics

```python
# Get performance statistics
stats = OptimizedBaseModel.get_performance_stats()
print(f"Average validation time: {stats['average_validation_time_ms']:.2f}ms")
print(f"Error rate: {stats['overall_error_rate']:.2%}")
```

### 2. Validation Statistics

```python
# Get validator statistics
validator_stats = validation_registry.get_stats()
for validator_name, stats in validator_stats.items():
    print(f"{validator_name}: {stats['success_rate']:.2%} success rate")
```

### 3. Audit Logging

```python
# Automatic audit logging
logger.info(
    "Input validation completed",
    model=input_model.__class__.__name__,
    request_id=getattr(input_model, 'request_id', None),
    sanitized_data=input_model.sanitize()
)
```

## Best Practices

### 1. Schema Design

- Use descriptive field names and descriptions
- Implement appropriate validation rules
- Add computed fields for derived data
- Use enums for constrained values
- Implement business rule validation

### 2. Performance Optimization

- Use ORJSON for serialization
- Implement caching for expensive operations
- Monitor validation performance
- Use appropriate validation levels
- Optimize field validation order

### 3. Error Handling

- Provide clear error messages
- Implement comprehensive validation
- Use standardized error responses
- Log validation errors for debugging
- Handle business rule violations

### 4. Integration

- Use decorators for automatic validation
- Implement consistent response formats
- Integrate with existing Onyx models
- Use dependency injection
- Monitor integration performance

## Migration Guide

### From Standard Pydantic Models

```python
# Before
class User(BaseModel):
    name: str
    email: str

# After
class UserInput(BaseInputModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., description="Email address")
    
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.lower().strip()

class UserOutput(BaseOutputModel):
    id: str = Field(..., description="User ID")
    name: str = Field(..., description="User name")
    email: str = Field(..., description="Email address")
    
    @computed_field
    @property
    def display_name(self) -> str:
        return self.name or f"User {self.id}"
```

### From Onyx Models

```python
# Before
onyx_user = OnyxUser.get_by_id(user_id)
return {"id": onyx_user.id, "name": onyx_user.name}

# After
onyx_user = OnyxUser.get_by_id(user_id)
user_output = OnyxSchemaIntegration.onyx_to_pydantic(onyx_user, UserOutput)
return SchemaResponseHandler.success_response(
    data=user_output.model_dump(),
    message="User retrieved successfully"
)
```

## Conclusion

The Pydantic Schema System provides a comprehensive, production-ready solution for input/output validation and response schemas. With its rich feature set, performance optimizations, and seamless integration capabilities, it enables developers to build robust, maintainable APIs with consistent validation and response patterns.

Key benefits:
- **Consistency**: Standardized validation and response patterns
- **Performance**: Optimized validation and serialization
- **Flexibility**: Dynamic schema creation and custom validation
- **Integration**: Seamless integration with FastAPI and Onyx
- **Observability**: Comprehensive monitoring and logging
- **Maintainability**: Clear patterns and best practices

This system serves as the foundation for all API development in the Blatam Academy backend, ensuring high quality, consistent, and performant data validation and response handling. 