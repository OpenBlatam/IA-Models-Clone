"""
Base Validator Implementation

Ultra-specialized base validator with advanced features for
data validation, verification, and compliance checking.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import weakref
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ValidatorType(Enum):
    """Validator type enumeration"""
    DATA = "data"
    FORMAT = "format"
    CONTENT = "content"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    AI = "ai"
    API = "api"
    DATABASE = "database"


class ValidationLevel(Enum):
    """Validation level enumeration"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidatorConfig:
    """Validator configuration"""
    name: str
    validator_type: ValidatorType
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enabled: bool = True
    timeout: Optional[float] = None
    retry_count: int = 0
    strict_mode: bool = False
    allow_warnings: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationContext:
    """Validation context"""
    validator_name: str
    validation_level: ValidationLevel
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    data_size: int = 0
    rules_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationError:
    """Validation error"""
    field: str
    message: str
    code: str
    severity: ValidationResult
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Validation report"""
    is_valid: bool
    result: ValidationResult
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    score: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseValidator(ABC, Generic[T]):
    """Base validator with advanced features"""
    
    def __init__(self, config: ValidatorConfig):
        self.config = config
        self._enabled = config.enabled
        self._validation_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._warning_count = 0
        self._error_count = 0
        self._total_duration = 0.0
        self._callbacks: List[Callable] = []
        self._error_handlers: List[Callable] = []
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def _validate(self, data: T, context: ValidationContext) -> ValidationReport:
        """Validate data (override in subclasses)"""
        pass
    
    async def validate(self, data: T) -> ValidationReport:
        """Validate data with context"""
        if not self._enabled:
            return ValidationReport(
                is_valid=True,
                result=ValidationResult.VALID,
                message="Validator disabled"
            )
        
        context = ValidationContext(
            validator_name=self.config.name,
            validation_level=self.config.validation_level,
            start_time=datetime.utcnow(),
            data_size=self._get_data_size(data)
        )
        
        try:
            # Pre-validation hook
            await self._pre_validate(context)
            
            # Validate data
            report = await asyncio.wait_for(
                self._validate(data, context),
                timeout=self.config.timeout
            )
            
            # Post-validation hook
            await self._post_validate(context, report)
            
            return report
            
        except Exception as e:
            await self._handle_error(context, e)
            return ValidationReport(
                is_valid=False,
                result=ValidationResult.ERROR,
                errors=[ValidationError(
                    field="system",
                    message=str(e),
                    code="VALIDATION_ERROR",
                    severity=ValidationResult.ERROR
                )]
            )
    
    async def validate_batch(self, data_list: List[T]) -> List[ValidationReport]:
        """Validate multiple data items"""
        if not self._enabled:
            return [
                ValidationReport(
                    is_valid=True,
                    result=ValidationResult.VALID,
                    message="Validator disabled"
                )
                for _ in data_list
            ]
        
        # Process in batches
        reports = []
        for data in data_list:
            report = await self.validate(data)
            reports.append(report)
        
        return reports
    
    def validate_sync(self, data: T) -> ValidationReport:
        """Validate data synchronously"""
        if not self._enabled:
            return ValidationReport(
                is_valid=True,
                result=ValidationResult.VALID,
                message="Validator disabled"
            )
        
        # Run async validate in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.validate(data))
    
    def validate_batch_sync(self, data_list: List[T]) -> List[ValidationReport]:
        """Validate multiple data items synchronously"""
        if not self._enabled:
            return [
                ValidationReport(
                    is_valid=True,
                    result=ValidationResult.VALID,
                    message="Validator disabled"
                )
                for _ in data_list
            ]
        
        # Run async validate_batch in sync context
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.validate_batch(data_list))
    
    def _get_data_size(self, data: Any) -> int:
        """Get data size in bytes"""
        try:
            if hasattr(data, '__sizeof__'):
                return data.__sizeof__()
            elif isinstance(data, (str, bytes)):
                return len(data)
            elif isinstance(data, (list, tuple)):
                return sum(self._get_data_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(
                    self._get_data_size(key) + self._get_data_size(value)
                    for key, value in data.items()
                )
            else:
                return 1  # Default size
        except Exception:
            return 1  # Default size
    
    async def _pre_validate(self, context: ValidationContext) -> None:
        """Pre-validation hook (override in subclasses)"""
        pass
    
    async def _post_validate(self, context: ValidationContext, report: ValidationReport) -> None:
        """Post-validation hook (override in subclasses)"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Update metrics
        self._update_metrics(report)
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, report)
                else:
                    callback(context, report)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def _handle_error(self, context: ValidationContext, error: Exception) -> None:
        """Handle validation errors"""
        context.end_time = datetime.utcnow()
        context.duration = (context.end_time - context.start_time).total_seconds()
        
        # Call error handlers
        for handler in self._error_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context, error)
                else:
                    handler(context, error)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")
    
    def _update_metrics(self, report: ValidationReport) -> None:
        """Update validator metrics"""
        self._validation_count += 1
        
        if report.is_valid:
            self._valid_count += 1
        else:
            self._invalid_count += 1
        
        # Count by severity
        for error in report.errors:
            if error.severity == ValidationResult.WARNING:
                self._warning_count += 1
            elif error.severity == ValidationResult.ERROR:
                self._error_count += 1
            elif error.severity == ValidationResult.CRITICAL:
                self._error_count += 1
        
        if report.duration:
            self._total_duration += report.duration
    
    def add_callback(self, callback: Callable) -> None:
        """Add callback for events"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def add_error_handler(self, handler: Callable) -> None:
        """Add error handler"""
        self._error_handlers.append(handler)
    
    def remove_error_handler(self, handler: Callable) -> None:
        """Remove error handler"""
        if handler in self._error_handlers:
            self._error_handlers.remove(handler)
    
    def enable(self) -> None:
        """Enable validator"""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable validator"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if validator is enabled"""
        return self._enabled
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get validator metrics"""
        avg_duration = (
            self._total_duration / self._validation_count
            if self._validation_count > 0 else 0
        )
        
        validity_rate = (
            self._valid_count / self._validation_count
            if self._validation_count > 0 else 0
        )
        
        error_rate = (
            self._error_count / self._validation_count
            if self._validation_count > 0 else 0
        )
        
        return {
            "name": self.config.name,
            "type": self.config.validator_type.value,
            "level": self.config.validation_level.value,
            "enabled": self._enabled,
            "validation_count": self._validation_count,
            "valid_count": self._valid_count,
            "invalid_count": self._invalid_count,
            "warning_count": self._warning_count,
            "error_count": self._error_count,
            "validity_rate": validity_rate,
            "error_rate": error_rate,
            "total_duration": self._total_duration,
            "average_duration": avg_duration
        }
    
    def reset_metrics(self) -> None:
        """Reset validator metrics"""
        self._validation_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._warning_count = 0
        self._error_count = 0
        self._total_duration = 0.0
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', enabled={self._enabled})"


class ValidatorChain:
    """Chain of validators with priority ordering"""
    
    def __init__(self):
        self._validators: List[BaseValidator] = []
        self._lock = asyncio.Lock()
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add validator to chain"""
        self._validators.append(validator)
        # Sort by validation level (higher level first)
        self._validators.sort(
            key=lambda v: v.config.validation_level.value,
            reverse=True
        )
    
    def remove_validator(self, name: str) -> None:
        """Remove validator from chain"""
        self._validators = [
            v for v in self._validators
            if v.config.name != name
        ]
    
    async def validate(self, data: Any) -> ValidationReport:
        """Validate data through validator chain"""
        combined_report = ValidationReport(
            is_valid=True,
            result=ValidationResult.VALID,
            score=100.0
        )
        
        for validator in self._validators:
            if validator.is_enabled():
                try:
                    report = await validator.validate(data)
                    
                    # Combine reports
                    if not report.is_valid:
                        combined_report.is_valid = False
                        combined_report.result = report.result
                    
                    combined_report.errors.extend(report.errors)
                    combined_report.warnings.extend(report.warnings)
                    combined_report.score = min(combined_report.score, report.score)
                    combined_report.duration += report.duration
                    
                    # Stop on critical errors if strict mode
                    if (report.result == ValidationResult.CRITICAL and 
                        validator.config.strict_mode):
                        break
                        
                except Exception as e:
                    logger.error(f"Error in validator '{validator.config.name}': {e}")
                    # Continue to next validator or re-raise based on configuration
                    raise
        
        return combined_report
    
    async def validate_batch(self, data_list: List[Any]) -> List[ValidationReport]:
        """Validate multiple data items through validator chain"""
        reports = []
        for data in data_list:
            report = await self.validate(data)
            reports.append(report)
        return reports
    
    def get_validators(self) -> List[BaseValidator]:
        """Get all validators"""
        return self._validators.copy()
    
    def get_validators_by_type(self, validator_type: ValidatorType) -> List[BaseValidator]:
        """Get validators by type"""
        return [
            v for v in self._validators
            if v.config.validator_type == validator_type
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all validators"""
        return {
            validator.config.name: validator.get_metrics()
            for validator in self._validators
        }


class ValidatorRegistry:
    """Registry for managing validators"""
    
    def __init__(self):
        self._validators: Dict[str, BaseValidator] = {}
        self._chains: Dict[str, ValidatorChain] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, validator: BaseValidator) -> None:
        """Register validator"""
        async with self._lock:
            self._validators[validator.config.name] = validator
            logger.info(f"Registered validator: {validator.config.name}")
    
    async def unregister(self, name: str) -> None:
        """Unregister validator"""
        async with self._lock:
            if name in self._validators:
                del self._validators[name]
                logger.info(f"Unregistered validator: {name}")
    
    def get(self, name: str) -> Optional[BaseValidator]:
        """Get validator by name"""
        return self._validators.get(name)
    
    def get_by_type(self, validator_type: ValidatorType) -> List[BaseValidator]:
        """Get validators by type"""
        return [
            validator for validator in self._validators.values()
            if validator.config.validator_type == validator_type
        ]
    
    def create_chain(self, name: str, validator_names: List[str]) -> ValidatorChain:
        """Create validator chain"""
        chain = ValidatorChain()
        
        for validator_name in validator_names:
            validator = self.get(validator_name)
            if validator:
                chain.add_validator(validator)
        
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[ValidatatorChain]:
        """Get validator chain"""
        return self._chains.get(name)
    
    def list_all(self) -> List[BaseValidator]:
        """List all validators"""
        return list(self._validators.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all validators"""
        return {
            name: validator.get_metrics()
            for name, validator in self._validators.items()
        }


# Global validator registry
validator_registry = ValidatorRegistry()


# Convenience functions
async def register_validator(validator: BaseValidator):
    """Register validator"""
    await validator_registry.register(validator)


def get_validator(name: str) -> Optional[BaseValidator]:
    """Get validator by name"""
    return validator_registry.get(name)


def create_validator_chain(name: str, validator_names: List[str]) -> ValidatorChain:
    """Create validator chain"""
    return validator_registry.create_chain(name, validator_names)


# Validator factory functions
def create_validator(validator_type: ValidatorType, name: str, **kwargs) -> BaseValidator:
    """Create validator by type"""
    config = ValidatorConfig(
        name=name,
        validator_type=validator_type,
        **kwargs
    )
    
    # This would be implemented with specific validator classes
    # For now, return a placeholder
    raise NotImplementedError(f"Validator type {validator_type} not implemented yet")


# Common validator combinations
def data_validation_chain(name: str = "data_validation") -> ValidatorChain:
    """Create data validation validator chain"""
    return create_validator_chain(name, [
        "format_validation",
        "schema_validation",
        "type_validation",
        "business_validation"
    ])


def security_validation_chain(name: str = "security_validation") -> ValidatorChain:
    """Create security validation validator chain"""
    return create_validator_chain(name, [
        "input_validation",
        "security_validation",
        "compliance_validation",
        "audit_validation"
    ])


def ai_validation_chain(name: str = "ai_validation") -> ValidatorChain:
    """Create AI validation validator chain"""
    return create_validator_chain(name, [
        "model_validation",
        "inference_validation",
        "output_validation",
        "performance_validation"
    ])


def api_validation_chain(name: str = "api_validation") -> ValidatorChain:
    """Create API validation validator chain"""
    return create_validator_chain(name, [
        "request_validation",
        "response_validation",
        "format_validation",
        "security_validation"
    ])





















