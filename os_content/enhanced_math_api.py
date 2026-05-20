from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Union, List, Optional, Dict, Any
import asyncio
import logging
import time
import json
from datetime import datetime
from refactored_math_system import (
    import uvicorn
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Enhanced Math API for OS Content
FastAPI integration with advanced mathematical operations.
"""


    MathService, create_math_service, OperationType, 
    CalculationMethod, MathOperation, MathResult
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Math API",
    description="Advanced mathematical operations with multiple calculation methods",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global math service instance
math_service: Optional[MathService] = None


# Pydantic models for request/response
class MathRequest(BaseModel):
    """Request model for mathematical operations."""
    operation: str = Field(..., description="Operation type (add, multiply, divide, power, sqrt, log)")
    operands: List[Union[int, float]] = Field(..., description="List of operands")
    method: str = Field(default="basic", description="Calculation method (basic, numpy, math, decimal)")
    precision: int = Field(default=10, description="Precision for decimal calculations")
    
    @validator('operation')
    def validate_operation(cls, v) -> bool:
        valid_operations = ['add', 'multiply', 'divide', 'power', 'sqrt', 'log']
        if v not in valid_operations:
            raise ValueError(f'Operation must be one of: {valid_operations}')
        return v
    
    @validator('method')
    def validate_method(cls, v) -> bool:
        valid_methods = ['basic', 'numpy', 'math', 'decimal']
        if v not in valid_methods:
            raise ValueError(f'Method must be one of: {valid_methods}')
        return v


class MathResponse(BaseModel):
    """Response model for mathematical operations."""
    result: Union[int, float, List[Union[int, float]]]
    operation: str
    method: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchMathRequest(BaseModel):
    """Request model for batch operations."""
    operations: List[MathRequest]
    parallel: bool = Field(default=True, description="Execute operations in parallel")


class PerformanceStats(BaseModel):
    """Performance statistics model."""
    total_operations: int
    cache_hits: int
    average_execution_time: float
    uptime: float
    active_connections: int


# Dependency injection
async def get_math_service() -> MathService:
    """Get math service instance."""
    global math_service
    if math_service is None:
        math_service = create_math_service(max_workers=8, cache_size=2000)
    return math_service


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global math_service
    math_service = create_math_service(max_workers=8, cache_size=2000)
    logger.info("Enhanced Math API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global math_service
    if math_service:
        await math_service.processor.shutdown()
    logger.info("Enhanced Math API shutdown complete")


# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Enhanced Math API",
        "version": "1.0.0"
    }


# Basic math operation endpoints
@app.post("/math/add", response_model=MathResponse)
async def add_numbers(
    request: MathRequest,
    background_tasks: BackgroundTasks,
    math_service: MathService = Depends(get_math_service)
):
    """Add multiple numbers."""
    try:
        operation = MathOperation(
            operation_type=OperationType.ADD,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="add",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error in add operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math/multiply", response_model=MathResponse)
async def multiply_numbers(
    request: MathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Multiply multiple numbers."""
    try:
        operation = MathOperation(
            operation_type=OperationType.MULTIPLY,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="multiply",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error in multiply operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math/divide", response_model=MathResponse)
async def divide_numbers(
    request: MathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Divide first number by second number."""
    if len(request.operands) != 2:
        raise HTTPException(status_code=400, detail="Division requires exactly 2 operands")
    
    try:
        operation = MathOperation(
            operation_type=OperationType.DIVIDE,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="divide",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero")
    except Exception as e:
        logger.error(f"Error in divide operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math/power", response_model=MathResponse)
async def power_operation(
    request: MathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Raise first number to the power of second number."""
    if len(request.operands) != 2:
        raise HTTPException(status_code=400, detail="Power operation requires exactly 2 operands")
    
    try:
        operation = MathOperation(
            operation_type=OperationType.POWER,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="power",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error in power operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch operations endpoint
@app.post("/math/batch", response_model=List[MathResponse])
async def batch_operations(
    request: BatchMathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Execute multiple operations in batch."""
    try:
        results = []
        
        if request.parallel:
            # Execute operations in parallel
            tasks = []
            for op_request in request.operations:
                operation = MathOperation(
                    operation_type=OperationType(op_request.operation.upper()),
                    operands=op_request.operands,
                    method=CalculationMethod(op_request.method)
                )
                tasks.append(math_service.processor.process_operation(operation))
            
            batch_results = await asyncio.gather(*tasks)
            
            for i, result in enumerate(batch_results):
                op_request = request.operations[i]
                results.append(MathResponse(
                    result=result.value,
                    operation=op_request.operation,
                    method=op_request.method,
                    execution_time=result.execution_time,
                    success=result.success,
                    error_message=result.error_message,
                    timestamp=datetime.now(),
                    metadata=result.metadata
                ))
        else:
            # Execute operations sequentially
            for op_request in request.operations:
                operation = MathOperation(
                    operation_type=OperationType(op_request.operation.upper()),
                    operands=op_request.operands,
                    method=CalculationMethod(op_request.method)
                )
                
                result = await math_service.processor.process_operation(operation)
                
                results.append(MathResponse(
                    result=result.value,
                    operation=op_request.operation,
                    method=op_request.method,
                    execution_time=result.execution_time,
                    success=result.success,
                    error_message=result.error_message,
                    timestamp=datetime.now(),
                    metadata=result.metadata
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch operations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.get("/stats", response_model=PerformanceStats)
async def get_performance_stats(
    math_service: MathService = Depends(get_math_service)
):
    """Get performance statistics."""
    try:
        stats = math_service.get_stats()
        return PerformanceStats(
            total_operations=stats.get("total_operations", 0),
            cache_hits=stats.get("cache_hits", 0),
            average_execution_time=stats.get("average_execution_time", 0.0),
            uptime=time.time() - app.startup_time if hasattr(app, 'startup_time') else 0.0,
            active_connections=1  # Simplified for demo
        )
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/clear")
async def clear_cache(
    math_service: MathService = Depends(get_math_service)
):
    """Clear operation cache."""
    try:
        math_service.processor.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Advanced mathematical functions
@app.post("/math/sqrt", response_model=MathResponse)
async def square_root(
    request: MathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Calculate square root."""
    if len(request.operands) != 1:
        raise HTTPException(status_code=400, detail="Square root requires exactly 1 operand")
    
    if request.operands[0] < 0:
        raise HTTPException(status_code=400, detail="Cannot calculate square root of negative number")
    
    try:
        operation = MathOperation(
            operation_type=OperationType.SQRT,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="sqrt",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error in square root operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/math/log", response_model=MathResponse)
async def logarithm(
    request: MathRequest,
    math_service: MathService = Depends(get_math_service)
):
    """Calculate logarithm."""
    if len(request.operands) < 1 or len(request.operands) > 2:
        raise HTTPException(status_code=400, detail="Logarithm requires 1 or 2 operands")
    
    if request.operands[0] <= 0:
        raise HTTPException(status_code=400, detail="Cannot calculate logarithm of non-positive number")
    
    try:
        operation = MathOperation(
            operation_type=OperationType.LOG,
            operands=request.operands,
            method=CalculationMethod(request.method)
        )
        
        result = await math_service.processor.process_operation(operation)
        
        return MathResponse(
            result=result.value,
            operation="log",
            method=request.method,
            execution_time=result.execution_time,
            success=result.success,
            error_message=result.error_message,
            timestamp=datetime.now(),
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error in logarithm operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc) -> Any:
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Example usage and testing
if __name__ == "__main__":
    
    # Set startup time for uptime calculation
    app.startup_time = time.time()
    
    # Run the API server
    uvicorn.run(
        "enhanced_math_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 