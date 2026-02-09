"""
ML NLP Benchmark Application for AI Document Processor
Real, working ML NLP Benchmark FastAPI application
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import logging
import time
import asyncio
from datetime import datetime
import uvicorn
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ML NLP Benchmark system
from ml_nlp_benchmark import ml_nlp_benchmark_system

# Import route modules
from ml_nlp_benchmark_routes import router as ml_nlp_benchmark_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_nlp_benchmark_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ML NLP Benchmark API",
    description="Real, working ML NLP Benchmark system for AI document processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Timing middleware
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Include ML NLP Benchmark router
app.include_router(ml_nlp_benchmark_router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "ML NLP Benchmark API",
        "version": "1.0.0",
        "description": "Real, working ML NLP Benchmark system for AI document processing",
        "features": [
            "NLP Analysis",
            "ML Analysis", 
            "Benchmark Analysis",
            "Comprehensive Analysis"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "stats": "/stats"
        },
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check for ML NLP Benchmark system"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Check ML NLP Benchmark system
        try:
            ml_nlp_benchmark_stats = ml_nlp_benchmark_system.get_ml_nlp_benchmark_stats()
            health_status["systems"]["ml_nlp_benchmark"] = {
                "status": "healthy",
                "uptime": ml_nlp_benchmark_stats.get("uptime_seconds", 0),
                "requests": ml_nlp_benchmark_stats.get("stats", {}).get("total_benchmark_requests", 0)
            }
        except Exception as e:
            health_status["systems"]["ml_nlp_benchmark"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall health
        unhealthy_systems = [name for name, system in health_status["systems"].items() if system["status"] == "unhealthy"]
        if unhealthy_systems:
            health_status["status"] = "degraded"
            health_status["unhealthy_systems"] = unhealthy_systems
        else:
            health_status["status"] = "healthy"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Statistics endpoint
@app.get("/stats")
async def get_comprehensive_stats():
    """Get comprehensive statistics from ML NLP Benchmark system"""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "systems": {}
        }
        
        # Get ML NLP Benchmark stats
        try:
            ml_nlp_benchmark_stats = ml_nlp_benchmark_system.get_ml_nlp_benchmark_stats()
            stats["systems"]["ml_nlp_benchmark"] = ml_nlp_benchmark_stats
        except Exception as e:
            stats["systems"]["ml_nlp_benchmark"] = {"error": str(e)}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting comprehensive stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Comparison endpoint
@app.get("/compare")
async def compare_ml_nlp_benchmark_systems():
    """Compare performance across ML NLP Benchmark systems"""
    try:
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "comparison": {}
        }
        
        # Get stats from ML NLP Benchmark system
        systems_stats = {}
        
        try:
            systems_stats["ml_nlp_benchmark"] = ml_nlp_benchmark_system.get_ml_nlp_benchmark_stats()
        except:
            systems_stats["ml_nlp_benchmark"] = None
        
        # Compare performance metrics
        performance_comparison = {}
        for system_name, stats in systems_stats.items():
            if stats:
                performance_comparison[system_name] = {
                    "average_processing_time": stats.get("average_processing_time", 0),
                    "throughput_per_second": stats.get("throughput_per_second", 0),
                    "success_rate": stats.get("success_rate", 0),
                    "total_requests": stats.get("stats", {}).get("total_benchmark_requests", 0)
                }
        
        comparison["performance_comparison"] = performance_comparison
        
        # Find best performing system
        if performance_comparison:
            best_throughput = max(performance_comparison.items(), key=lambda x: x[1]["throughput_per_second"])
            best_success_rate = max(performance_comparison.items(), key=lambda x: x[1]["success_rate"])
            fastest_processing = min(performance_comparison.items(), key=lambda x: x[1]["average_processing_time"])
            
            comparison["best_performance"] = {
                "highest_throughput": {
                    "system": best_throughput[0],
                    "throughput": best_throughput[1]["throughput_per_second"]
                },
                "highest_success_rate": {
                    "system": best_success_rate[0],
                    "success_rate": best_success_rate[1]["success_rate"]
                },
                "fastest_processing": {
                    "system": fastest_processing[0],
                    "processing_time": fastest_processing[1]["average_processing_time"]
                }
            }
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error in comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System information endpoint
@app.get("/info")
async def get_system_info():
    """Get comprehensive system information"""
    return {
        "system": "ML NLP Benchmark API",
        "version": "1.0.0",
        "description": "Real, working ML NLP Benchmark system for AI document processing",
        "features": [
            "NLP Analysis - Natural Language Processing with comprehensive features",
            "ML Analysis - Machine Learning analysis with advanced algorithms",
            "Benchmark Analysis - Performance benchmarking and evaluation",
            "Comprehensive Analysis - Complete text analysis with all features"
        ],
        "capabilities": {
            "text_analysis": "Comprehensive text analysis with NLP and ML capabilities",
            "batch_processing": "High-performance batch processing capabilities",
            "real_time_processing": "Real-time text analysis with streaming support",
            "performance_optimization": "Performance optimizations including caching, compression, quantization",
            "scalability": "Horizontal and vertical scaling capabilities",
            "monitoring": "Comprehensive monitoring and statistics",
            "health_checks": "System health monitoring and diagnostics"
        },
        "endpoints": {
            "ml_nlp_benchmark": "/api/v1/ml-nlp-benchmark"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        }
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "ml_nlp_benchmark_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )












