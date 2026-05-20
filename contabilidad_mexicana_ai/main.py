"""
Main entry point for Contabilidad Mexicana AI API
"""

import uvicorn
from fastapi import FastAPI
from api.contador_api import router
from api.middleware import RateLimitMiddleware, RequestLoggingMiddleware

app = FastAPI(
    title="Contabilidad Mexicana AI",
    description="Sistema de IA para resolver problemas contables y fiscales mexicanos",
    version="1.0.0"
)

# Add middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=60, requests_per_hour=1000)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Contabilidad Mexicana AI",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
