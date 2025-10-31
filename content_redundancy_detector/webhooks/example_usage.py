"""
Example usage of high-performance webhooks
"""

import asyncio
from fastapi import FastAPI
from . import (
    webhooks_router,
    start_webhook_workers,
    stop_webhook_workers,
    setup_fastapi_optimization,
    run_optimized_server
)

# Example webhook handler
async def webhook_handler(event):
    """Process webhook event"""
    print(f"Processing webhook: {event.id} - {event.type}")
    # Your business logic here
    return {"status": "processed"}

# Create FastAPI app
app = FastAPI(title="High-Performance Webhooks")

# Setup optimizations
setup_fastapi_optimization(app)

# Include webhook router
app.include_router(webhooks_router)

# Startup event
@app.on_event("startup")
async def startup():
    await start_webhook_workers(webhook_handler)

# Shutdown event
@app.on_event("shutdown")
async def shutdown():
    await stop_webhook_workers()

if __name__ == "__main__":
    # Run with optimizations
    run_optimized_server(app)





