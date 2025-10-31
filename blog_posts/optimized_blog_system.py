from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from .router import router
from .middlewares import performance_middleware, error_handling_middleware, logging_and_timing_middleware

from typing import Any, List, Dict, Optional
import asyncio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blog_system")

app = FastAPI()
app.include_router(router, prefix="/blog")
app.middleware("http")(performance_middleware)
app.middleware("http")(error_handling_middleware)
app.middleware("http")(logging_and_timing_middleware)

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc) -> Any:
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc) -> Any:
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    ) 