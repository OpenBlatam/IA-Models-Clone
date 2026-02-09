"""Error Handling Middleware"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from ...domain.exceptions import QualityControlException

async def error_handler_middleware(request: Request, call_next):
    """Global error handler middleware."""
    try:
        response = await call_next(request)
        return response
    except QualityControlException as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=e.to_dict(),
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": str(e),
            },
        )



