from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.core.logger import get_logger

logger = get_logger(__name__)

def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Extracts and formats the first validation error into a readable message."""
    if not isinstance(exc, RequestValidationError):
        raise exc

    errors = exc.errors()
    if not errors:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Invalid input data"}
        )
    
    # Grab the first error in the list
    first_error = errors[0]
    logger.debug(f"Validation error details: {errors}")
    clean_message = first_error.get("msg", "Invalid input data")
    
    # Strip messy prefixes like Pydantic's default "Value error, "
    clean_message = clean_message.replace("Value error, ", "")
    
    # Capitalise for clean UI display
    clean_message = clean_message[0].upper() + clean_message[1:]

    logger.warning(f"Validation failed on {request.url.path}: {clean_message}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": clean_message}
    )

def setup_exception_handlers(app: FastAPI) -> None:
    """Registers custom exception handlers into the FastAPI app instance."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)