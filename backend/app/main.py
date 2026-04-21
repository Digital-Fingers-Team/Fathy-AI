from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import init_db
from app.routes.auth import router as auth_router
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.memory import router as memory_router
from app.routes.teach import router as teach_router
from app.serving import AuthMiddleware, RateLimitMiddleware, create_router as create_serving_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("Starting up", extra={"app_name": settings.app_name, "env": settings.environment})
    init_db(settings.database_url)
    logger.info("Database initialized")

    # Initialize long-lived services (no background "self-learning" claims; this is a wrapper/client init).
    from app.services.ai_service import AIService

    app.state.settings = settings
    app.state.ai_service = AIService(settings)

    yield

    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    if not settings.secret_key or settings.secret_key == "changeme":
        raise RuntimeError("Invalid SECRET_KEY: set a non-default secret before starting the app.")

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    logger = get_logger("app.http")

    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        start = perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed_ms = int((perf_counter() - start) * 1000)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": getattr(locals().get("response", None), "status_code", None),
                    "elapsed_ms": elapsed_ms,
                },
            )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception", extra={"path": request.url.path, "method": request.method})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    cors_origins = settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, tags=["health"])
    app.include_router(auth_router)
    app.include_router(chat_router, prefix="", tags=["chat"])
    app.include_router(teach_router, prefix="", tags=["teach"])
    app.include_router(memory_router, prefix="", tags=["memory"])

    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.serving_rate_limit_rpm)
    if settings.serving_auth_enabled:
        app.add_middleware(
            AuthMiddleware,
            api_keys=settings.serving_api_keys,
            jwt_secret=settings.serving_jwt_secret,
            exempt_paths={"/health", "/docs", "/openapi.json"},
        )

    app.include_router(create_serving_router(), prefix="", tags=["serving"])

    return app


app = create_app()
