from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Iterable

import jwt
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple auth middleware with API key or JWT support."""

    def __init__(
        self,
        app,
        *,
        api_keys: set[str] | None = None,
        jwt_secret: str | None = None,
        jwt_algorithms: Iterable[str] = ("HS256",),
        exempt_paths: set[str] | None = None,
    ):
        super().__init__(app)
        self.api_keys = api_keys or set()
        self.jwt_secret = jwt_secret
        self.jwt_algorithms = tuple(jwt_algorithms)
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        authz = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("x-api-key")

        if api_key_header and api_key_header in self.api_keys:
            return await call_next(request)

        if authz.lower().startswith("bearer "):
            token = authz.split(" ", 1)[1].strip()

            if token in self.api_keys:
                return await call_next(request)

            if self.jwt_secret:
                try:
                    jwt.decode(token, self.jwt_secret, algorithms=list(self.jwt_algorithms))
                    return await call_next(request)
                except jwt.PyJWTError:
                    pass

        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid or missing credentials"},
            },
        )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter keyed by API token or client IP."""

    def __init__(self, app, *, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        key = self._identify_client(request)
        now = time.time()
        window_start = now - 60

        bucket = self._buckets[key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()

        if len(bucket) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded",
                        "limit": self.requests_per_minute,
                        "window_seconds": 60,
                    },
                },
                headers={"Retry-After": "60"},
            )

        bucket.append(now)
        return await call_next(request)

    @staticmethod
    def _identify_client(request: Request) -> str:
        authz = request.headers.get("Authorization", "")
        if authz.lower().startswith("bearer "):
            return f"token:{authz.split(' ', 1)[1].strip()}"

        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"

        return f"ip:{request.client.host if request.client else 'unknown'}"
