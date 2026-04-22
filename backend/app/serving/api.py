from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import Settings
from app.services.ai_service import AIService, HistoryMessage

from .schemas import (
    AnthropicMessageResponse,
    AnthropicMessagesRequest,
    AnthropicTextBlock,
    AnthropicUsage,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatMessage,
    OpenAIChoice,
    OpenAIUsage,
)


class InferenceBackend(Protocol):
    def generate(self, request: InferenceRequest) -> InferenceResponse: ...


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    context_window: int
    supports_streaming: bool = True


class FathyInferenceBackend:
    """Connects the serving layer to the real Fathy AI service."""

    def __init__(self, settings: Settings):
        self._ai = AIService(settings)

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        history: list[HistoryMessage] = []
        last_user_message = ""

        for msg in request.messages:
            if msg.role not in {"user", "assistant"}:
                continue
            history.append(HistoryMessage(role=msg.role, content=msg.content))
            if msg.role == "user":
                last_user_message = msg.content

        if not last_user_message:
            last_user_message = request.messages[-1].content if request.messages else ""

        prior_history = history[:-1] if history and history[-1].role == "user" else history
        result = self._ai.answer(last_user_message, memories=[], history=prior_history)

        return InferenceResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            model=request.model,
            output_text=result.answer,
            input_tokens=max(1, len(last_user_message.split())),
            output_tokens=max(1, len(result.answer.split())),
        )


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "claude-3-5-sonnet": ModelSpec(
        id="claude-3-5-sonnet",
        provider="anthropic",
        context_window=200_000,
        supports_streaming=True,
    ),
    "gpt-4o-mini": ModelSpec(
        id="gpt-4o-mini",
        provider="openai",
        context_window=128_000,
        supports_streaming=True,
    ),
}


def _error(status_code: int, err_type: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": err_type, "message": message}},
    )


def _ensure_model_available(model: str) -> ModelSpec:
    spec = MODEL_REGISTRY.get(model)
    if not spec:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "invalid_request_error",
            f"Model '{model}' not found",
        )
    return spec


def _anthropic_to_inference(payload: AnthropicMessagesRequest) -> InferenceRequest:
    return InferenceRequest(
        model=payload.model,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        messages=[
            InferenceMessage(role=("assistant" if m.role == "assistant" else "user" if m.role == "user" else "system"), content=m.content)
            for m in payload.messages
        ],
    )


def _openai_to_inference(payload: OpenAIChatCompletionRequest) -> InferenceRequest:
    msgs: list[InferenceMessage] = []
    for msg in payload.messages:
        if msg.role not in {"system", "user", "assistant"}:
            continue
        msgs.append(InferenceMessage(role=msg.role, content=msg.content or ""))

    return InferenceRequest(
        model=payload.model,
        max_tokens=payload.max_tokens or 1024,
        temperature=payload.temperature or 1.0,
        messages=msgs,
    )


def _inference_to_anthropic(response: InferenceResponse) -> AnthropicMessageResponse:
    return AnthropicMessageResponse(
        id=response.id,
        model=response.model,
        content=[AnthropicTextBlock(text=response.output_text)],
        usage=AnthropicUsage(input_tokens=response.input_tokens, output_tokens=response.output_tokens),
    )


def _inference_to_openai(response: InferenceResponse) -> OpenAIChatCompletionResponse:
    prompt_tokens = response.input_tokens
    completion_tokens = response.output_tokens
    return OpenAIChatCompletionResponse(
        id=response.id,
        created=int(time.time()),
        model=response.model,
        choices=[
            OpenAIChoice(
                index=0,
                finish_reason="stop",
                message=OpenAIChatMessage(role="assistant", content=response.output_text),
            )
        ],
        usage=OpenAIUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def create_router(backend: InferenceBackend | None = None) -> APIRouter:
    if backend is None:
        raise RuntimeError("Serving backend must be provided")
    inference_backend = backend
    router = APIRouter(prefix="/v1", tags=["serving"])

    @router.post("/messages", response_model=AnthropicMessageResponse)
    def create_message(payload: AnthropicMessagesRequest):
        _ensure_model_available(payload.model)
        if not payload.messages:
            raise _error(status.HTTP_400_BAD_REQUEST, "invalid_request_error", "messages cannot be empty")

        inf_request = _anthropic_to_inference(payload)
        inf_response = inference_backend.generate(inf_request)
        return _inference_to_anthropic(inf_response)

    @router.post("/messages/stream")
    def create_message_stream(payload: AnthropicMessagesRequest):
        spec = _ensure_model_available(payload.model)
        if not spec.supports_streaming:
            raise _error(status.HTTP_400_BAD_REQUEST, "invalid_request_error", "model does not support streaming")

        inf_request = _anthropic_to_inference(payload)
        inf_response = inference_backend.generate(inf_request)
        text = inf_response.output_text

        def _event(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def event_stream():
            yield _event("message_start", {"type": "message_start", "message": {"id": inf_response.id, "model": inf_response.model}})

            for token in text.split():
                yield _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": f"{token} "},
                    },
                )

            yield _event(
                "message_stop",
                {
                    "type": "message_stop",
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": inf_response.input_tokens,
                        "output_tokens": inf_response.output_tokens,
                    },
                },
            )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/chat/completions", response_model=OpenAIChatCompletionResponse)
    def create_chat_completion(payload: OpenAIChatCompletionRequest):
        _ensure_model_available(payload.model)
        inf_request = _openai_to_inference(payload)
        if not inf_request.messages:
            raise _error(status.HTTP_400_BAD_REQUEST, "invalid_request_error", "no supported messages found")
        inf_response = inference_backend.generate(inf_request)
        return _inference_to_openai(inf_response)

    @router.get("/models")
    def list_models():
        now = int(time.time())
        data = [
            {
                "id": model.id,
                "object": "model",
                "created": now,
                "owned_by": model.provider,
                "context_window": model.context_window,
                "supports_streaming": model.supports_streaming,
            }
            for model in MODEL_REGISTRY.values()
        ]
        return {"object": "list", "data": data}

    return router
