from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# -----------------------------
# Anthropic-style /v1/messages
# -----------------------------


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AnthropicMessagesRequest(BaseModel):
    model: str
    messages: list[AnthropicMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False
    metadata: dict[str, Any] | None = None


class AnthropicTextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicTextBlock]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence"] = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# -------------------------------------
# OpenAI-compatible /v1/chat/completions
# -------------------------------------


class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage] = Field(default_factory=list)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = False

    @model_validator(mode="after")
    def validate_messages(self) -> "OpenAIChatCompletionRequest":
        if not self.messages:
            raise ValueError("messages must contain at least one item")
        return self


class OpenAIChoice(BaseModel):
    index: int
    finish_reason: Literal["stop", "length"] = "stop"
    message: OpenAIChatMessage


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)


# -----------------------------
# Shared internal inferencing
# -----------------------------


class InferenceMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InferenceRequest(BaseModel):
    model: str
    messages: list[InferenceMessage]
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


class InferenceResponse(BaseModel):
    id: str
    model: str
    output_text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ErrorBody(BaseModel):
    type: str = "error"
    error: dict[str, Any]
