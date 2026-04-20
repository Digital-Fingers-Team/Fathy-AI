from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from .sampling import sample_next_token
from .streaming import TokenChunk, format_sse_chunk, format_sse_done


@dataclass(slots=True)
class QueuedGenerationRequest:
    prompt: str
    kwargs: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]


class FathyInferenceEngine:
    """Inference runtime for ``FathyForCausalLM`` generation and streaming."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str | torch.device | None = None,
        enable_continuous_batching: bool = False,
        max_batch_size: int = 8,
        queue_timeout_s: float = 0.01,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or getattr(model, "device", "cpu"))
        self.enable_continuous_batching = enable_continuous_batching
        self.max_batch_size = max_batch_size
        self.queue_timeout_s = queue_timeout_s

        self._queue: queue.Queue[QueuedGenerationRequest] | None = None
        self._worker_thread: threading.Thread | None = None

        if self.enable_continuous_batching:
            self._start_worker()

    @classmethod
    def from_model_loader_hooks(
        cls,
        *,
        model_loader,
        model_name_or_path: str,
        tokenizer,
        precision: str = "bf16",
        **engine_kwargs: Any,
    ) -> "FathyInferenceEngine":
        """Construct from model loader hooks for bf16/int8/int4 loading paths."""
        precision = precision.lower()
        if precision not in {"bf16", "int8", "int4"}:
            raise ValueError("precision must be one of {'bf16', 'int8', 'int4'}")

        load_kwargs: dict[str, Any] = {}
        if precision == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif precision == "int8":
            load_kwargs["load_in_8bit"] = True
        elif precision == "int4":
            load_kwargs["load_in_4bit"] = True

        model = model_loader(model_name_or_path, **load_kwargs)
        return cls(model, tokenizer, **engine_kwargs)

    def _tokenize_batch(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer.pad_token is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return {k: v.to(self.device) for k, v in model_inputs.items()}

    @staticmethod
    def _build_generator(seed: int | None, device: torch.device) -> torch.Generator | None:
        if seed is None:
            return None
        return torch.Generator(device=device.type).manual_seed(seed)

    def generate(
        self,
        prompt: str | list[str],
        *,
        max_new_tokens: int = 128,
        deterministic: bool = False,
        seed: int | None = None,
        do_sample: bool | None = None,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Run generation via ``FathyForCausalLM.generate`` with dynamic padding."""
        prompts = [prompt] if isinstance(prompt, str) else prompt
        inputs = self._tokenize_batch(prompts)

        if deterministic:
            do_sample = False
            generate_kwargs.setdefault("num_beams", 1)

        if do_sample is not None:
            generate_kwargs["do_sample"] = do_sample

        generator = self._build_generator(seed, self.device)
        if generator is not None:
            generate_kwargs["generator"] = generator

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return {"sequences": output_ids, "texts": texts}

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> Iterator[TokenChunk]:
        """Token-stream generation for single prompt."""
        self.model.eval()
        inputs = self._tokenize_batch([prompt])
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated = input_ids
        generator = self._build_generator(seed, self.device)

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(input_ids=generated, attention_mask=attention_mask)
                next_logits = outputs.logits[:, -1, :]

                next_token = sample_next_token(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    generated_token_ids=generated[0],
                    do_sample=not deterministic,
                    generator=generator,
                )

                generated = torch.cat([generated, next_token[:, None]], dim=-1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=self.device),
                    ],
                    dim=-1,
                )

                token_id = int(next_token[0].item())
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                finish_reason = "eos_token" if token_id == self.tokenizer.eos_token_id else None

                yield TokenChunk(index=step, token_id=token_id, text=text, finish_reason=finish_reason)
                if finish_reason:
                    break

    def stream_generate_sse(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        for chunk in self.stream_generate(prompt, **kwargs):
            yield format_sse_chunk(chunk)
        yield format_sse_done()

    async def submit(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Queue-based API for optional continuous batching."""
        if not self.enable_continuous_batching:
            result = self.generate(prompt, **kwargs)
            return {"text": result["texts"][0], "sequence": result["sequences"][0]}

        assert self._queue is not None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._queue.put(QueuedGenerationRequest(prompt=prompt, kwargs=kwargs, future=future))
        return await future

    def _start_worker(self) -> None:
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Micro-batch worker that groups requests for one batched ``generate`` call."""
        assert self._queue is not None
        while True:
            first = self._queue.get()
            batch = [first]

            while len(batch) < self.max_batch_size:
                try:
                    batch.append(self._queue.get(timeout=self.queue_timeout_s))
                except queue.Empty:
                    break

            prompts = [request.prompt for request in batch]
            shared_kwargs = batch[0].kwargs
            results = self.generate(prompts, **shared_kwargs)

            texts = results["texts"]
            sequences = results["sequences"]
            for idx, request in enumerate(batch):
                payload = {"text": texts[idx], "sequence": sequences[idx]}
                request.future.get_loop().call_soon_threadsafe(request.future.set_result, payload)
