"""LLM generators with async support and memory management."""

import asyncio
import gc
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional

import anthropic
import openai

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseGenerator(ABC):
    """Base class for async LLM generators."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Asynchronously generate content from a prompt."""
        pass


class OpenAIGenerator(BaseGenerator):
    """Async OpenAI GPT generator."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        async_client_cls = getattr(openai, "AsyncOpenAI", None)
        if async_client_cls is None:
            raise ImportError(
                "Async OpenAI client is not available. Upgrade the `openai` package "
                "to a version that provides `AsyncOpenAI`."
            )

        self.client = async_client_cls(api_key=api_key)
        self.model = model
        self.default_kwargs = kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using OpenAI API asynchronously."""
        merged_kwargs = {**self.default_kwargs, **kwargs}

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **merged_kwargs,
        )
        return response.choices[0].message.content.strip()


class AnthropicGenerator(BaseGenerator):
    """Async Anthropic Claude generator."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        **kwargs,
    ):
        async_client_cls = getattr(anthropic, "AsyncAnthropic", None)
        if async_client_cls is None:
            raise ImportError(
                "Async Anthropic client is not available. Upgrade the `anthropic` package "
                "to a version that provides `AsyncAnthropic`."
            )

        self.client = async_client_cls(api_key=api_key)
        self.model = model
        self.default_kwargs = kwargs

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Anthropic API asynchronously."""
        merged_kwargs = {**self.default_kwargs, **kwargs}

        response = await self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=merged_kwargs.pop("max_tokens", 1000),
            **merged_kwargs,
        )
        return response.content[0].text.strip()


class TransformersGenerator:
    """Local HuggingFace/transformers generator with memory management.

    Note: This generator is synchronous as local model inference
    doesn't benefit from async I/O.
    """

    def __init__(self, model: str, force_cpu: bool = False, **kwargs):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers and PyTorch are required for local model generation. "
                "Install with: pip install chatan[local]"
            )

        self.model_name = model
        self.force_cpu = force_cpu
        self.model = None
        self.tokenizer = None
        self.device = None
        self.pipeline_kwargs = kwargs
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model - force CPU to avoid MPS tensor size bug."""

        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        print(f"Loading {self.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs
        )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {self.device}")

    def _clear_cache(self):
        """Clear all possible caches."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception as e:
                print(f"Failed to clear MPS cache: {e}")
        gc.collect()

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate content - runs sync inference in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt, kwargs)

    def _generate_sync(self, prompt: str, kwargs: dict) -> str:
        """Synchronous generation for local models."""
        try:
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Extract new tokens
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Cleanup
            del inputs, outputs, generated_tokens
            gc.collect()

            return result.strip() if isinstance(result, str) else result

        except Exception as e:
            print(f"Generation failed: {e}")
            gc.collect()
            raise e

    def __del__(self):
        """Cleanup when destroyed."""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
            self._clear_cache()
        except Exception as e:
            print(f"Cleanup failed with exception: {e}")
            pass


class GeneratorFunction:
    """Callable async generator function with concurrency support."""

    def __init__(
        self,
        generator: BaseGenerator,
        prompt_template: str,
        variables: Optional[Dict[str, Any]] = None,
    ):
        self.generator = generator
        self.prompt_template = prompt_template
        self.variables = variables or {}

    async def __call__(self, context: Dict[str, Any], **kwargs) -> str:
        """Generate content with context substitution asynchronously."""
        merged = dict(context)
        for key, value in self.variables.items():
            merged[key] = value(context) if callable(value) else value

        prompt = self.prompt_template.format(**merged)
        result = await self.generator.generate(prompt, **kwargs)
        return result.strip() if isinstance(result, str) else result

    async def stream(
        self,
        contexts: Iterable[Dict[str, Any]],
        *,
        concurrency: int = 5,
        return_exceptions: bool = False,
        **kwargs,
    ):
        """Asynchronously yield results for many contexts with bounded concurrency."""

        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")

        contexts_list = list(contexts)
        if not contexts_list:
            return

        semaphore = asyncio.Semaphore(concurrency)

        async def worker(index: int, ctx: Dict[str, Any]):
            async with semaphore:
                try:
                    result = await self(ctx, **kwargs)
                    return index, result, None
                except Exception as exc:
                    return index, None, exc

        tasks = [
            asyncio.create_task(worker(index, ctx))
            for index, ctx in enumerate(contexts_list)
        ]

        next_index = 0
        buffer: Dict[int, Any] = {}

        try:
            for coro in asyncio.as_completed(tasks):
                index, value, error = await coro

                if error is not None and not return_exceptions:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise error

                buffer[index] = error if error is not None else value

                while next_index in buffer:
                    item = buffer.pop(next_index)
                    next_index += 1
                    yield item

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


class GeneratorClient:
    """Main interface for creating generators."""

    def __init__(self, provider: str, api_key: Optional[str] = None, **kwargs):
        provider_lower = provider.lower()
        try:
            if provider_lower == "openai":
                if api_key is None:
                    raise ValueError("API key is required for OpenAI")
                self._generator = OpenAIGenerator(api_key, **kwargs)
            elif provider_lower == "anthropic":
                if api_key is None:
                    raise ValueError("API key is required for Anthropic")
                self._generator = AnthropicGenerator(api_key, **kwargs)
            elif provider_lower in {"huggingface", "transformers", "hf"}:
                if "model" not in kwargs:
                    raise ValueError(
                        "Model is required for transformers provider. "
                        "Example: generator('transformers', model='Qwen2.5-1.5B-Instruct')"
                    )
                self._generator = TransformersGenerator(**kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except ImportError as e:
            if "transformers" in str(e) or "PyTorch" in str(e):
                raise ImportError(
                    f"Local model support requires additional dependencies. "
                    f"Install with: pip install chatan[local]\n"
                    f"Original error: {str(e)}"
                ) from e
            raise

        except Exception as e:
            if not isinstance(
                e, (ValueError, RuntimeError)
            ) or "Failed to load model" not in str(e):
                raise ValueError(
                    f"Failed to initialize generator for provider '{provider}'. "
                    f"Check your configuration and try again. Original error: {str(e)}"
                ) from e
            else:
                raise

    def __call__(self, prompt_template: str, **variables) -> GeneratorFunction:
        """Create a generator function."""
        return GeneratorFunction(self._generator, prompt_template, variables)


def generator(
    provider: str = "openai", api_key: Optional[str] = None, **kwargs
) -> GeneratorClient:
    """Create a generator client."""
    if provider.lower() in {"openai", "anthropic"} and api_key is None:
        raise ValueError("API key is required")
    return GeneratorClient(provider, api_key, **kwargs)
