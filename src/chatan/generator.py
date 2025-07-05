"""LLM generators with CPU fallback and aggressive memory management."""

import gc
import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import anthropic
import openai

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BaseGenerator(ABC):
    """Base class for LLM generators."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content from a prompt."""
        pass


class OpenAIGenerator(BaseGenerator):
    """OpenAI GPT generator."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.default_kwargs = kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using OpenAI API."""
        merged_kwargs = {**self.default_kwargs, **kwargs}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **merged_kwargs,
        )
        return response.choices[0].message.content.strip()


class AnthropicGenerator(BaseGenerator):
    """Anthropic Claude generator."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.default_kwargs = kwargs

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content using Anthropic API."""
        merged_kwargs = {**self.default_kwargs, **kwargs}

        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=merged_kwargs.pop("max_tokens", 1000),
            **merged_kwargs,
        )
        return response.content[0].text.strip()


class TransformersGenerator(BaseGenerator):
    """Local HuggingFace/transformers generator with robust error handling and numerical stability."""

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
        """Initialize model with proper device handling."""
        if torch.cuda.is_available() and not self.force_cpu:
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

        # Ensure model is on correct device
        self.model = self.model.to(self.device)

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

    def _validate_generation_params(self, **kwargs) -> Dict[str, Any]:
        """Validate and sanitize generation parameters to prevent numerical issues."""
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Handle sampling parameters with numerical stability
        do_sample = kwargs.get("do_sample", True)
        temperature = kwargs.get("temperature", 0.7)
        
        if do_sample:
            # Ensure temperature is in a safe range to prevent inf/nan
            temperature = max(0.01, min(temperature, 2.0))  # Clamp between 0.01 and 2.0
            generation_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            })
        else:
            # Use greedy decoding
            generation_kwargs["do_sample"] = False

        # Add additional safety parameters
        generation_kwargs.update({
            "no_repeat_ngram_size": kwargs.get("no_repeat_ngram_size", 3),
            "early_stopping": kwargs.get("early_stopping", True),
            "use_cache": True,
        })

        return generation_kwargs

    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Validate input tensors for numerical stability."""
        # Move inputs to correct device
        for key, tensor in inputs.items():
            inputs[key] = tensor.to(self.device)
        
        # Ensure attention_mask is present
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        
        return inputs

    def _safe_generate(self, inputs: Dict[str, torch.Tensor], generation_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Safely generate with fallback strategies."""
        try:
            # First attempt with provided parameters
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            return outputs
        
        except RuntimeError as e:
            if "probability tensor contains" in str(e):
                print(f"Numerical instability detected, falling back to greedy decoding: {e}")
                
                # Fallback to greedy decoding
                safe_kwargs = generation_kwargs.copy()
                safe_kwargs.update({
                    "do_sample": False,
                })
                # Remove sampling-specific parameters
                for key in ["temperature", "top_p", "top_k"]:
                    safe_kwargs.pop(key, None)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **safe_kwargs)
                return outputs
            else:
                raise e

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content with robust error handling and numerical stability."""
        try:
            # Validate generation parameters
            generation_kwargs = self._validate_generation_params(**kwargs)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=kwargs.get("max_input_length", 2048),
            )
            
            # Validate inputs
            inputs = self._validate_inputs(inputs)
            
            # Generate with safety measures
            outputs = self._safe_generate(inputs, generation_kwargs)

            # Extract new tokens only
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            # Decode
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Cleanup
            del inputs, outputs, generated_tokens
            self._clear_cache()

            return result.strip() if isinstance(result, str) else str(result).strip()

        except Exception as e:
            print(f"Generation failed: {e}")
            self._clear_cache()
            
            # Return a descriptive error message instead of crashing
            return f"[GENERATION_ERROR: {type(e).__name__}: {str(e)[:100]}]"

    def __del__(self):
        """Cleanup when destroyed."""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
            self._clear_cache()
        except:
            pass


class GeneratorFunction:
    """Callable generator function for use in dataset schemas."""

    def __init__(
        self,
        generator: BaseGenerator,
        prompt_template: str,
        variables: Optional[Dict[str, Any]] = None,
    ):
        self.generator = generator
        self.prompt_template = prompt_template
        self.variables = variables or {}

    def __call__(self, context: Dict[str, Any]) -> str:
        """Generate content with context substitution."""
        merged = dict(context)
        for key, value in self.variables.items():
            merged[key] = value(context) if callable(value) else value

        prompt = self.prompt_template.format(**merged)
        result = self.generator.generate(prompt)
        return result.strip() if isinstance(result, str) else result


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


# Factory function
def generator(
    provider: str = "openai", api_key: Optional[str] = None, **kwargs
) -> GeneratorClient:
    """Create a generator client."""
    if provider.lower() in {"openai", "anthropic"} and api_key is None:
        raise ValueError("API key is required")
    return GeneratorClient(provider, api_key, **kwargs)
