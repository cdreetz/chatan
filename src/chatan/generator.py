"""LLM generators with CPU fallback and aggressive memory management."""

from typing import Dict, Any, Optional, Union, List
import openai
import anthropic
from abc import ABC, abstractmethod
import torch
import gc
import os


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
            **merged_kwargs
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
            **merged_kwargs
        )
        return response.content[0].text.strip()


class TransformersGenerator(BaseGenerator):
    """Local HuggingFace/transformers generator with aggressive memory management."""

    def __init__(self, model: str, force_cpu: bool = False, **kwargs):
        self.model_name = model
        self.force_cpu = force_cpu
        self.model = None
        self.tokenizer = None
        self.device = None
        self.pipeline_kwargs = kwargs
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model - force CPU to avoid MPS tensor size bug."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Force CPU - MPS has a known bug with large tensor allocations
        self.device = "cpu"
        print("Using CPU to avoid MPS tensor allocation bug (not a memory issue)")

        # Optimized settings for CPU with 16GB RAM
        model_kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        print(f"Loading {self.model_name} on CPU...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully on CPU")

    def _clear_cache(self):
        """Clear all possible caches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass  # Ignore MPS cache clearing errors
        gc.collect()

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate content - optimized for CPU with 16GB RAM."""
        try:
            # Reasonable settings for 1.5B model on 16GB
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
                max_length=2048
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )

            # Extract new tokens
            input_length = inputs['input_ids'].shape[1]
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
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
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
                # Remove the auto CPU forcing - let user decide
                self._generator = TransformersGenerator(**kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            if not isinstance(e, (ValueError, RuntimeError)) or "Failed to load model" not in str(e):
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
def generator(provider: str = "openai", api_key: Optional[str] = None, **kwargs) -> GeneratorClient:
    """Create a generator client."""
    if provider.lower() in {"openai", "anthropic"} and api_key is None:
        raise ValueError("API key is required")
    return GeneratorClient(provider, api_key, **kwargs)
