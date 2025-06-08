"""LLM generators for synthetic data creation."""

from typing import Dict, Any, Optional, Union, List
import openai
import anthropic
from abc import ABC, abstractmethod


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
    
    def __init__(self, provider: str, api_key: str, **kwargs):
        if provider.lower() == "openai":
            self._generator = OpenAIGenerator(api_key, **kwargs)
        elif provider.lower() == "anthropic":
            self._generator = AnthropicGenerator(api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def __call__(self, prompt_template: str, **variables) -> GeneratorFunction:
        """Create a generator function.

        Parameters
        ----------
        prompt_template:
            Template string for the prompt.
        **variables:
            Optional variables to include when formatting the prompt. If a value
            is callable it will be invoked with the row context when the
            generator function is executed.
        """
        return GeneratorFunction(self._generator, prompt_template, variables)


# Factory function
def generator(provider: str = "openai", api_key: Optional[str] = None, **kwargs) -> GeneratorClient:
    """Create a generator client."""
    if api_key is None:
        raise ValueError("API key is required")
    return GeneratorClient(provider, api_key, **kwargs)
