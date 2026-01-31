"""Comprehensive tests for generator module."""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock
from chatan.generator import (
    OpenAIGenerator,
    AnthropicGenerator,
    BaseGenerator,
    GeneratorFunction,
    GeneratorClient,
    generator,
)

# Conditional imports for torch-dependent tests
try:
    import torch
    from chatan.generator import TransformersGenerator

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.asyncio
class TestOpenAIGenerator:
    """Test OpenAI generator implementation."""

    @patch("openai.AsyncOpenAI")
    async def test_init_default_model(self, mock_async_openai):
        """Test OpenAI generator initialization with default model."""
        gen = OpenAIGenerator("test-key")
        assert gen.model == "gpt-3.5-turbo"
        mock_async_openai.assert_called_once_with(api_key="test-key")

    @patch("openai.AsyncOpenAI")
    async def test_init_custom_model(self, mock_async_openai):
        """Test OpenAI generator initialization with custom model."""
        gen = OpenAIGenerator("test-key", model="gpt-4", temperature=0.8)
        assert gen.model == "gpt-4"
        assert gen.default_kwargs == {"temperature": 0.8}

    @patch("openai.AsyncOpenAI")
    async def test_generate_basic(self, mock_async_openai):
        """Test basic content generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "  Generated content  "
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key")
        result = await gen.generate("Test prompt")

        assert result == "Generated content"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test prompt"}]
        )

    @patch("openai.AsyncOpenAI")
    async def test_generate_with_kwargs(self, mock_async_openai):
        """Test generation with additional kwargs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated"
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key", temperature=0.5)
        result = await gen.generate("Test", max_tokens=100)

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            max_tokens=100,
        )

    @patch("openai.AsyncOpenAI")
    async def test_kwargs_override(self, mock_async_openai):
        """Test that call-time kwargs override defaults."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Generated"
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key", temperature=0.5)
        await gen.generate("Test", temperature=0.9)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.9


@pytest.mark.asyncio
class TestAnthropicGenerator:
    """Test Anthropic generator implementation."""

    @patch("anthropic.AsyncAnthropic")
    async def test_init_default_model(self, mock_async_anthropic):
        """Test Anthropic generator initialization."""
        gen = AnthropicGenerator("test-key")
        assert gen.model == "claude-3-sonnet-20240229"
        mock_async_anthropic.assert_called_once_with(api_key="test-key")

    @patch("anthropic.AsyncAnthropic")
    async def test_generate_basic(self, mock_async_anthropic):
        """Test basic content generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "  Generated content  "
        mock_response.content = [mock_content]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.messages.create.return_value = future
        mock_async_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        result = await gen.generate("Test prompt")

        assert result == "Generated content"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=1000,
        )

    @patch("anthropic.AsyncAnthropic")
    async def test_max_tokens_extraction(self, mock_async_anthropic):
        """Test that max_tokens is extracted from kwargs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Generated"
        mock_response.content = [mock_content]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.messages.create.return_value = future
        mock_async_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        await gen.generate("Test", max_tokens=500, temperature=0.7)

        call_args = mock_client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["temperature"] == 0.7


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformersGenerator:
    """Test TransformersGenerator functionality (only when torch is available)."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_transformers_init(self, mock_model, mock_tokenizer):
        """Test TransformersGenerator initialization."""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "[EOS]"
        mock_tokenizer.return_value = mock_tok

        # Mock model
        mock_mdl = Mock()
        mock_model.return_value = mock_mdl

        with patch("torch.cuda.is_available", return_value=False):
            gen = TransformersGenerator("gpt2")

        assert gen.model_name == "gpt2"
        assert gen.device == "cpu"
        mock_tokenizer.assert_called_once_with("gpt2")


@pytest.mark.asyncio
class TestGeneratorFunction:
    """Test GeneratorFunction wrapper."""

    async def test_template_substitution(self):
        """Test template variable substitution."""

        class DummyGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "Generated content"

        func = GeneratorFunction(DummyGenerator(), "Write about {topic} in {style}")
        result = await func({"topic": "AI", "style": "casual"})

        assert result == "Generated content"

    async def test_missing_context_variable(self):
        """Test behavior with missing context variables."""

        class DummyGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return prompt

        func = GeneratorFunction(DummyGenerator(), "Write about {topic}")

        with pytest.raises(KeyError):
            await func({"wrong_key": "value"})

    async def test_extra_context_variables(self):
        """Test behavior with extra context variables."""

        class DummyGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "Generated"

        func = GeneratorFunction(DummyGenerator(), "Write about {topic}")
        result = await func({"topic": "AI", "extra": "ignored"})

        assert result == "Generated"

    async def test_stream_concurrency(self):
        """Ensure stream runs with bounded concurrency and preserves order."""

        class ConcurrentGenerator(BaseGenerator):
            def __init__(self):
                self.active = 0
                self.max_active = 0

            async def generate(self, prompt: str, **kwargs) -> str:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep(0.01)
                    return prompt
                finally:
                    self.active -= 1

        gen = ConcurrentGenerator()
        func = GeneratorFunction(gen, "item {value}")
        contexts = [{"value": i} for i in range(4)]

        results = []
        async for value in func.stream(contexts, concurrency=2):
            results.append(value)

        assert results == [f"item {i}" for i in range(4)]
        assert gen.max_active == 2

    async def test_stream_exceptions(self):
        """Ensure exceptions can be captured or raised."""

        class FailingGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                if "fail" in prompt:
                    raise ValueError("boom")
                return prompt

        func = GeneratorFunction(FailingGenerator(), "{value}")
        contexts = [{"value": "ok"}, {"value": "fail"}, {"value": "later"}]

        results = []
        async for value in func.stream(contexts, return_exceptions=True):
            results.append(value)

        assert isinstance(results[1], ValueError)
        assert results[0] == "ok"
        assert results[2] == "later"

        with pytest.raises(ValueError):
            async for _ in func.stream(contexts):
                pass


class TestGeneratorClient:
    """Test GeneratorClient interface."""

    @patch("chatan.generator.OpenAIGenerator")
    def test_openai_client_creation(self, mock_openai_gen):
        """Test OpenAI client creation."""
        client = GeneratorClient("openai", "test-key", temperature=0.7)
        mock_openai_gen.assert_called_once_with("test-key", temperature=0.7)

    @patch("chatan.generator.AnthropicGenerator")
    def test_anthropic_client_creation(self, mock_anthropic_gen):
        """Test Anthropic client creation."""
        client = GeneratorClient("anthropic", "test-key", model="claude-3-opus-20240229")
        mock_anthropic_gen.assert_called_once_with(
            "test-key", model="claude-3-opus-20240229"
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch("chatan.generator.TransformersGenerator")
    def test_transformers_client_creation(self, mock_hf_gen):
        """Test Transformers client creation."""
        client = GeneratorClient("transformers", model="gpt2")
        mock_hf_gen.assert_called_once_with(model="gpt2")

    def test_transformers_client_creation_no_torch(self):
        """Test Transformers client creation when torch is not available."""
        with patch("chatan.generator.TRANSFORMERS_AVAILABLE", False):
            with pytest.raises(
                ImportError, match="Local model support requires additional dependencies"
            ):
                GeneratorClient("transformers", model="gpt2")

    def test_unsupported_provider(self):
        """Test error handling for unsupported providers."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            GeneratorClient("invalid", "test-key")

    @patch("chatan.generator.OpenAIGenerator")
    def test_callable_returns_generator_function(self, mock_openai_gen):
        """Test that calling client returns GeneratorFunction."""
        client = GeneratorClient("openai", "test-key")
        func = client("Template {var}")

        assert isinstance(func, GeneratorFunction)
        assert func.prompt_template == "Template {var}"


class TestGeneratorFactory:
    """Test generator factory function."""

    def test_missing_api_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API key is required"):
            generator("openai")

    @patch("chatan.generator.GeneratorClient")
    def test_factory_creates_client(self, mock_client):
        """Test factory function creates GeneratorClient."""
        result = generator("openai", "test-key", temperature=0.5)
        mock_client.assert_called_once_with("openai", "test-key", temperature=0.5)

    @patch("chatan.generator.GeneratorClient")
    def test_default_provider(self, mock_client):
        """Test default provider is openai."""
        generator(api_key="test-key")
        mock_client.assert_called_once_with("openai", "test-key")

    @patch("chatan.generator.GeneratorClient")
    def test_transformers_provider_no_key(self, mock_client):
        """Transformers provider should not require API key."""
        generator("transformers", model="gpt2")
        mock_client.assert_called_once_with("transformers", None, model="gpt2")


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for generator components."""

    @patch("openai.AsyncOpenAI")
    async def test_end_to_end_openai(self, mock_async_openai):
        """Test complete OpenAI generation pipeline."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "The capital of France is Paris."
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = generator("openai", "test-key")
        func = gen("What is the capital of {country}?")
        result = await func({"country": "France"})

        assert result == "The capital of France is Paris."
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

    @patch("anthropic.AsyncAnthropic")
    async def test_end_to_end_anthropic(self, mock_async_anthropic):
        """Test complete Anthropic generation pipeline."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Python is a programming language."
        mock_response.content = [mock_content]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.messages.create.return_value = future
        mock_async_anthropic.return_value = mock_client

        gen = generator("anthropic", "test-key")
        func = gen("Explain {topic}")
        result = await func({"topic": "Python"})

        assert result == "Python is a programming language."

    @patch("openai.AsyncOpenAI")
    async def test_multiple_generations(self, mock_async_openai):
        """Test multiple generations with same generator."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = generator("openai", "test-key")
        func = gen("Generate {type}")

        result1 = await func({"type": "poem"})
        result2 = await func({"type": "story"})

        assert result1 == "Response"
        assert result2 == "Response"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("openai.AsyncOpenAI")
    async def test_generator_function_with_variables(self, mock_async_openai):
        """GeneratorFunction should accept default variables."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Question about elephants"
        mock_response.choices = [mock_choice]

        future = asyncio.Future()
        future.set_result(mock_response)
        mock_client.chat.completions.create.return_value = future
        mock_async_openai.return_value = mock_client

        gen = generator("openai", "test-key")
        func = gen("Question about {animal}", animal="elephants")
        result = await func({})

        assert result == "Question about elephants"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Question about elephants"}],
        )

    def test_case_insensitive_provider(self):
        """Test that provider names are case insensitive."""
        with patch("chatan.generator.OpenAIGenerator") as mock_gen:
            generator("OPENAI", "test-key")
            mock_gen.assert_called_once()

        with patch("chatan.generator.AnthropicGenerator") as mock_gen:
            generator("ANTHROPIC", "test-key")
            mock_gen.assert_called_once()

        if TORCH_AVAILABLE:
            with patch("chatan.generator.TransformersGenerator") as mock_gen:
                generator("TRANSFORMERS", model="gpt2")
                mock_gen.assert_called_once()


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling scenarios."""

    @patch("openai.AsyncOpenAI")
    async def test_openai_api_error(self, mock_async_openai):
        """Test handling of OpenAI API errors."""
        mock_client = MagicMock()

        async def raise_error(*args, **kwargs):
            raise Exception("API Error")

        mock_client.chat.completions.create = raise_error
        mock_async_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key")
        with pytest.raises(Exception, match="API Error"):
            await gen.generate("Test prompt")

    @patch("anthropic.AsyncAnthropic")
    async def test_anthropic_api_error(self, mock_async_anthropic):
        """Test handling of Anthropic API errors."""
        mock_client = MagicMock()

        async def raise_error(*args, **kwargs):
            raise Exception("API Error")

        mock_client.messages.create = raise_error
        mock_async_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        with pytest.raises(Exception, match="API Error"):
            await gen.generate("Test prompt")

    async def test_empty_response_handling(self):
        """Test handling of empty responses."""

        class WhitespaceGenerator(BaseGenerator):
            async def generate(self, prompt: str, **kwargs) -> str:
                return "   "

        func = GeneratorFunction(WhitespaceGenerator(), "Generate {thing}")
        result = await func({"thing": "content"})

        assert result == ""  # Should be stripped to empty string
