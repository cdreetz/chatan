"""Comprehensive tests for generator module."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from chatan.generator import (
    OpenAIGenerator,
    AnthropicGenerator,
    GeneratorFunction,
    GeneratorClient,
    generator
)

# Conditional imports for torch-dependent tests
try:
    import torch
    from chatan.generator import TransformersGenerator
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestOpenAIGenerator:
    """Test OpenAI generator implementation."""

    @patch('openai.OpenAI')
    def test_init_default_model(self, mock_openai):
        """Test OpenAI generator initialization with default model."""
        gen = OpenAIGenerator("test-key")
        assert gen.model == "gpt-3.5-turbo"
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch('openai.OpenAI')
    def test_init_custom_model(self, mock_openai):
        """Test OpenAI generator initialization with custom model."""
        gen = OpenAIGenerator("test-key", model="gpt-4", temperature=0.8)
        assert gen.model == "gpt-4"
        assert gen.default_kwargs == {"temperature": 0.8}

    @patch('openai.OpenAI')
    def test_generate_basic(self, mock_openai):
        """Test basic content generation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "  Generated content  "
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key")
        result = gen.generate("Test prompt")

        assert result == "Generated content"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test prompt"}]
        )

    @patch('openai.OpenAI')
    def test_generate_with_kwargs(self, mock_openai):
        """Test generation with additional kwargs."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Generated"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key", temperature=0.5)
        result = gen.generate("Test", max_tokens=100)

        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.5,
            max_tokens=100
        )

    @patch('openai.OpenAI')
    def test_kwargs_override(self, mock_openai):
        """Test that call-time kwargs override defaults."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Generated"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key", temperature=0.5)
        gen.generate("Test", temperature=0.9)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.9


class TestAnthropicGenerator:
    """Test Anthropic generator implementation."""

    @patch('anthropic.Anthropic')
    def test_init_default_model(self, mock_anthropic):
        """Test Anthropic generator initialization."""
        gen = AnthropicGenerator("test-key")
        assert gen.model == "claude-3-sonnet-20240229"
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @patch('anthropic.Anthropic')
    def test_generate_basic(self, mock_anthropic):
        """Test basic content generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "  Generated content  "
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        result = gen.generate("Test prompt")

        assert result == "Generated content"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": "Test prompt"}],
            max_tokens=1000
        )

    @patch('anthropic.Anthropic')
    def test_max_tokens_extraction(self, mock_anthropic):
        """Test that max_tokens is extracted from kwargs."""
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Generated"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        gen.generate("Test", max_tokens=500, temperature=0.7)

        call_args = mock_client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 500
        assert call_args[1]["temperature"] == 0.7


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTransformersGenerator:
    """Test TransformersGenerator functionality (only when torch is available)."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
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

        with patch('torch.cuda.is_available', return_value=False):
            gen = TransformersGenerator("gpt2")
            
        assert gen.model_name == "gpt2"
        assert gen.device == "cpu"
        mock_tokenizer.assert_called_once_with("gpt2")


class TestGeneratorFunction:
    """Test GeneratorFunction wrapper."""

    def test_template_substitution(self):
        """Test template variable substitution."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "Generated content"
        
        func = GeneratorFunction(mock_generator, "Write about {topic} in {style}")
        result = func({"topic": "AI", "style": "casual"})
        
        assert result == "Generated content"
        mock_generator.generate.assert_called_once_with("Write about AI in casual")

    def test_missing_context_variable(self):
        """Test behavior with missing context variables."""
        mock_generator = Mock()
        func = GeneratorFunction(mock_generator, "Write about {topic}")
        
        with pytest.raises(KeyError):
            func({"wrong_key": "value"})

    def test_extra_context_variables(self):
        """Test behavior with extra context variables."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "Generated"
        
        func = GeneratorFunction(mock_generator, "Write about {topic}")
        result = func({"topic": "AI", "extra": "ignored"})
        
        assert result == "Generated"
        mock_generator.generate.assert_called_once_with("Write about AI")


class TestGeneratorClient:
    """Test GeneratorClient interface."""

    @patch('chatan.generator.OpenAIGenerator')
    def test_openai_client_creation(self, mock_openai_gen):
        """Test OpenAI client creation."""
        client = GeneratorClient("openai", "test-key", temperature=0.7)
        mock_openai_gen.assert_called_once_with("test-key", temperature=0.7)

    @patch('chatan.generator.AnthropicGenerator')
    def test_anthropic_client_creation(self, mock_anthropic_gen):
        """Test Anthropic client creation."""
        client = GeneratorClient("anthropic", "test-key", model="claude-3-opus-20240229")
        mock_anthropic_gen.assert_called_once_with("test-key", model="claude-3-opus-20240229")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('chatan.generator.TransformersGenerator')
    def test_transformers_client_creation(self, mock_hf_gen):
        """Test Transformers client creation."""
        client = GeneratorClient("transformers", model="gpt2")
        mock_hf_gen.assert_called_once_with(model="gpt2")

    def test_transformers_client_creation_no_torch(self):
        """Test Transformers client creation when torch is not available."""
        # Temporarily patch TRANSFORMERS_AVAILABLE to False
        with patch('chatan.generator.TRANSFORMERS_AVAILALBE', False):
            with pytest.raises(ImportError, match="Local model support requires additional dependencies"):
                GeneratorClient("transformers", model="gpt2")

    def test_unsupported_provider(self):
        """Test error handling for unsupported providers."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            GeneratorClient("invalid", "test-key")

    @patch('chatan.generator.OpenAIGenerator')
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

    @patch('chatan.generator.GeneratorClient')
    def test_factory_creates_client(self, mock_client):
        """Test factory function creates GeneratorClient."""
        result = generator("openai", "test-key", temperature=0.5)
        mock_client.assert_called_once_with("openai", "test-key", temperature=0.5)

    @patch('chatan.generator.GeneratorClient')
    def test_default_provider(self, mock_client):
        """Test default provider is openai."""
        generator(api_key="test-key")
        mock_client.assert_called_once_with("openai", "test-key")

    @patch('chatan.generator.GeneratorClient')
    def test_transformers_provider_no_key(self, mock_client):
        """Transformers provider should not require API key."""
        generator("transformers", model="gpt2")
        mock_client.assert_called_once_with("transformers", None, model="gpt2")


class TestIntegration:
    """Integration tests for generator components."""

    @patch('openai.OpenAI')
    def test_end_to_end_openai(self, mock_openai):
        """Test complete OpenAI generation pipeline."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "The capital of France is Paris."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Use the API
        gen = generator("openai", "test-key")
        func = gen("What is the capital of {country}?")
        result = func({"country": "France"})

        assert result == "The capital of France is Paris."
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "What is the capital of France?"}]
        )

    @patch('anthropic.Anthropic')
    def test_end_to_end_anthropic(self, mock_anthropic):
        """Test complete Anthropic generation pipeline."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Python is a programming language."
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Use the API
        gen = generator("anthropic", "test-key")
        func = gen("Explain {topic}")
        result = func({"topic": "Python"})

        assert result == "Python is a programming language."

    @patch('openai.OpenAI')
    def test_multiple_generations(self, mock_openai):
        """Test multiple generations with same generator."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Response"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = generator("openai", "test-key")
        func = gen("Generate {type}")
        
        result1 = func({"type": "poem"})
        result2 = func({"type": "story"})
        
        assert result1 == "Response"
        assert result2 == "Response"
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_end_to_end_transformers(self, mock_model, mock_tokenizer):
        """Test complete Transformers generation pipeline."""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "[EOS]"
        mock_tok.eos_token_id = 2
        mock_tok.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tok.decode.return_value = "Hello"
        mock_tokenizer.return_value = mock_tok
        
        # Mock model
        mock_mdl = Mock()
        mock_mdl.generate.return_value = [torch.tensor([1, 2, 3, 4, 5])]
        mock_model.return_value = mock_mdl

        gen = generator("transformers", model="gpt2")
        func = gen("Say hi to {name}")
        
        with patch('torch.no_grad'):
            result = func({"name": "Bob"})

        assert result == "Hello"
        mock_tokenizer.assert_called_once_with("gpt2")

    @patch('openai.OpenAI')
    def test_generator_function_with_variables(self, mock_openai):
        """GeneratorFunction should accept default variables."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Question about elephants"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        gen = generator("openai", "test-key")
        func = gen("Question about {animal}", animal="elephants")
        result = func({})

        assert result == "Question about elephants"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Question about elephants"}]
        )

    def test_case_insensitive_provider(self):
        """Test that provider names are case insensitive."""
        with patch('chatan.generator.OpenAIGenerator') as mock_gen:
            generator("OPENAI", "test-key")
            mock_gen.assert_called_once()
            
        with patch('chatan.generator.AnthropicGenerator') as mock_gen:
            generator("ANTHROPIC", "test-key")
            mock_gen.assert_called_once()

        if TORCH_AVAILABLE:
            with patch('chatan.generator.TransformersGenerator') as mock_gen:
                generator("TRANSFORMERS", model="gpt2")
                mock_gen.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios."""

    @patch('openai.OpenAI')
    def test_openai_api_error(self, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        gen = OpenAIGenerator("test-key")
        with pytest.raises(Exception, match="API Error"):
            gen.generate("Test prompt")

    @patch('anthropic.Anthropic')
    def test_anthropic_api_error(self, mock_anthropic):
        """Test handling of Anthropic API errors."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        gen = AnthropicGenerator("test-key")
        with pytest.raises(Exception, match="API Error"):
            gen.generate("Test prompt")

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        mock_generator = Mock()
        mock_generator.generate.return_value = "   "  # Whitespace only
        
        func = GeneratorFunction(mock_generator, "Generate {thing}")
        result = func({"thing": "content"})
        
        assert result == ""  # Should be stripped to empty string