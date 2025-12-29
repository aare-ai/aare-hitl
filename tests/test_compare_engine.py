"""Tests for CompareEngine with mocked generation."""

import pytest
from unittest.mock import MagicMock, patch
import torch

from aare.core.inference import CompareEngine


class TestCompareEngineGeneration:
    """Tests for CompareEngine generation methods."""

    def test_generate_base_with_loaded_state(self):
        """Generate from base model with preloaded state."""
        engine = CompareEngine(device="cpu")

        # Setup mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Base model response"

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test prompt")

        assert result == "Base model response"
        mock_base_model.generate.assert_called_once()

    def test_generate_finetuned_with_loaded_state(self):
        """Generate from finetuned model with preloaded state."""
        engine = CompareEngine(device="cpu")

        # Setup mock model and tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Finetuned model response"

        mock_ft_model = MagicMock()
        mock_ft_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_ft_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.ft_model = mock_ft_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_finetuned("Test prompt")

        assert result == "Finetuned model response"
        mock_ft_model.generate.assert_called_once()

    def test_generate_base_empty_response(self):
        """Handle empty response from base model."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = ""

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test")
        assert result == "(Empty response)"

    def test_generate_base_error(self):
        """Handle error in base model generation."""
        engine = CompareEngine(device="cpu")

        mock_base_model = MagicMock()
        mock_base_model.generate.side_effect = Exception("Test error")

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test")
        assert "Error" in result

    def test_generate_finetuned_error(self):
        """Handle error in finetuned model generation."""
        engine = CompareEngine(device="cpu")

        mock_ft_model = MagicMock()
        mock_ft_model.generate.side_effect = Exception("Test error")

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

        engine.ft_model = mock_ft_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_finetuned("Test")
        assert "Error" in result


class TestCompareEngineChatTemplate:
    """Tests for CompareEngine chat template handling."""

    def test_generate_with_chat_template(self):
        """Generate using chat template."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = "Template: {{ messages }}"
        # apply_chat_template returns a tensor when return_tensors="pt"
        mock_tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "Response with template"

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test prompt")
        assert result == "Response with template"

    def test_generate_without_chat_template(self):
        """Generate using fallback when no chat template."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None  # No chat template
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Response without template"

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test prompt")
        assert result == "Response without template"


class TestCompareEngineMaxTokens:
    """Tests for CompareEngine max_tokens handling."""

    def test_generate_with_custom_max_tokens(self):
        """Generate with custom max_tokens."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Response"

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test prompt", max_tokens=1024)

        # Verify max_new_tokens was passed to generate
        call_kwargs = mock_base_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 1024

    def test_generate_default_max_tokens(self):
        """Generate uses default 128 max_tokens (CompareEngine default)."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Response"

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test prompt")

        # Verify default max_new_tokens of 128 (CompareEngine default)
        call_kwargs = mock_base_model.generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == 128


class TestCompareEngineWhitespaceHandling:
    """Tests for CompareEngine whitespace response handling."""

    def test_whitespace_only_response(self):
        """Handle whitespace-only response."""
        engine = CompareEngine(device="cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.chat_template = None
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "   \n\t  "

        mock_base_model = MagicMock()
        mock_base_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_base_model.generate.return_value = torch.tensor([[1, 2, 3]])

        engine.base_model = mock_base_model
        engine.tokenizer = mock_tokenizer
        engine._loaded = True

        result = engine.generate_base("Test")
        # After stripping, result should be empty response
        assert result.strip() == "" or result == "(Empty response)"
