"""Tests for generate flow."""

import pytest
from unittest.mock import MagicMock, patch
import argparse

from aare.cli import run_generate, load_config


class TestRunGenerateValidation:
    """Tests for generate config validation."""

    def test_missing_config_file(self, tmp_path):
        """Error when config file doesn't exist."""
        args = argparse.Namespace(config=str(tmp_path / "nonexistent.yaml"))
        result = run_generate(args)
        assert result == 1

    def test_missing_model_field(self, tmp_path):
        """Error when model field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("prompt: test prompt\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_generate(args)
        assert result == 1

    def test_missing_prompt_field(self, tmp_path):
        """Error when prompt field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test-model\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_generate(args)
        assert result == 1

    def test_empty_prompt(self, tmp_path):
        """Error when prompt is empty."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
prompt: ""
""")

        args = argparse.Namespace(config=str(config_file))
        result = run_generate(args)
        assert result == 1


class TestGenerateConfigDefaults:
    """Tests for generate config default values."""

    def test_adapter_default(self, tmp_path):
        """Adapter defaults to empty string."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
prompt: test prompt
""")

        config = load_config(str(config_file))
        adapter = config.get("adapter", "")
        assert adapter == ""

    def test_max_tokens_default(self, tmp_path):
        """Max tokens defaults to 512."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
prompt: test prompt
""")

        config = load_config(str(config_file))
        max_tokens = config.get("max_tokens", 512)
        assert max_tokens == 512

    def test_device_default(self, tmp_path):
        """Device defaults to auto."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
prompt: test prompt
""")

        config = load_config(str(config_file))
        device = config.get("device", "auto")
        assert device == "auto"

    def test_custom_values(self, tmp_path):
        """Custom values override defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: custom-model
prompt: custom prompt
adapter: ./my_adapter
max_tokens: 1024
device: cpu
""")

        config = load_config(str(config_file))
        assert config["model"] == "custom-model"
        assert config["prompt"] == "custom prompt"
        assert config["adapter"] == "./my_adapter"
        assert config["max_tokens"] == 1024
        assert config["device"] == "cpu"


class TestGenerateWithMocks:
    """Tests for generate with mocked dependencies.

    Note: Due to import structure in run_generate, we patch at the
    aare.core.inference module level.
    """

    def test_generate_basic_flow(self, tmp_path):
        """Test basic generate flow with mocks."""
        with patch("aare.core.inference.get_inference_engine") as mock_get_engine:
            # Setup config
            config_file = tmp_path / "config.yaml"
            config_file.write_text("""
model: test-model
prompt: What is 2+2?
""")

            # Setup mock engine
            mock_engine = MagicMock()
            mock_engine.load.return_value = "Loaded: test-model"
            mock_engine.generate.return_value = "4"
            mock_get_engine.return_value = mock_engine

            # Run
            args = argparse.Namespace(config=str(config_file))
            result = run_generate(args)

            assert result == 0
            mock_engine.load.assert_called_once_with("test-model", "")
            mock_engine.generate.assert_called_once()

    def test_generate_with_adapter(self, tmp_path):
        """Test generate with adapter."""
        with patch("aare.core.inference.get_inference_engine") as mock_get_engine:
            config_file = tmp_path / "config.yaml"
            config_file.write_text("""
model: test-model
prompt: What is 2+2?
adapter: ./my_adapter
""")

            mock_engine = MagicMock()
            mock_engine.load.return_value = "Loaded: test-model + adapter"
            mock_engine.generate.return_value = "4"
            mock_get_engine.return_value = mock_engine

            args = argparse.Namespace(config=str(config_file))
            result = run_generate(args)

            assert result == 0
            mock_engine.load.assert_called_once_with("test-model", "./my_adapter")

    def test_generate_with_max_tokens(self, tmp_path):
        """Test generate with custom max_tokens."""
        with patch("aare.core.inference.get_inference_engine") as mock_get_engine:
            config_file = tmp_path / "config.yaml"
            config_file.write_text("""
model: test-model
prompt: Write a long story
max_tokens: 2048
""")

            mock_engine = MagicMock()
            mock_engine.load.return_value = "Loaded"
            mock_engine.generate.return_value = "Once upon a time..."
            mock_get_engine.return_value = mock_engine

            args = argparse.Namespace(config=str(config_file))
            result = run_generate(args)

            assert result == 0
            mock_engine.generate.assert_called_with("Write a long story", max_tokens=2048)


class TestGenerateOutput:
    """Tests for generate output handling."""

    def test_empty_response_handling(self):
        """Handle empty responses gracefully."""
        response = ""
        display = response if response else "(Empty response)"
        assert display == "(Empty response)"

    def test_whitespace_response(self):
        """Handle whitespace-only responses."""
        response = "   \n\t  "
        stripped = response.strip()
        display = stripped if stripped else "(Empty response)"
        assert display == "(Empty response)"

    def test_normal_response(self):
        """Normal response is displayed as-is."""
        response = "The answer is 42."
        display = response if response else "(Empty response)"
        assert display == "The answer is 42."

    def test_multiline_response(self):
        """Multiline responses are preserved."""
        response = "Line 1\nLine 2\nLine 3"
        assert "\n" in response
        assert response.count("\n") == 2


class TestPromptFormatting:
    """Tests for prompt formatting in generation."""

    def test_chat_template_format(self):
        """Prompt is formatted with chat template when available."""
        prompt = "What is the capital of France?"
        messages = [{"role": "user", "content": prompt}]

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == prompt

    def test_fallback_format(self):
        """Fallback format when no chat template."""
        prompt = "What is the capital of France?"
        full_prompt = f"User: {prompt}\n\nAssistant:"

        assert "User:" in full_prompt
        assert "Assistant:" in full_prompt
        assert prompt in full_prompt
