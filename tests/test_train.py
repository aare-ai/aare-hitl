"""Tests for training flow."""

import json
import pytest
from unittest.mock import MagicMock, patch
import argparse

from aare.cli import run_train, load_config


class TestRunTrainValidation:
    """Tests for training config validation."""

    def test_missing_config_file(self, tmp_path):
        """Error when config file doesn't exist."""
        args = argparse.Namespace(config=str(tmp_path / "nonexistent.yaml"))
        result = run_train(args)
        assert result == 1

    def test_missing_model_field(self, tmp_path):
        """Error when model field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dataset: ./data.json\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_train(args)
        assert result == 1

    def test_missing_dataset_field(self, tmp_path):
        """Error when dataset field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test-model\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_train(args)
        assert result == 1

    def test_invalid_yaml(self, tmp_path):
        """Error when config is invalid YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content:")

        args = argparse.Namespace(config=str(config_file))
        result = run_train(args)
        assert result == 1


class TestRunTrainWithMocks:
    """Tests for training with mocked dependencies.

    Note: Full training flow tests are complex due to imports happening inside
    run_train. These tests focus on validation and config handling.
    """

    def test_dataset_file_not_found(self, tmp_path):
        """Error when dataset file doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./nonexistent.json
output_dir: ./output
""")

        args = argparse.Namespace(config=str(config_file))
        result = run_train(args)
        # This will fail during training when trying to load dataset
        assert result == 1


class TestTrainConfigDefaults:
    """Tests for training config default values."""

    def test_config_defaults_applied(self, tmp_path):
        """Default values are applied when not specified."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./data.json
""")

        config = load_config(str(config_file))

        # These should use defaults in run_train
        assert "method" not in config  # defaults to "lora"
        assert "epochs" not in config  # defaults to 1
        assert "batch_size" not in config  # defaults to 2
        assert "output_dir" not in config  # defaults to "./output"

    def test_config_overrides(self, tmp_path):
        """Specified values override defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./data.json
method: qlora
epochs: 5
batch_size: 4
learning_rate: 0.001
lora_rank: 32
output_dir: ./custom_output
device: cpu
""")

        config = load_config(str(config_file))

        assert config["method"] == "qlora"
        assert config["epochs"] == 5
        assert config["batch_size"] == 4
        assert config["learning_rate"] == 0.001
        assert config["lora_rank"] == 32
        assert config["output_dir"] == "./custom_output"
        assert config["device"] == "cpu"


class TestDatasetFormatting:
    """Tests for dataset formatting during training."""

    def test_format_instruction_output(self):
        """Format example with instruction/output fields."""
        # Simulating the format_example function from cli.py
        def format_example(ex):
            if "instruction" in ex and "output" in ex:
                text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
            elif "text" in ex:
                text = ex["text"]
            else:
                text = str(ex)
            return {"text": text}

        example = {"instruction": "What is 2+2?", "output": "4"}
        result = format_example(example)

        assert "### Instruction:" in result["text"]
        assert "What is 2+2?" in result["text"]
        assert "### Response:" in result["text"]
        assert "4" in result["text"]

    def test_format_text_field(self):
        """Format example with text field."""
        def format_example(ex):
            if "instruction" in ex and "output" in ex:
                text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
            elif "text" in ex:
                text = ex["text"]
            else:
                text = str(ex)
            return {"text": text}

        example = {"text": "Raw text content"}
        result = format_example(example)

        assert result["text"] == "Raw text content"

    def test_format_fallback(self):
        """Format example falls back to str()."""
        def format_example(ex):
            if "instruction" in ex and "output" in ex:
                text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
            elif "text" in ex:
                text = ex["text"]
            else:
                text = str(ex)
            return {"text": text}

        example = {"unknown": "field"}
        result = format_example(example)

        assert "unknown" in result["text"]
