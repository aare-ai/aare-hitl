"""Tests for configuration loading."""

import pytest

from aare.cli import load_config


class TestLoadConfig:
    """Tests for YAML config loading."""

    def test_load_valid_yaml(self, tmp_path):
        """Load a valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset: ./data/my_dataset.json
output_dir: ./output
epochs: 3
batch_size: 2
learning_rate: 0.0002
""")

        config = load_config(str(config_file))

        assert config["model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config["dataset"] == "./data/my_dataset.json"
        assert config["epochs"] == 3
        assert config["batch_size"] == 2
        assert config["learning_rate"] == 0.0002

    def test_load_yml_extension(self, tmp_path):
        """Load config with .yml extension."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("model: test-model\n")

        config = load_config(str(config_file))
        assert config["model"] == "test-model"

    def test_file_not_found(self, tmp_path):
        """Raise error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_wrong_extension(self, tmp_path):
        """Raise error for non-YAML file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"model": "test"}')

        with pytest.raises(ValueError, match="YAML"):
            load_config(str(config_file))

    def test_load_with_comments(self, tmp_path):
        """Load YAML with comments."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
# Training configuration
model: test-model  # The model to train
# Dataset path
dataset: ./data.json
""")

        config = load_config(str(config_file))
        assert config["model"] == "test-model"
        assert config["dataset"] == "./data.json"

    def test_load_nested_config(self, tmp_path):
        """Load YAML with nested structures."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
""")

        config = load_config(str(config_file))
        assert config["lora"]["rank"] == 16
        assert config["lora"]["alpha"] == 32

    def test_load_with_lists(self, tmp_path):
        """Load YAML with list values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
target_modules:
  - q_proj
  - v_proj
  - k_proj
""")

        config = load_config(str(config_file))
        assert config["target_modules"] == ["q_proj", "v_proj", "k_proj"]

    def test_load_empty_file(self, tmp_path):
        """Load empty YAML file returns None."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = load_config(str(config_file))
        assert config is None

    def test_load_with_special_characters(self, tmp_path):
        """Load YAML with special characters in values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: "org/model-name_v1.0"
prompt: "What's the answer to: 2+2?"
""")

        config = load_config(str(config_file))
        assert config["model"] == "org/model-name_v1.0"
        assert config["prompt"] == "What's the answer to: 2+2?"

    def test_load_boolean_values(self, tmp_path):
        """Load YAML with boolean values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test
debug: true
verbose: false
""")

        config = load_config(str(config_file))
        assert config["debug"] is True
        assert config["verbose"] is False

    def test_load_null_values(self, tmp_path):
        """Load YAML with null values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test
adapter: null
""")

        config = load_config(str(config_file))
        assert config["adapter"] is None
