"""Additional tests for config edge cases."""

import pytest
from pathlib import Path

from aare.cli import load_config


class TestLoadConfigValidation:
    """Tests for config validation in load_config."""

    def test_config_not_yaml_extension(self, tmp_path):
        """Error when config file isn't YAML."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("model: test\n")

        with pytest.raises(ValueError, match="must be YAML"):
            load_config(str(config_file))

    def test_config_json_extension_rejected(self, tmp_path):
        """Error when config file is JSON (not YAML)."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"model": "test"}')

        with pytest.raises(ValueError, match="must be YAML"):
            load_config(str(config_file))

    def test_config_yaml_extension(self, tmp_path):
        """Accept .yaml extension."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n")

        config = load_config(str(config_file))
        assert config["model"] == "test"

    def test_config_yml_extension(self, tmp_path):
        """Accept .yml extension."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("model: test\n")

        config = load_config(str(config_file))
        assert config["model"] == "test"

    def test_config_with_lists(self, tmp_path):
        """Config with list values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test
prompts:
  - prompt 1
  - prompt 2
  - prompt 3
""")

        config = load_config(str(config_file))
        assert len(config["prompts"]) == 3

    def test_config_with_nested_dicts(self, tmp_path):
        """Config with nested dict values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
""")

        config = load_config(str(config_file))
        assert config["lora"]["rank"] == 16
        assert config["lora"]["alpha"] == 32

    def test_config_with_numbers(self, tmp_path):
        """Config with various number types."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
epochs: 5
learning_rate: 0.0001
batch_size: 16
temperature: 0.7
""")

        config = load_config(str(config_file))
        assert config["epochs"] == 5
        assert config["learning_rate"] == 0.0001
        assert isinstance(config["temperature"], float)

    def test_config_with_booleans(self, tmp_path):
        """Config with boolean values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
use_flash_attention: true
debug: false
gradient_checkpointing: yes
""")

        config = load_config(str(config_file))
        assert config["use_flash_attention"] is True
        assert config["debug"] is False
        assert config["gradient_checkpointing"] is True

    def test_config_multiline_string(self, tmp_path):
        """Config with multiline string."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
prompt: |
  This is a
  multiline
  prompt
""")

        config = load_config(str(config_file))
        assert "multiline" in config["prompt"]
        assert "\n" in config["prompt"]

    def test_config_null_values(self, tmp_path):
        """Config with null values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test
adapter: null
optional_param: ~
""")

        config = load_config(str(config_file))
        assert config["adapter"] is None
        assert config["optional_param"] is None

    def test_config_empty_file(self, tmp_path):
        """Config with empty file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = load_config(str(config_file))
        assert config is None

    def test_config_only_comments(self, tmp_path):
        """Config with only comments."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
# This is a comment
# Another comment
""")

        config = load_config(str(config_file))
        assert config is None


class TestConfigPathHandling:
    """Tests for config path handling."""

    def test_absolute_path(self, tmp_path):
        """Load config with absolute path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test\n")

        config = load_config(str(config_file.absolute()))
        assert config["model"] == "test"

    def test_relative_paths_in_config(self, tmp_path):
        """Config can contain relative paths."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: ./models/base
adapter: ./adapters/lora
dataset: ./data/train.json
output_dir: ./output
""")

        config = load_config(str(config_file))
        assert config["model"] == "./models/base"
        assert config["dataset"] == "./data/train.json"


class TestConfigSpecialCharacters:
    """Tests for config with special characters."""

    def test_config_with_colons_in_values(self, tmp_path):
        """Config with colons in values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
prompt: "What is the time: 12:00?"
""")

        config = load_config(str(config_file))
        assert "12:00" in config["prompt"]

    def test_config_with_unicode(self, tmp_path):
        """Config with unicode characters."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
prompt: "What is 你好 in English?"
emoji: "🤖"
""")

        config = load_config(str(config_file))
        assert "你好" in config["prompt"]
        assert config["emoji"] == "🤖"
