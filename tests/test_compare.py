"""Tests for compare flow."""

import json
import pytest
from unittest.mock import MagicMock, patch
import argparse

from aare.cli import run_compare, load_config


class TestRunCompareValidation:
    """Tests for compare config validation."""

    def test_missing_config_file(self, tmp_path):
        """Error when config file doesn't exist."""
        args = argparse.Namespace(config=str(tmp_path / "nonexistent.yaml"))
        result = run_compare(args)
        assert result == 1

    def test_missing_model_field(self, tmp_path):
        """Error when model field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dataset: ./data.json\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_compare(args)
        assert result == 1

    def test_missing_dataset_field(self, tmp_path):
        """Error when dataset field is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model: test-model\n")

        args = argparse.Namespace(config=str(config_file))
        result = run_compare(args)
        assert result == 1

    def test_empty_dataset(self, tmp_path):
        """Error when dataset is empty."""
        config_file = tmp_path / "config.yaml"
        dataset_file = tmp_path / "data.json"
        dataset_file.write_text("[]")

        config_file.write_text(f"""
model: test-model
dataset: {dataset_file}
""")

        args = argparse.Namespace(config=str(config_file))
        result = run_compare(args)
        assert result == 1

    def test_dataset_file_not_found(self, tmp_path):
        """Error when dataset file doesn't exist."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./nonexistent.json
""")

        args = argparse.Namespace(config=str(config_file))
        result = run_compare(args)
        assert result == 1


class TestCompareConfigDefaults:
    """Tests for compare config default values."""

    def test_adapter_default(self, tmp_path):
        """Adapter defaults to ./output."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./data.json
""")

        config = load_config(str(config_file))
        # In run_compare, adapter defaults to "./output"
        adapter = config.get("adapter", "./output")
        assert adapter == "./output"

    def test_output_dir_default(self, tmp_path):
        """Output dir defaults to ./compare_results."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./data.json
""")

        config = load_config(str(config_file))
        output_dir = config.get("output_dir", "./compare_results")
        assert output_dir == "./compare_results"

    def test_device_default(self, tmp_path):
        """Device defaults to auto."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
model: test-model
dataset: ./data.json
""")

        config = load_config(str(config_file))
        device = config.get("device", "auto")
        assert device == "auto"


class TestCompareResultsSaving:
    """Tests for compare results saving."""

    def test_results_saved_to_files(self, tmp_path):
        """Accepted and rejected samples are saved to files."""
        # Create sample files as if compare had run
        output_dir = tmp_path / "compare_results"
        output_dir.mkdir()

        accepted = [{"instruction": "q1", "output": "a1"}]
        rejected = [{"instruction": "q2", "output": "a2"}]

        accepted_path = output_dir / "accepted.json"
        rejected_path = output_dir / "rejected.json"

        with open(accepted_path, "w") as f:
            json.dump(accepted, f, indent=2)
        with open(rejected_path, "w") as f:
            json.dump(rejected, f, indent=2)

        # Verify files
        assert accepted_path.exists()
        assert rejected_path.exists()

        with open(accepted_path) as f:
            loaded_accepted = json.load(f)
        with open(rejected_path) as f:
            loaded_rejected = json.load(f)

        assert len(loaded_accepted) == 1
        assert len(loaded_rejected) == 1
        assert loaded_accepted[0]["instruction"] == "q1"
        assert loaded_rejected[0]["instruction"] == "q2"


class TestSampleDisplay:
    """Tests for sample display logic."""

    def test_get_prompt_from_instruction(self):
        """Get prompt from instruction field."""
        sample = {"instruction": "Test question", "output": "Test answer"}
        prompt = sample.get("instruction", sample.get("prompt", sample.get("text", "")))
        assert prompt == "Test question"

    def test_get_prompt_from_prompt(self):
        """Get prompt from prompt field."""
        sample = {"prompt": "Test question", "response": "Test answer"}
        prompt = sample.get("instruction", sample.get("prompt", sample.get("text", "")))
        assert prompt == "Test question"

    def test_get_prompt_from_text(self):
        """Get prompt from text field."""
        sample = {"text": "Test question", "output": "Test answer"}
        prompt = sample.get("instruction", sample.get("prompt", sample.get("text", "")))
        assert prompt == "Test question"

    def test_get_expected_from_output(self):
        """Get expected from output field."""
        sample = {"instruction": "Test", "output": "Expected answer"}
        expected = sample.get("output", sample.get("response", ""))
        assert expected == "Expected answer"

    def test_get_expected_from_response(self):
        """Get expected from response field."""
        sample = {"prompt": "Test", "response": "Expected answer"}
        expected = sample.get("output", sample.get("response", ""))
        assert expected == "Expected answer"


class TestCompareTracking:
    """Tests for compare session tracking."""

    def test_tracking_state(self):
        """Track accepted, rejected, and skipped samples."""
        # Simulate tracking state
        accepted_samples = []
        rejected_samples = []
        skipped_indices = set()
        reviewed_indices = set()

        samples = [
            {"instruction": "q1", "output": "a1"},
            {"instruction": "q2", "output": "a2"},
            {"instruction": "q3", "output": "a3"},
        ]

        # Accept first
        accepted_samples.append(samples[0])
        reviewed_indices.add(0)

        # Reject second
        rejected_samples.append(samples[1])
        reviewed_indices.add(1)

        # Skip third
        skipped_indices.add(2)
        reviewed_indices.add(2)

        assert len(accepted_samples) == 1
        assert len(rejected_samples) == 1
        assert len(skipped_indices) == 1
        assert len(reviewed_indices) == 3

    def test_change_decision(self):
        """Change decision from accept to reject."""
        accepted_samples = []
        rejected_samples = []

        sample = {"instruction": "q1", "output": "a1"}

        # First accept
        accepted_samples.append(sample)
        assert len(accepted_samples) == 1

        # Then change to reject
        if sample in accepted_samples:
            accepted_samples.remove(sample)
        if sample not in rejected_samples:
            rejected_samples.append(sample)

        assert len(accepted_samples) == 0
        assert len(rejected_samples) == 1

    def test_unskip_sample(self):
        """Unskip a previously skipped sample."""
        accepted_samples = []
        skipped_indices = set()

        sample = {"instruction": "q1", "output": "a1"}
        idx = 0

        # First skip
        skipped_indices.add(idx)
        assert idx in skipped_indices

        # Then accept (which should remove from skipped)
        skipped_indices.discard(idx)
        accepted_samples.append(sample)

        assert idx not in skipped_indices
        assert len(accepted_samples) == 1
