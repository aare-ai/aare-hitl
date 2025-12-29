"""Tests for dataset validation functionality."""

import json
import pytest

from aare.cli import validate_dataset


class TestValidateDataset:
    """Tests for dataset validation."""

    def test_valid_json_dataset(self, tmp_path):
        """Validate a properly formatted JSON dataset."""
        dataset = tmp_path / "valid.json"
        dataset.write_text(json.dumps([
            {"instruction": "Question 1", "output": "Answer 1"},
            {"instruction": "Question 2", "output": "Answer 2"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 0

    def test_valid_with_prompt_response(self, tmp_path):
        """Accept alternative field names: prompt/response."""
        dataset = tmp_path / "valid.json"
        dataset.write_text(json.dumps([
            {"prompt": "Question 1", "response": "Answer 1"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 0

    def test_valid_with_text_field(self, tmp_path):
        """Accept text field for raw samples."""
        dataset = tmp_path / "valid.json"
        dataset.write_text(json.dumps([
            {"text": "Some raw text", "output": "Response"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 0

    def test_invalid_not_array(self, tmp_path):
        """Reject dataset that's not an array."""
        dataset = tmp_path / "invalid.json"
        dataset.write_text(json.dumps({"instruction": "test", "output": "test"}))

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_invalid_empty_array(self, tmp_path):
        """Reject empty dataset."""
        dataset = tmp_path / "empty.json"
        dataset.write_text("[]")

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_invalid_missing_instruction(self, tmp_path):
        """Report items missing instruction field."""
        dataset = tmp_path / "invalid.json"
        dataset.write_text(json.dumps([
            {"output": "Answer without question"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_invalid_missing_output(self, tmp_path):
        """Report items missing output field."""
        dataset = tmp_path / "invalid.json"
        dataset.write_text(json.dumps([
            {"instruction": "Question without answer"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_file_not_found(self):
        """Error when file doesn't exist."""
        result = validate_dataset("/nonexistent/file.json")
        assert result == 1

    def test_invalid_json(self, tmp_path):
        """Error when file contains invalid JSON."""
        dataset = tmp_path / "invalid.json"
        dataset.write_text("not valid json {")

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_mixed_valid_invalid(self, tmp_path):
        """Report when some items are valid and some invalid."""
        dataset = tmp_path / "mixed.json"
        dataset.write_text(json.dumps([
            {"instruction": "Valid", "output": "Valid"},
            {"instruction": "Missing output"},
            {"output": "Missing instruction"},
        ]))

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_item_not_dict(self, tmp_path):
        """Report when array items are not dicts."""
        dataset = tmp_path / "invalid.json"
        dataset.write_text(json.dumps([
            "just a string",
            123,
        ]))

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_yaml_dataset(self, tmp_path):
        """Validate YAML format dataset."""
        dataset = tmp_path / "valid.yaml"
        dataset.write_text("""
- instruction: Question 1
  output: Answer 1
- instruction: Question 2
  output: Answer 2
""")

        result = validate_dataset(str(dataset))
        assert result == 0
