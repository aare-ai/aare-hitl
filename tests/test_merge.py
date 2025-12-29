"""Tests for dataset merge functionality."""

import json
import pytest

from aare.cli import merge_datasets, samples_match


class TestSamplesMatch:
    """Tests for sample matching logic."""

    def test_match_by_instruction(self):
        """Match samples by instruction field."""
        a = {"instruction": "test", "output": "a"}
        b = {"instruction": "test", "output": "b"}
        assert samples_match(a, b) is True

    def test_match_by_prompt(self):
        """Match samples by prompt field."""
        a = {"prompt": "test", "response": "a"}
        b = {"prompt": "test", "response": "b"}
        assert samples_match(a, b) is True

    def test_match_by_text(self):
        """Match samples by text field."""
        a = {"text": "test", "output": "a"}
        b = {"text": "test", "output": "b"}
        assert samples_match(a, b) is True

    def test_no_match_different_instruction(self):
        """Don't match samples with different instructions."""
        a = {"instruction": "test1", "output": "a"}
        b = {"instruction": "test2", "output": "b"}
        assert samples_match(a, b) is False

    def test_match_mixed_fields(self):
        """Match instruction to prompt."""
        a = {"instruction": "test", "output": "a"}
        b = {"prompt": "test", "response": "b"}
        # Both fall back through the chain: instruction -> prompt -> text
        assert samples_match(a, b) is True


class TestMergeDatasets:
    """Tests for dataset merging."""

    def test_remove_samples(self, tmp_path):
        """Remove samples from base dataset."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([
            {"instruction": "keep1", "output": "a"},
            {"instruction": "remove1", "output": "b"},
            {"instruction": "keep2", "output": "c"},
        ]))

        remove = tmp_path / "remove.json"
        remove.write_text(json.dumps([
            {"instruction": "remove1", "output": "different"},
        ]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), None, str(remove), str(output))

        assert result == 0

        with open(output) as f:
            data = json.load(f)

        assert len(data) == 2
        instructions = [d["instruction"] for d in data]
        assert "keep1" in instructions
        assert "keep2" in instructions
        assert "remove1" not in instructions

    def test_add_samples(self, tmp_path):
        """Add new samples to base dataset."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([
            {"instruction": "existing", "output": "a"},
        ]))

        add = tmp_path / "add.json"
        add.write_text(json.dumps([
            {"instruction": "new1", "output": "b"},
            {"instruction": "new2", "output": "c"},
        ]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), str(add), None, str(output))

        assert result == 0

        with open(output) as f:
            data = json.load(f)

        assert len(data) == 3

    def test_add_skips_duplicates(self, tmp_path):
        """Don't add samples that already exist."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([
            {"instruction": "existing", "output": "original"},
        ]))

        add = tmp_path / "add.json"
        add.write_text(json.dumps([
            {"instruction": "existing", "output": "duplicate"},
            {"instruction": "new", "output": "new"},
        ]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), str(add), None, str(output))

        assert result == 0

        with open(output) as f:
            data = json.load(f)

        assert len(data) == 2
        # Original output is preserved
        existing = next(d for d in data if d["instruction"] == "existing")
        assert existing["output"] == "original"

    def test_remove_and_add(self, tmp_path):
        """Remove and add in one operation."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([
            {"instruction": "keep", "output": "a"},
            {"instruction": "remove", "output": "b"},
        ]))

        remove = tmp_path / "remove.json"
        remove.write_text(json.dumps([
            {"instruction": "remove", "output": "x"},
        ]))

        add = tmp_path / "add.json"
        add.write_text(json.dumps([
            {"instruction": "new", "output": "c"},
        ]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), str(add), str(remove), str(output))

        assert result == 0

        with open(output) as f:
            data = json.load(f)

        assert len(data) == 2
        instructions = [d["instruction"] for d in data]
        assert "keep" in instructions
        assert "new" in instructions
        assert "remove" not in instructions

    def test_base_file_not_found(self, tmp_path):
        """Error when base file doesn't exist."""
        output = tmp_path / "output.json"
        result = merge_datasets("/nonexistent.json", None, None, str(output))
        assert result == 1

    def test_remove_file_not_found(self, tmp_path):
        """Error when remove file doesn't exist."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([{"instruction": "test", "output": "test"}]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), None, "/nonexistent.json", str(output))
        assert result == 1

    def test_add_file_not_found(self, tmp_path):
        """Error when add file doesn't exist."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([{"instruction": "test", "output": "test"}]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), "/nonexistent.json", None, str(output))
        assert result == 1

    def test_no_changes(self, tmp_path):
        """Handle case with no add or remove."""
        base = tmp_path / "base.json"
        base.write_text(json.dumps([
            {"instruction": "test", "output": "test"},
        ]))

        output = tmp_path / "output.json"
        result = merge_datasets(str(base), None, None, str(output))

        assert result == 0

        with open(output) as f:
            data = json.load(f)

        assert len(data) == 1
