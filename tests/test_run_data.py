"""Tests for run_data command routing."""

import argparse
import pytest
from unittest.mock import patch

from aare.cli import run_data


class TestRunDataRouting:
    """Tests for run_data command routing."""

    def test_validate_route(self, tmp_path):
        """Route to validate subcommand."""
        dataset = tmp_path / "data.json"
        dataset.write_text('[{"instruction": "q", "output": "a"}]')

        args = argparse.Namespace(
            data_command="validate",
            file=str(dataset),
        )
        result = run_data(args)
        assert result == 0

    def test_merge_route(self, tmp_path):
        """Route to merge subcommand."""
        base = tmp_path / "base.json"
        base.write_text('[{"instruction": "q", "output": "a"}]')
        output = tmp_path / "output.json"

        args = argparse.Namespace(
            data_command="merge",
            base=str(base),
            add=None,
            remove=None,
            output=str(output),
        )
        result = run_data(args)
        assert result == 0

    def test_import_route(self, tmp_path):
        """Route to import subcommand."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("instruction,output\nq,a\n")
        output = tmp_path / "output.json"

        args = argparse.Namespace(
            data_command="import",
            source=str(csv_file),
            output=str(output),
            instruction_col="instruction",
            output_col="output",
            sheet=None,
            api_key=None,
        )
        result = run_data(args)
        assert result == 0

    def test_unknown_command(self):
        """Unknown data command returns error."""
        args = argparse.Namespace(data_command="unknown")
        result = run_data(args)
        assert result == 1

    def test_none_command(self):
        """None data command returns error."""
        args = argparse.Namespace(data_command=None)
        result = run_data(args)
        assert result == 1


class TestValidateDatasetEdgeCases:
    """Additional tests for validate_dataset edge cases."""

    def test_validate_non_list_dataset(self, tmp_path):
        """Error when dataset is not a list."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.json"
        dataset.write_text('{"key": "value"}')

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_validate_empty_list(self, tmp_path):
        """Error when dataset is empty list."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.json"
        dataset.write_text("[]")

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_validate_item_not_dict(self, tmp_path):
        """Error when item is not a dict."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.json"
        dataset.write_text('["string item"]')

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_validate_item_missing_output(self, tmp_path):
        """Partial error when item has instruction but no output."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.json"
        dataset.write_text('[{"instruction": "test"}]')

        result = validate_dataset(str(dataset))
        assert result == 1

    def test_validate_many_issues(self, tmp_path, capsys):
        """Show truncated issues when more than 10."""
        from aare.cli import validate_dataset

        # Create 15 invalid items
        items = [{"invalid": i} for i in range(15)]
        dataset = tmp_path / "data.json"
        import json
        dataset.write_text(json.dumps(items))

        result = validate_dataset(str(dataset))
        assert result == 1

        captured = capsys.readouterr()
        assert "and 5 more" in captured.out


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_json(self, tmp_path):
        """Load JSON dataset."""
        from aare.cli import load_dataset

        dataset = tmp_path / "data.json"
        dataset.write_text('[{"instruction": "q", "output": "a"}]')

        result = load_dataset(str(dataset))
        assert len(result) == 1
        assert result[0]["instruction"] == "q"

    def test_load_yaml(self, tmp_path):
        """Load YAML dataset."""
        from aare.cli import load_dataset

        dataset = tmp_path / "data.yaml"
        dataset.write_text("- instruction: q\n  output: a\n")

        result = load_dataset(str(dataset))
        assert len(result) == 1
        assert result[0]["instruction"] == "q"

    def test_load_yml_extension(self, tmp_path):
        """Load .yml dataset."""
        from aare.cli import load_dataset

        dataset = tmp_path / "data.yml"
        dataset.write_text("- instruction: q\n  output: a\n")

        result = load_dataset(str(dataset))
        assert len(result) == 1

    def test_load_nonexistent(self, tmp_path):
        """Error when file doesn't exist."""
        from aare.cli import load_dataset

        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nonexistent.json"))


class TestValidateYaml:
    """Tests for YAML validation."""

    def test_validate_yaml_file(self, tmp_path):
        """Validate YAML file."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.yaml"
        dataset.write_text("""
- instruction: What is Python?
  output: A programming language
- instruction: What is Java?
  output: Another programming language
""")

        result = validate_dataset(str(dataset))
        assert result == 0

    def test_validate_yml_extension(self, tmp_path):
        """Validate .yml file."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.yml"
        dataset.write_text("""
- instruction: test
  output: answer
""")

        result = validate_dataset(str(dataset))
        assert result == 0

    def test_validate_yaml_parse_error(self, tmp_path):
        """Error on invalid YAML."""
        from aare.cli import validate_dataset

        dataset = tmp_path / "data.yaml"
        dataset.write_text("invalid: yaml: content:")

        result = validate_dataset(str(dataset))
        assert result == 1
