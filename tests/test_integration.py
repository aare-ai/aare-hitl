"""Integration tests for end-to-end workflows."""

import json
import subprocess
import sys
import pytest
from pathlib import Path


class TestDataWorkflow:
    """Integration tests for data pipeline workflow."""

    def test_import_validate_merge_workflow(self, tmp_path):
        """Complete workflow: import CSV -> validate -> merge."""
        # Step 1: Create CSV
        csv_file = tmp_path / "input.csv"
        csv_file.write_text("""instruction,output
What is Python?,A programming language
What is Java?,Another programming language
What is Rust?,A systems programming language
""")

        # Step 2: Import to JSON
        output_json = tmp_path / "dataset.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "import",
                str(csv_file),
                "-o", str(output_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert output_json.exists()

        with open(output_json) as f:
            data = json.load(f)
        assert len(data) == 3

        # Step 3: Validate
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "validate",
                str(output_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

        # Step 4: Create rejected samples
        rejected_json = tmp_path / "rejected.json"
        rejected_json.write_text(json.dumps([
            {"instruction": "What is Java?", "output": "Another programming language"}
        ]))

        # Step 5: Merge (remove rejected)
        cleaned_json = tmp_path / "cleaned.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "merge",
                str(output_json),
                "--remove", str(rejected_json),
                "-o", str(cleaned_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert cleaned_json.exists()

        with open(cleaned_json) as f:
            cleaned_data = json.load(f)
        assert len(cleaned_data) == 2
        instructions = [d["instruction"] for d in cleaned_data]
        assert "What is Java?" not in instructions

    def test_import_with_custom_columns(self, tmp_path):
        """Import CSV with custom column names."""
        csv_file = tmp_path / "custom.csv"
        csv_file.write_text("""question,answer,extra
What is 2+2?,4,math
What is 3+3?,6,math
""")

        output_json = tmp_path / "output.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "import",
                str(csv_file),
                "-o", str(output_json),
                "--instruction-col", "question",
                "--output-col", "answer",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_json) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["instruction"] == "What is 2+2?"
        assert data[0]["output"] == "4"

    def test_merge_add_and_remove(self, tmp_path):
        """Merge with both add and remove."""
        # Base dataset
        base_json = tmp_path / "base.json"
        base_json.write_text(json.dumps([
            {"instruction": "q1", "output": "a1"},
            {"instruction": "q2", "output": "a2"},
            {"instruction": "q3", "output": "a3"},
        ]))

        # Remove these
        remove_json = tmp_path / "remove.json"
        remove_json.write_text(json.dumps([
            {"instruction": "q2", "output": "a2"},
        ]))

        # Add these
        add_json = tmp_path / "add.json"
        add_json.write_text(json.dumps([
            {"instruction": "q4", "output": "a4"},
            {"instruction": "q5", "output": "a5"},
        ]))

        output_json = tmp_path / "output.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "merge",
                str(base_json),
                "--remove", str(remove_json),
                "--add", str(add_json),
                "-o", str(output_json),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        with open(output_json) as f:
            data = json.load(f)

        assert len(data) == 4  # 3 - 1 + 2 = 4
        instructions = [d["instruction"] for d in data]
        assert "q1" in instructions
        assert "q2" not in instructions
        assert "q3" in instructions
        assert "q4" in instructions
        assert "q5" in instructions


class TestValidationWorkflow:
    """Integration tests for validation workflow."""

    def test_validate_good_json(self, tmp_path):
        """Validate a properly formatted JSON dataset."""
        dataset = tmp_path / "good.json"
        dataset.write_text(json.dumps([
            {"instruction": "q1", "output": "a1"},
            {"instruction": "q2", "output": "a2"},
        ]))

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", str(dataset)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "valid" in result.stdout.lower()
        assert "2" in result.stdout  # 2 samples

    def test_validate_good_yaml(self, tmp_path):
        """Validate a properly formatted YAML dataset."""
        dataset = tmp_path / "good.yaml"
        dataset.write_text("""
- instruction: What is Python?
  output: A programming language
- instruction: What is Java?
  output: Another programming language
""")

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", str(dataset)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

    def test_validate_bad_json(self, tmp_path):
        """Validate detects invalid JSON dataset."""
        dataset = tmp_path / "bad.json"
        dataset.write_text(json.dumps([
            {"instruction": "missing output"},
            {"output": "missing instruction"},
        ]))

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", str(dataset)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "issue" in result.stdout.lower()

    def test_validate_mixed_formats(self, tmp_path):
        """Validate accepts alternative field names."""
        dataset = tmp_path / "mixed.json"
        dataset.write_text(json.dumps([
            {"instruction": "q1", "output": "a1"},
            {"prompt": "q2", "response": "a2"},
            {"text": "q3", "output": "a3"},
        ]))

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", str(dataset)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "3" in result.stdout  # 3 valid samples


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_import_nonexistent_file(self, tmp_path):
        """Import nonexistent file returns error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "import",
                "/nonexistent/file.csv",
                "-o", str(tmp_path / "output.json"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1

    def test_validate_nonexistent_file(self):
        """Validate nonexistent file returns error."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", "/nonexistent/file.json"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1

    def test_merge_nonexistent_base(self, tmp_path):
        """Merge with nonexistent base returns error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "merge",
                "/nonexistent/base.json",
                "-o", str(tmp_path / "output.json"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1

    def test_import_wrong_columns(self, tmp_path):
        """Import with wrong column names returns error."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2\n")

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "import",
                str(csv_file),
                "-o", str(tmp_path / "output.json"),
                "--instruction-col", "nonexistent",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()


class TestDebugMode:
    """Integration tests for debug mode."""

    def test_debug_flag_accepted(self, tmp_path):
        """Debug flag is accepted without error."""
        dataset = tmp_path / "data.json"
        dataset.write_text(json.dumps([{"instruction": "q", "output": "a"}]))

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "--debug",
                "data", "validate",
                str(dataset),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_log_format_json(self, tmp_path):
        """JSON log format is accepted."""
        dataset = tmp_path / "data.json"
        dataset.write_text(json.dumps([{"instruction": "q", "output": "a"}]))

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "--log-format", "json",
                "data", "validate",
                str(dataset),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
