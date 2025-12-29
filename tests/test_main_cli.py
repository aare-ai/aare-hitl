"""Tests for main CLI entry point and argument parsing."""

import subprocess
import sys
import pytest


class TestMainEntryPoint:
    """Tests for CLI main entry point."""

    def test_no_arguments_shows_help(self):
        """No arguments shows help and returns 0."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout or "usage" in result.stdout.lower()

    def test_invalid_command(self):
        """Invalid command returns error."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "invalid_command"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_train_help(self):
        """Train command help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_compare_help(self):
        """Compare command help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "compare", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_generate_help(self):
        """Generate command help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "generate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "config" in result.stdout.lower()

    def test_data_validate_help(self):
        """Data validate help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "file" in result.stdout.lower()

    def test_data_merge_help(self):
        """Data merge help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "merge", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "add" in result.stdout.lower()
        assert "remove" in result.stdout.lower()

    def test_data_import_help(self):
        """Data import help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "import", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "source" in result.stdout.lower()
        assert "output" in result.stdout.lower()


class TestCLIOutputFormat:
    """Tests for CLI output formatting."""

    def test_validate_output_format(self, tmp_path):
        """Validate command outputs proper format."""
        import json
        dataset = tmp_path / "data.json"
        dataset.write_text(json.dumps([
            {"instruction": "q1", "output": "a1"},
            {"instruction": "q2", "output": "a2"},
        ]))

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate", str(dataset)],
            capture_output=True,
            text=True,
        )

        assert "Dataset:" in result.stdout
        assert "Total samples:" in result.stdout
        assert "Valid samples:" in result.stdout

    def test_merge_output_format(self, tmp_path):
        """Merge command outputs proper format."""
        import json
        base = tmp_path / "base.json"
        base.write_text(json.dumps([{"instruction": "q", "output": "a"}]))
        output = tmp_path / "output.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "merge",
                str(base), "-o", str(output),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Merge complete:" in result.stdout

    def test_import_output_format(self, tmp_path):
        """Import command outputs proper format."""
        csv = tmp_path / "data.csv"
        csv.write_text("instruction,output\nq,a\n")
        output = tmp_path / "output.json"

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "data", "import",
                str(csv), "-o", str(output),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Imported" in result.stdout or output.exists()


class TestCLIGlobalFlags:
    """Tests for CLI global flags."""

    def test_debug_with_validate(self, tmp_path):
        """Debug flag works with validate."""
        import json
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

    def test_log_format_json_with_validate(self, tmp_path):
        """JSON log format works with validate."""
        import json
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

    def test_log_format_text_with_validate(self, tmp_path):
        """Text log format works with validate."""
        import json
        dataset = tmp_path / "data.json"
        dataset.write_text(json.dumps([{"instruction": "q", "output": "a"}]))

        result = subprocess.run(
            [
                sys.executable, "-m", "aare.cli",
                "--log-format", "text",
                "data", "validate",
                str(dataset),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0


class TestCLIErrorMessages:
    """Tests for CLI error messages."""

    def test_train_missing_config_error(self):
        """Train without config shows error about missing argument."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "train"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "config" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_compare_missing_config_error(self):
        """Compare without config shows error about missing argument."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "compare"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    def test_generate_missing_config_error(self):
        """Generate without config shows error about missing argument."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "generate"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    def test_data_import_missing_source_error(self):
        """Data import without source shows error."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "import"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0

    def test_data_merge_missing_output_error(self, tmp_path):
        """Data merge without output shows error."""
        base = tmp_path / "base.json"
        base.write_text('[]')

        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "merge", str(base)],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "output" in result.stderr.lower() or "required" in result.stderr.lower()
