"""Tests for CLI argument parsing and basic command structure."""

import subprocess
import sys


class TestCliBasics:
    """Tests for basic CLI functionality."""

    def test_version(self):
        """Show version with --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self):
        """Show help with no arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout
        assert "compare" in result.stdout
        assert "generate" in result.stdout
        assert "data" in result.stdout

    def test_data_help(self):
        """Show data subcommand help."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "validate" in result.stdout
        assert "import" in result.stdout
        assert "merge" in result.stdout

    def test_train_requires_config(self):
        """Train command requires --config."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "train"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "config" in result.stderr.lower()

    def test_compare_requires_config(self):
        """Compare command requires --config."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "compare"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_generate_requires_config(self):
        """Generate command requires --config."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "generate"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_data_validate_requires_file(self):
        """Data validate requires file argument."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "validate"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_data_import_requires_source_and_output(self):
        """Data import requires source and output arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "import"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_data_merge_requires_base_and_output(self):
        """Data merge requires base and output arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "data", "merge"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_debug_flag(self):
        """Debug flag is accepted."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "--debug", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_log_format_text(self):
        """Log format text is accepted."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "--log-format", "text", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_log_format_json(self):
        """Log format json is accepted."""
        result = subprocess.run(
            [sys.executable, "-m", "aare.cli", "--log-format", "json", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
