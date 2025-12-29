"""Tests for data import edge cases."""

import json
import pytest
from pathlib import Path

from aare.cli import import_data


class TestImportDataEdgeCases:
    """Tests for import_data edge cases."""

    def test_import_non_csv_file(self, tmp_path):
        """Error when importing non-CSV file."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("instruction,output\nq,a\n")
        output = tmp_path / "output.json"

        result = import_data(
            str(txt_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 1

    def test_import_empty_csv(self, tmp_path):
        """Error when CSV is empty (no header)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 1

    def test_import_header_only_csv(self, tmp_path):
        """Error when CSV has only header, no data."""
        csv_file = tmp_path / "header_only.csv"
        csv_file.write_text("instruction,output\n")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 1

    def test_import_missing_instruction_column(self, tmp_path):
        """Error when instruction column doesn't exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("question,answer\nq,a\n")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",  # This column doesn't exist
            "answer",
            None,
            None,
        )
        assert result == 1

    def test_import_missing_output_column(self, tmp_path):
        """Error when output column doesn't exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("instruction,response\nq,a\n")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",  # This column doesn't exist
            None,
            None,
        )
        assert result == 1

    def test_import_skips_empty_rows(self, tmp_path):
        """Empty rows are skipped."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("""instruction,output
q1,a1
,
q2,
,a3
q3,a3
""")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 0

        with open(output) as f:
            data = json.load(f)
        # Only q1,a1 and q3,a3 should be imported (rows with both fields)
        assert len(data) == 2
        assert data[0]["instruction"] == "q1"
        assert data[1]["instruction"] == "q3"

    def test_import_strips_whitespace(self, tmp_path):
        """Whitespace is stripped from values."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("""instruction,output
  q1  ,  a1
q2,a2
""")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 0

        with open(output) as f:
            data = json.load(f)
        assert data[0]["instruction"] == "q1"
        assert data[0]["output"] == "a1"

    def test_import_nonexistent_file(self, tmp_path):
        """Error when file doesn't exist."""
        output = tmp_path / "output.json"

        result = import_data(
            str(tmp_path / "nonexistent.csv"),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 1

    def test_import_extra_columns_ignored(self, tmp_path):
        """Extra columns in CSV are ignored."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("""instruction,output,extra,more
q1,a1,x,y
q2,a2,z,w
""")
        output = tmp_path / "output.json"

        result = import_data(
            str(csv_file),
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 0

        with open(output) as f:
            data = json.load(f)
        assert len(data) == 2
        # Only instruction and output should be in the result
        assert set(data[0].keys()) == {"instruction", "output"}


class TestImportGoogleSheetsDetection:
    """Tests for Google Sheets URL detection in import."""

    def test_detect_google_sheets_url(self, tmp_path):
        """Detect Google Sheets URL format - returns error for non-existent sheet."""
        output = tmp_path / "output.json"

        # This will fail trying to fetch, but should be detected as Google Sheets
        # and return 1 (error code)
        result = import_data(
            "https://docs.google.com/spreadsheets/d/abc123/edit",
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        assert result == 1

    def test_detect_regular_url_not_google(self, tmp_path):
        """Regular URLs are not treated as Google Sheets."""
        output = tmp_path / "output.json"

        # This should try to treat it as a local file path
        result = import_data(
            "https://example.com/data.csv",
            str(output),
            "instruction",
            "output",
            None,
            None,
        )
        # Will fail because it's not a valid local path
        assert result == 1


class TestMergeOutputError:
    """Tests for merge output error handling."""

    def test_merge_output_write_error(self, tmp_path, monkeypatch):
        """Handle error when writing output file."""
        from aare.cli import merge_datasets

        base = tmp_path / "base.json"
        base.write_text('[{"instruction": "q", "output": "a"}]')

        # Try to write to a non-existent directory
        result = merge_datasets(
            str(base),
            None,
            None,
            "/nonexistent/dir/output.json"
        )
        assert result == 1
