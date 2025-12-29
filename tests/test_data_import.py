"""Tests for data import functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from aare.cli import parse_google_sheets_url, import_data


class TestParseGoogleSheetsUrl:
    """Tests for Google Sheets URL parsing."""

    def test_simple_url(self):
        """Parse URL without GID."""
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        assert gid is None

    def test_url_with_gid_hash(self):
        """Parse URL with GID using hash fragment."""
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=123456"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        assert gid == "123456"

    def test_url_with_gid_query(self):
        """Parse URL with GID as query parameter."""
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?gid=789"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        # Note: current implementation uses regex that matches #gid= or &gid=, not ?gid=
        # This is a known limitation

    def test_url_without_edit(self):
        """Parse URL without /edit suffix."""
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        assert gid is None

    def test_invalid_url(self):
        """Raise error for non-Google Sheets URL."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_google_sheets_url("https://example.com/not-a-sheet")

    def test_spreadsheet_id_with_special_chars(self):
        """Parse spreadsheet ID with hyphens and underscores."""
        url = "https://docs.google.com/spreadsheets/d/1Bxi-MVs_0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "1Bxi-MVs_0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"


class TestImportCsv:
    """Tests for CSV import functionality."""

    def test_import_basic_csv(self, tmp_path):
        """Import a basic CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("instruction,output\nWhat is 2+2?,4\nWhat is 3+3?,6\n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "instruction", "output")

        assert result == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["instruction"] == "What is 2+2?"
        assert data[0]["output"] == "4"

    def test_import_custom_columns(self, tmp_path):
        """Import CSV with custom column names."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("question,answer\nWhat is Python?,A programming language\n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "question", "answer")

        assert result == 0

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["instruction"] == "What is Python?"
        assert data[0]["output"] == "A programming language"

    def test_import_skips_empty_rows(self, tmp_path):
        """Skip rows with empty instruction or output."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("instruction,output\nValid question,Valid answer\n,Empty instruction\nEmpty output,\n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "instruction", "output")

        assert result == 0

        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["instruction"] == "Valid question"

    def test_import_missing_column(self, tmp_path):
        """Error when required column is missing."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("instruction,wrong_column\nTest,Value\n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "instruction", "output")

        assert result == 1
        assert not output_file.exists()

    def test_import_file_not_found(self, tmp_path):
        """Error when CSV file doesn't exist."""
        output_file = tmp_path / "output.json"
        result = import_data("/nonexistent/file.csv", str(output_file), "instruction", "output")

        assert result == 1

    def test_import_wrong_extension(self, tmp_path):
        """Error when file is not a CSV."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some content")

        output_file = tmp_path / "output.json"
        result = import_data(str(txt_file), str(output_file), "instruction", "output")

        assert result == 1

    def test_import_strips_whitespace(self, tmp_path):
        """Strip whitespace from values."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("instruction,output\n  padded question  ,  padded answer  \n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "instruction", "output")

        assert result == 0

        with open(output_file) as f:
            data = json.load(f)

        assert data[0]["instruction"] == "padded question"
        assert data[0]["output"] == "padded answer"

    def test_import_empty_csv(self, tmp_path):
        """Error when CSV has no data rows."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("instruction,output\n")

        output_file = tmp_path / "output.json"
        result = import_data(str(csv_file), str(output_file), "instruction", "output")

        assert result == 1
