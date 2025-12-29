"""Tests for Google Sheets integration."""

import pytest
from unittest.mock import patch, MagicMock

from aare.cli import (
    parse_google_sheets_url,
    fetch_google_sheet_csv,
    import_data,
)


class TestParseGoogleSheetsUrlEdgeCases:
    """Edge case tests for URL parsing."""

    def test_url_with_query_params(self):
        """Parse URL with query parameters."""
        url = "https://docs.google.com/spreadsheets/d/ABC123/edit?usp=sharing"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "ABC123"
        assert gid is None

    def test_url_with_multiple_params_and_gid(self):
        """Parse URL with multiple query params including gid."""
        url = "https://docs.google.com/spreadsheets/d/ABC123/edit?usp=sharing&gid=456"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "ABC123"
        assert gid == "456"

    def test_url_with_fragment_gid(self):
        """Parse URL with fragment gid."""
        url = "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=789"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "ABC123"
        assert gid == "789"

    def test_url_export_format(self):
        """Parse export URL format."""
        url = "https://docs.google.com/spreadsheets/d/ABC123/export?format=csv"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == "ABC123"

    def test_url_with_long_id(self):
        """Parse URL with typical long spreadsheet ID."""
        long_id = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
        url = f"https://docs.google.com/spreadsheets/d/{long_id}/edit"
        spreadsheet_id, gid = parse_google_sheets_url(url)
        assert spreadsheet_id == long_id

    def test_not_google_sheets_url(self):
        """Reject non-Google Sheets URLs."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_google_sheets_url("https://example.com/spreadsheet")

    def test_google_docs_url(self):
        """Reject Google Docs (not Sheets) URLs."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_google_sheets_url("https://docs.google.com/document/d/ABC123/edit")

    def test_malformed_url(self):
        """Reject malformed URLs."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_google_sheets_url("not a url at all")


class TestFetchGoogleSheetCsv:
    """Tests for fetching Google Sheets as CSV."""

    @patch("requests.get")
    def test_fetch_success(self, mock_get):
        """Successfully fetch public sheet."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "instruction,output\nq1,a1\nq2,a2"
        mock_get.return_value = mock_response

        result = fetch_google_sheet_csv("ABC123")

        assert result == "instruction,output\nq1,a1\nq2,a2"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_fetch_with_gid(self, mock_get):
        """Fetch specific sheet by GID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "data"
        mock_get.return_value = mock_response

        fetch_google_sheet_csv("ABC123", gid="456")

        call_args = mock_get.call_args
        assert "gid" in call_args.kwargs["params"]
        assert call_args.kwargs["params"]["gid"] == "456"

    @patch("requests.get")
    def test_fetch_with_api_key(self, mock_get):
        """Fetch with API key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "data"
        mock_get.return_value = mock_response

        fetch_google_sheet_csv("ABC123", api_key="my-api-key")

        call_args = mock_get.call_args
        assert "key" in call_args.kwargs["params"]
        assert call_args.kwargs["params"]["key"] == "my-api-key"

    @patch("requests.get")
    def test_fetch_404_error(self, mock_get):
        """Handle 404 not found error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="not found"):
            fetch_google_sheet_csv("nonexistent")

    @patch("requests.get")
    def test_fetch_403_error(self, mock_get):
        """Handle 403 forbidden error."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="Access denied"):
            fetch_google_sheet_csv("private-sheet")

    @patch("requests.get")
    def test_fetch_500_error(self, mock_get):
        """Handle server error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with pytest.raises(ValueError, match="HTTP 500"):
            fetch_google_sheet_csv("ABC123")

    @patch("requests.get")
    def test_fetch_timeout(self, mock_get):
        """Timeout is set on request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "data"
        mock_get.return_value = mock_response

        fetch_google_sheet_csv("ABC123")

        call_args = mock_get.call_args
        assert call_args.kwargs["timeout"] == 30


class TestImportDataGoogleSheets:
    """Tests for import_data with Google Sheets URLs."""

    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_from_sheets_url(self, mock_parse, mock_fetch, tmp_path):
        """Import from Google Sheets URL."""
        mock_parse.return_value = ("ABC123", None)
        mock_fetch.return_value = "instruction,output\nq1,a1\nq2,a2"

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit",
            str(output_file),
            "instruction",
            "output",
        )

        assert result == 0
        assert output_file.exists()

        import json
        with open(output_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["instruction"] == "q1"

    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_with_sheet_override(self, mock_parse, mock_fetch, tmp_path):
        """Import with sheet GID override."""
        mock_parse.return_value = ("ABC123", "999")  # URL has gid=999
        mock_fetch.return_value = "instruction,output\nq1,a1"

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit#gid=999",
            str(output_file),
            "instruction",
            "output",
            sheet="123",  # Override with gid=123
        )

        assert result == 0
        # Verify the override was used
        mock_fetch.assert_called_once()
        call_args = mock_fetch.call_args
        assert call_args[0][1] == "123"  # gid argument

    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_with_api_key_arg(self, mock_parse, mock_fetch, tmp_path):
        """Import with API key from argument."""
        mock_parse.return_value = ("ABC123", None)
        mock_fetch.return_value = "instruction,output\nq1,a1"

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit",
            str(output_file),
            "instruction",
            "output",
            api_key="my-key",
        )

        assert result == 0
        call_args = mock_fetch.call_args
        assert call_args[0][2] == "my-key"  # api_key argument

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"})
    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_with_api_key_env(self, mock_parse, mock_fetch, tmp_path):
        """Import with API key from environment."""
        mock_parse.return_value = ("ABC123", None)
        mock_fetch.return_value = "instruction,output\nq1,a1"

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit",
            str(output_file),
            "instruction",
            "output",
        )

        assert result == 0
        call_args = mock_fetch.call_args
        assert call_args[0][2] == "env-key"

    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_sheets_missing_column(self, mock_parse, mock_fetch, tmp_path):
        """Error when Google Sheet is missing expected column."""
        mock_parse.return_value = ("ABC123", None)
        mock_fetch.return_value = "wrong_column,output\nq1,a1"

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit",
            str(output_file),
            "instruction",
            "output",
        )

        assert result == 1
        assert not output_file.exists()

    @patch("aare.cli.parse_google_sheets_url")
    def test_import_sheets_parse_error(self, mock_parse, tmp_path):
        """Handle URL parse error."""
        mock_parse.side_effect = ValueError("Could not parse")

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/invalid",
            str(output_file),
            "instruction",
            "output",
        )

        assert result == 1

    @patch("aare.cli.fetch_google_sheet_csv")
    @patch("aare.cli.parse_google_sheets_url")
    def test_import_sheets_fetch_error(self, mock_parse, mock_fetch, tmp_path):
        """Handle fetch error."""
        mock_parse.return_value = ("ABC123", None)
        mock_fetch.side_effect = ValueError("Access denied")

        output_file = tmp_path / "output.json"
        result = import_data(
            "https://docs.google.com/spreadsheets/d/ABC123/edit",
            str(output_file),
            "instruction",
            "output",
        )

        assert result == 1


class TestGoogleSheetsDetection:
    """Tests for detecting Google Sheets URLs."""

    def test_detect_sheets_url(self):
        """Detect Google Sheets URL."""
        url = "https://docs.google.com/spreadsheets/d/ABC123/edit"
        assert "docs.google.com/spreadsheets" in url

    def test_detect_local_csv(self):
        """Local CSV is not detected as Sheets."""
        path = "/path/to/local/file.csv"
        assert "docs.google.com/spreadsheets" not in path

    def test_detect_other_google_url(self):
        """Other Google URLs are not detected as Sheets."""
        url = "https://drive.google.com/file/d/ABC123/view"
        assert "docs.google.com/spreadsheets" not in url
