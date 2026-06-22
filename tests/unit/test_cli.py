"""Unit tests for the CLI entry point."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from balloon_quantity_analyzer.cli import main, _load_config
from balloon_quantity_analyzer.models import (
    AnalyzerConfig,
    InvalidConfigurationError,
    UnreadableFileError,
    UnsupportedFormatError,
)


class TestBuildParser:
    """Tests for argument parsing via main()."""

    def test_missing_file_path_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_invalid_format_choice_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["drawing.pdf", "--format", "xml"])
        assert exc_info.value.code != 0


class TestLoadConfig:
    """Tests for _load_config."""

    def test_load_full_config(self, tmp_path):
        cfg = {
            "proximity_radius": 75.0,
            "confidence_threshold": 0.8,
            "custom_multiplier_phrases": {"EACH SIDE": 2},
        }
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(cfg))

        result = _load_config(str(cfg_file))
        assert result.proximity_radius == 75.0
        assert result.confidence_threshold == 0.8
        assert result.custom_multiplier_phrases == {"EACH SIDE": 2}

    def test_load_partial_config_uses_defaults(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text("{}")

        result = _load_config(str(cfg_file))
        assert result.proximity_radius == 50.0
        assert result.confidence_threshold == 0.5
        assert result.custom_multiplier_phrases == {}

    def test_load_invalid_json_raises(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("not json")

        with pytest.raises(json.JSONDecodeError):
            _load_config(str(cfg_file))

    def test_load_missing_file_raises(self):
        with pytest.raises(OSError):
            _load_config("/nonexistent/config.json")


class TestMainIntegration:
    """Tests for the main() function end-to-end behaviour."""

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_json_output_to_stdout(self, mock_analyzer_cls, capsys):
        instance = mock_analyzer_cls.return_value
        instance.analyze_to_json.return_value = '{"tally": {}}'

        main(["drawing.pdf"])

        instance.analyze_to_json.assert_called_once_with("drawing.pdf")
        captured = capsys.readouterr()
        assert '{"tally": {}}' in captured.out

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_tabular_output_to_stdout(self, mock_analyzer_cls, capsys):
        instance = mock_analyzer_cls.return_value
        instance.analyze_to_tabular.return_value = "TALLY\n-----"

        main(["drawing.pdf", "--format", "tabular"])

        instance.analyze_to_tabular.assert_called_once_with("drawing.pdf")
        captured = capsys.readouterr()
        assert "TALLY" in captured.out

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_output_to_file(self, mock_analyzer_cls, tmp_path):
        instance = mock_analyzer_cls.return_value
        instance.analyze_to_json.return_value = '{"tally": {"1": 3}}'

        out_file = tmp_path / "result.json"
        main(["drawing.pdf", "--output", str(out_file)])

        assert out_file.read_text() == '{"tally": {"1": 3}}'

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_config_file_loaded(self, mock_analyzer_cls, tmp_path, capsys):
        cfg = {"proximity_radius": 100.0, "confidence_threshold": 0.9}
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps(cfg))

        instance = mock_analyzer_cls.return_value
        instance.analyze_to_json.return_value = "{}"

        main(["drawing.pdf", "--config", str(cfg_file)])

        # Verify the analyzer was created with a config
        call_kwargs = mock_analyzer_cls.call_args
        config_arg = call_kwargs.kwargs.get("config") or call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("config")
        assert config_arg is not None
        assert config_arg.proximity_radius == 100.0

    def test_bad_config_file_exits(self, tmp_path):
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("not json")

        with pytest.raises(SystemExit) as exc_info:
            main(["drawing.pdf", "--config", str(cfg_file)])
        assert exc_info.value.code == 1

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_unsupported_format_error_exits(self, mock_analyzer_cls):
        instance = mock_analyzer_cls.return_value
        instance.analyze_to_json.side_effect = UnsupportedFormatError("bmp")

        with pytest.raises(SystemExit) as exc_info:
            main(["drawing.bmp"])
        assert exc_info.value.code == 1

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_unreadable_file_error_exits(self, mock_analyzer_cls):
        instance = mock_analyzer_cls.return_value
        instance.analyze_to_json.side_effect = UnreadableFileError("corrupt")

        with pytest.raises(SystemExit) as exc_info:
            main(["drawing.pdf"])
        assert exc_info.value.code == 1

    @patch("balloon_quantity_analyzer.cli.BalloonAnalyzer")
    def test_invalid_config_error_exits(self, mock_analyzer_cls):
        mock_analyzer_cls.side_effect = InvalidConfigurationError("bad radius")

        with pytest.raises(SystemExit) as exc_info:
            main(["drawing.pdf"])
        assert exc_info.value.code == 1
