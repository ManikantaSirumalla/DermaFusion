"""Step 1 logging tests."""

from __future__ import annotations

from pathlib import Path

from src.utils.logging import setup_logger


def test_setup_logger_creates_output_file(tmp_path: Path) -> None:
    """Logger should write messages into the configured log file."""
    log_file = tmp_path / "logs" / "train.log"
    logger = setup_logger(name="step1-test", log_file=log_file)
    logger.info("logger write test")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "logger write test" in content

