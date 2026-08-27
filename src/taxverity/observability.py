import json
import logging
import re
import sys
from typing import Any

from taxverity.config import Settings

ROOT_LOGGER_NAME = "taxverity"

PAN_MASK = "[PAN-REDACTED]"
AADHAAR_MASK = "[AADHAAR-REDACTED]"
ACCOUNT_MASK = "[ACCOUNT-REDACTED]"

_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
# Runs shorter than 9 digits are left alone so statutory markers, page numbers
# and amounts survive; \b keeps hex digests (digits bounded by letters) intact.
_ACCOUNT = re.compile(r"\b\d{9,18}\b")


def redact(text: str) -> str:
    """Mask PAN, Aadhaar and account numbers before any egress from the process."""
    text = _PAN.sub(PAN_MASK, text)
    # Aadhaar first: a bare 12-digit run also satisfies the account pattern.
    text = _AADHAAR.sub(AADHAAR_MASK, text)
    return _ACCOUNT.sub(ACCOUNT_MASK, text)


def _redact_value(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


class RedactingFilter(logging.Filter):
    # Applied to the handler rather than left to call sites: a masking rule that
    # depends on whoever wrote the log line is a convention, not a control.
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, dict):
            record.args = {k: _redact_value(v) for k, v in record.args.items()}
        elif record.args:
            record.args = tuple(_redact_value(a) for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    settings = settings or Settings()
    logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Idempotent: repeated calls (scripts importing scripts, pytest re-entry)
    # must not stack duplicate handlers onto the same logger.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    handler.addFilter(RedactingFilter())

    logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    # Diagnostics must not surface through whatever the host application has
    # attached to the root logger.
    logger.propagate = False
    return logger
