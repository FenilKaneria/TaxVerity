import json
import logging

import pytest

from taxverity.config import Settings
from taxverity.observability import (
    AADHAAR_MASK,
    ACCOUNT_MASK,
    PAN_MASK,
    ROOT_LOGGER_NAME,
    JsonFormatter,
    RedactingFilter,
    configure_logging,
    get_logger,
    redact,
)

PAN = "ABCDE1234F"
AADHAAR = "1234 5678 9012"
ACCOUNT = "123456789012345"


@pytest.fixture
def root_logger():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield logger
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def emit(logger, stream_holder, *args, level=logging.INFO, **kwargs):
    logger.log(level, *args, **kwargs)
    return stream_holder.getvalue()


@pytest.mark.parametrize(
    ("text", "mask"),
    [
        (f"user PAN is {PAN}", PAN_MASK),
        (f"aadhaar {AADHAAR}", AADHAAR_MASK),
        ("aadhaar 123456789012", AADHAAR_MASK),
        (f"account {ACCOUNT}", ACCOUNT_MASK),
    ],
)
def test_identifiers_are_masked(text, mask):
    masked = redact(text)
    assert mask in masked
    assert PAN not in masked
    assert "123456789012" not in masked


@pytest.mark.parametrize(
    "text",
    [
        "deduction under section 80C",
        "section 354A applies",
        "the nature described in 2(5)(b)(ii)",
        "Schedule XVI(1)(a)",
        "salary of 1400000 for the year 2025",
        "page 246, top 63.07",
    ],
)
def test_statutory_tokens_and_ordinary_numbers_survive(text):
    # A redactor that eats citations is worse than none: every downstream
    # diagnostic in this project is written in this vocabulary.
    assert redact(text) == text


def test_a_sha256_digest_is_not_mistaken_for_an_account_number():
    digest = "9f2c" + "0" * 56
    assert redact(digest) == digest


def test_the_filter_masks_the_message_and_its_arguments():
    record = logging.LogRecord(
        "taxverity.test", logging.INFO, __file__, 1, "pan %s", (PAN,), None
    )
    RedactingFilter().filter(record)
    assert PAN not in record.getMessage()
    assert record.args == (PAN_MASK,)


def test_the_filter_masks_dict_arguments():
    # Two keys deliberately: logging unwraps a single-key mapping differently.
    args = {"pan": PAN, "section": "80C"}
    record = logging.LogRecord(
        "taxverity.test", logging.INFO, __file__, 1, "pan %(pan)s in %(section)s",
        args, None,
    )
    RedactingFilter().filter(record)
    assert record.args == {"pan": PAN_MASK, "section": "80C"}
    assert PAN not in record.getMessage()


def test_non_string_arguments_are_left_alone():
    record = logging.LogRecord(
        "taxverity.test", logging.INFO, __file__, 1, "%d of %d", (3, 537), None
    )
    RedactingFilter().filter(record)
    assert record.getMessage() == "3 of 537"


def test_a_pan_never_reaches_an_emitted_record(root_logger, capsys):
    configure_logging(Settings(log_level="INFO"))
    get_logger("taxverity.corpus.test").info("extracted facts for %s", PAN)
    emitted = capsys.readouterr().err
    assert PAN not in emitted
    assert PAN_MASK in emitted


def test_json_format_emits_one_parseable_object_per_record(root_logger, capsys):
    configure_logging(Settings(log_format="json", log_level="INFO"))
    get_logger("taxverity.corpus.test").warning("aadhaar %s in section 80C", AADHAAR)
    payload = json.loads(capsys.readouterr().err.strip())
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "taxverity.corpus.test"
    assert payload["message"] == f"aadhaar {AADHAAR_MASK} in section 80C"
    assert set(payload) == {"ts", "level", "logger", "message"}


def test_json_format_carries_an_exception(root_logger, capsys):
    configure_logging(Settings(log_format="json"))
    try:
        raise ValueError("bad marker")
    except ValueError:
        get_logger("taxverity.corpus.test").exception("build failed")
    payload = json.loads(capsys.readouterr().err.strip())
    assert "ValueError: bad marker" in payload["exc"]


def test_the_level_honours_settings(root_logger, capsys):
    configure_logging(Settings(log_level="WARNING"))
    logger = get_logger("taxverity.corpus.test")
    logger.info("not emitted")
    logger.warning("emitted")
    emitted = capsys.readouterr().err
    assert "not emitted" not in emitted
    assert "emitted" in emitted


def test_configure_logging_is_idempotent(root_logger, capsys):
    for _ in range(3):
        configure_logging(Settings())
    assert len(root_logger.handlers) == 1
    get_logger("taxverity.corpus.test").info("once")
    assert capsys.readouterr().err.count("once") == 1


def test_diagnostics_do_not_escape_to_the_root_logger(root_logger):
    configure_logging(Settings())
    assert root_logger.propagate is False


def test_every_source_module_logs_through_get_logger():
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        path.relative_to(src).as_posix()
        for path in src.rglob("*.py")
        if "print(" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_the_json_formatter_needs_no_configuration_to_be_used_directly():
    record = logging.LogRecord(
        "taxverity.test", logging.ERROR, __file__, 1, "pan %s", (PAN,), None
    )
    RedactingFilter().filter(record)
    assert json.loads(JsonFormatter().format(record))["message"] == f"pan {PAN_MASK}"


def test_a_script_logger_is_adopted_into_the_taxverity_tree(root_logger, capsys):
    configure_logging(Settings())
    get_logger("__main__").info("driver started")
    assert "taxverity.__main__" in capsys.readouterr().err


def test_a_module_logger_keeps_its_own_name():
    assert get_logger("taxverity.corpus.loader").name == "taxverity.corpus.loader"
    assert get_logger(ROOT_LOGGER_NAME).name == ROOT_LOGGER_NAME
