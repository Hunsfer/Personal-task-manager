import json
import logging
import sys

from task_manager.common.log import SafeJsonFormatter

SECRET = "Иван Петров, позвони на +79001234567"


def _format(**kwargs: object) -> str:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=kwargs.pop("msg", "event_name"),
        args=kwargs.pop("args", ()),
        exc_info=kwargs.pop("exc_info", None),
    )
    record.__dict__.update(kwargs)
    return SafeJsonFormatter("collector").format(record)


def test_output_is_json_with_allowed_fields() -> None:
    entry = json.loads(_format(chat_id=-100, msg_id=7))
    assert entry["event"] == "event_name"
    assert entry["service"] == "collector"
    assert entry["chat_id"] == -100
    assert entry["msg_id"] == 7


def test_unlisted_extra_fields_are_dropped() -> None:
    output = _format(text=SECRET, message_text=SECRET)
    assert SECRET not in output
    assert "message_text" not in output


def test_percent_args_are_not_applied() -> None:
    output = _format(msg="message_saved %s", args=(SECRET,))
    assert SECRET not in output
    assert json.loads(output)["event"] == "message_saved %s"


def test_non_string_message_is_replaced_by_its_type() -> None:
    assert SECRET not in _format(msg=ValueError(SECRET))


def test_exception_message_is_dropped_but_stack_is_kept() -> None:
    try:
        raise ValueError(SECRET)
    except ValueError:
        output = _format(exc_info=sys.exc_info())
    entry = json.loads(output)
    assert SECRET not in output
    assert entry["exc_type"] == "ValueError"
    assert "test_exception_message" in entry["stack"]
