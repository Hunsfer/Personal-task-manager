"""JSON logging that keeps message text out of the logs.

The formatter is deliberately restrictive. It writes the log call's event name, without applying
%-style arguments, and only the extra fields listed in ALLOWED_FIELDS. For exceptions it writes the
type and stack, never str(exc), because exception messages can quote the data being processed.

Log a fixed event name and pass data through `extra`, for example:
    logger.info("message_saved", extra={"chat_id": chat_id, "msg_id": msg_id})
An f-string event name bypasses this protection, so never build one from message content.
"""

import json
import logging
import sys
import traceback
from datetime import UTC, datetime

ALLOWED_FIELDS = frozenset(
    {"chat_id", "msg_id", "status", "count", "revision", "seconds", "reason"}
)


class SafeJsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "event": record.msg if isinstance(record.msg, str) else type(record.msg).__name__,
        }
        for field in ALLOWED_FIELDS:
            if field in record.__dict__:
                payload[field] = record.__dict__[field]
        if record.exc_info and record.exc_info[0] is not None:
            exc_type, _, exc_traceback = record.exc_info
            payload["exc_type"] = exc_type.__name__
            payload["stack"] = "".join(traceback.format_tb(exc_traceback))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeJsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
