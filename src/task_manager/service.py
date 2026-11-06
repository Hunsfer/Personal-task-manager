"""Service lifecycle shared by all containers: startup checks and heartbeat."""

import asyncio
import contextlib
import signal
from pathlib import Path

SERVICES = ("collector", "anonymizer", "analyzer", "review_bot", "jira_export")
HEARTBEAT_INTERVAL_S = 10.0


def stop_on_signals(stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        # Signal handlers aren't available on Windows, where this only runs in tests.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_number, stop.set)


async def beat(path: Path, finished: asyncio.Event) -> None:
    """Refresh a heartbeat file until told to finish.

    It ends on an event rather than by cancellation, and touches only a file, so it is safe next to
    work that outlasts the heartbeat's limit.
    """
    while not finished.is_set():
        path.touch()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(finished.wait(), HEARTBEAT_INTERVAL_S)
