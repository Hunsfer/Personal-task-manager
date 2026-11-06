"""The check Docker runs in every container every 15 seconds.

Standard library only, and nothing heavy from it: this process starts so often that importing the
application here would be most of the system's idle CPU use. A bad configuration needs no check
here, since the service then stops at startup and never writes a heartbeat.
"""

import os
import sys
import time
from pathlib import Path

EXIT_OK = 0
EXIT_FAILURE = 1
HEARTBEAT_MAX_AGE_S = 30.0


def heartbeat_path(heartbeat_dir: Path, service: str) -> Path:
    return heartbeat_dir / f"{service}.heartbeat"


def is_healthy(path: Path, now: float | None = None) -> bool:
    """A service is healthy if it refreshed its heartbeat recently."""
    try:
        age = (time.time() if now is None else now) - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age <= HEARTBEAT_MAX_AGE_S


def main(arguments: list[str]) -> int:
    directory = os.environ.get("HEARTBEAT_DIR")
    if len(arguments) != 1 or not directory:
        return EXIT_FAILURE
    healthy = is_healthy(heartbeat_path(Path(directory), arguments[0]))
    return EXIT_OK if healthy else EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
