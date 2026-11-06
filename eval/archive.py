"""Keeps the labelled messages and answers encrypted while they aren't in use.

    uv run python eval/archive.py seal     # eval/dataset and eval/results into eval/archive.db
    uv run python eval/archive.py open     # back out, to compare prompts

`tm eval seal` and `tm eval open` run this with the database key. Sealing deletes the plain files.
"""

import argparse
import sys
from pathlib import Path

from task_manager.storage.db import connect_raw

ROOT = Path(__file__).parent
ARCHIVE = ROOT / "archive.db"
FOLDERS = ("dataset", "results")


def seal(archive: Path = ARCHIVE, root: Path = ROOT) -> int:
    """The plain files go only after every one of them has been read back from the archive."""
    files = [
        path
        for folder in FOLDERS
        if (root / folder).is_dir()
        for path in (root / folder).rglob("*")
    ]
    files = [path for path in files if path.is_file()]
    connection = connect_raw(archive)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS files (name TEXT PRIMARY KEY, data BLOB)")
        for path in files:
            connection.execute(
                "INSERT OR REPLACE INTO files VALUES (?, ?)",
                (path.relative_to(root).as_posix(), path.read_bytes()),
            )
        connection.commit()
        for path in files:
            stored = connection.execute(
                "SELECT data FROM files WHERE name = ?", (path.relative_to(root).as_posix(),)
            ).fetchone()
            if stored is None or bytes(stored[0]) != path.read_bytes():
                raise ValueError("a file didn't reach the archive intact; nothing was deleted")
    finally:
        connection.close()
    for path in files:
        path.unlink()
    return len(files)


def open_archive(archive: Path = ARCHIVE, root: Path = ROOT) -> int:
    connection = connect_raw(archive, read_only=True)
    try:
        rows = connection.execute("SELECT name, data FROM files").fetchall()
    finally:
        connection.close()
    for name, data in rows:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "open"))
    args = parser.parse_args()
    if args.command == "seal":
        print(f"запечатано файлов: {seal()}")
    else:
        print(f"извлечено файлов: {open_archive()}. После работы выполните seal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
