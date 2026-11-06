"""Alembic environment. The database path comes from the application settings."""

from alembic import context

from task_manager.common.config import load_settings
from task_manager.storage.db import make_sync_engine
from task_manager.storage.models import Base


def run_migrations() -> None:
    if context.is_offline_mode():
        raise RuntimeError("offline migrations are not supported")

    database_path = load_settings().database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)

    engine = make_sync_engine(database_path)
    try:
        with engine.connect() as connection:
            # A batch migration drops and recreates a table. With foreign keys on, that drop runs
            # ON DELETE CASCADE and silently empties every table pointing at it.
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            connection.commit()
            # render_as_batch: SQLite can't alter most constraints in place.
            context.configure(
                connection=connection,
                target_metadata=Base.metadata,
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
                broken = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise RuntimeError(f"migration broke {len(broken)} foreign key references")
    finally:
        engine.dispose()


run_migrations()
