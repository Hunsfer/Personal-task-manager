"""Each service records that it's alive, so the bot can tell the owner when one isn't.

The heartbeat file only serves Docker's healthcheck and is private to its container. A service's
state in the database is what another service can see. It holds no message text.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from task_manager.storage.models import ServiceState

STATE_INTERVAL_S = 30.0

logger = logging.getLogger(__name__)


async def record_state(
    sessions: async_sessionmaker[AsyncSession], service: str, detail: str | None = None
) -> None:
    now = datetime.now(UTC)
    async with sessions() as session:
        statement = insert(ServiceState).values(service=service, last_seen=now, detail=detail)
        await session.execute(
            statement.on_conflict_do_update(
                index_elements=[ServiceState.service],
                set_={"last_seen": now, "detail": detail},
            )
        )
        await session.commit()


async def state_loop(
    sessions: async_sessionmaker[AsyncSession],
    service: str,
    finished: asyncio.Event,
    detail: Callable[[], str | None] = lambda: None,
) -> None:
    """Ends on an event rather than by cancellation, because it uses the database.

    A database hiccup must not take the service down with it: the owner would rather see one
    missed report than a restart.
    """
    while not finished.is_set():
        try:
            await record_state(sessions, service, detail())
        except Exception:
            logger.exception("service_state_failed")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(finished.wait(), STATE_INTERVAL_S)
