"""The review bot service."""

import asyncio
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from task_manager.common.config import ConfigError
from task_manager.monitoring import state_loop
from task_manager.service import beat
from task_manager.storage.db import current_revision
from task_manager.storage.models import Person

logger = logging.getLogger(__name__)


async def check_owner(engine: AsyncEngine, owner_id: int) -> None:
    """A wrong ID would send real names to someone else, so it stops the service."""
    async with AsyncSession(engine) as session:
        me = await session.scalar(select(Person.user_id).where(Person.is_me.is_(True)))
    if me is not None and me != owner_id:
        raise ConfigError("BOT_OWNER_ID doesn't match the account the collector reads")


async def run_until_stopped(
    engine: AsyncEngine,
    sessions,
    owner_id: int,
    messenger,
    controller,
    outbox,
    heartbeat: Path,
    stop,
) -> None:
    """Fails if the database isn't migrated, so the container restarts instead of idling."""
    await current_revision(engine)
    await check_owner(engine, owner_id)
    finished = asyncio.Event()
    beating = asyncio.create_task(beat(heartbeat, finished))
    logger.info("review_bot_ready")
    # The first report comes after the outbox has read the previous one, to measure the downtime.
    await outbox.flush()
    reporting = asyncio.create_task(state_loop(sessions, "review_bot", finished))
    tasks = [
        asyncio.create_task(outbox.run(stop)),
        asyncio.create_task(messenger.run(controller, stop)),
    ]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()
    finally:
        # Everything ends on the event rather than by cancellation: see the database note on
        # aiosqlite in the project's documentation.
        stop.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        finished.set()
        await beating
        await reporting
        heartbeat.unlink(missing_ok=True)
